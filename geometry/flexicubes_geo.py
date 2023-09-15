# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited

import numpy as np
import torch

from render import mesh
from render import render
from render import util

from geometry.flexicubes import FlexiCubes

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff


###############################################################################
#  Geometry interface
###############################################################################

class FlexiCubesGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(FlexiCubesGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.flexicubes    = FlexiCubes()
        verts, indices     = self.flexicubes.construct_voxel_grid(grid_res) 

        n_cubes = indices.shape[0]
        per_cube_weights = torch.ones((n_cubes, 21),dtype=torch.float,device='cuda')

        self.verts    = verts * scale
        self.indices  = indices
        print("FlexiCubes grid min/max", torch.min(self.verts).item(), torch.max(self.verts).item())
        self.generate_edges()

        # Random init
        sdf = torch.rand_like(self.verts[:,0]) - 0.1

        self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.per_cube_weights = torch.nn.Parameter(torch.ones_like(per_cube_weights), requires_grad=True)
        self.register_parameter('weight', self.per_cube_weights)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    @torch.no_grad()
    def generate_edges(self):
        with torch.no_grad():
            edges = self.flexicubes.cube_edges
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)
            self.max_displacement = util.length(self.verts[self.all_edges[:, 0]] - self.verts[self.all_edges[:, 1]]).mean() / 4

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    @torch.no_grad()
    def map_uv2(self, faces):
        uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float, device='cuda')
        uv_idx = torch.tensor([0,1,2], dtype=torch.long, device='cuda').repeat(faces.shape[0],1)
        return uvs, uv_idx

    @torch.no_grad()
    def map_uv(self, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='floor')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, N, rounding_mode='floor'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    def getMesh(self, material, _training=False):

        # Run FlexiCubes to get a base mesh
        v_deformed = self.verts + self.max_displacement * torch.tanh(self.deform)
        verts, faces, reg_loss = self.flexicubes(v_deformed, self.sdf, self.indices, self.grid_res, 
                            self.per_cube_weights[:,:12], self.per_cube_weights[:,12:20], self.per_cube_weights[:,20],
                            training=_training)

        self.flexi_reg_loss = reg_loss.mean()

        face_gidx = torch.arange(faces.shape[0], dtype=torch.long, device="cuda")
        uvs, uv_idx = self.map_uv(face_gidx, faces.shape[0])

        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None, training = False):
        opt_mesh = self.getMesh(opt_material, training)
        return render.render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
                                        msaa=True, background=target['background'], bsdf=bsdf)


    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target, lgt, opt_material, training=True)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.l1_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])*5.0 
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01)*min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight # Dropoff to 0.01

        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 0.001 * min(1.0, iteration / 500)

        # FlexiCubes reg loss
        reg_loss += self.flexi_reg_loss* 0.25

        # Light white balance regularizer
        reg_loss = reg_loss + lgt.regularizer() * 0.005

        return img_loss, reg_loss

