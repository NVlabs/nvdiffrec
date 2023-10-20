# Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import slangpy
import os
import sys
import torch
import torch.utils.cpp_extension

from .loss import *
from .bsdf import *

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)

slang_normal = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "normal.slang"))

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = slang_normal.prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return slang_normal.prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm.contiguous(), smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Slang pbr shader

slang_module = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "pbr.slang")) 

class _pbr_bsdf_slang_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        out = slang_module.pbr_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        return slang_module.pbr_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, dout) + (None, None)

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
    
    Returns:
        Shaded color.
    '''    

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
    else:
        out = _pbr_bsdf_slang_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

slang_cubemap = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "cubemap.slang"))

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = slang_cubemap.diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = slang_cubemap.diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = slang_cubemap.specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = slang_cubemap.specular_cubemap_bwd(cubemap, bounds, ctx.roughness, ctx.theta_cutoff, dout)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = slang_cubemap.specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]

#----------------------------------------------------------------------------
# Fast image loss function

def strToLoss(s):
    if s == "mse":
        return 1 #LOSS_MSE;
    elif s == "relmse":
        return 2 #LOSS_RELMSE;
    elif s == "smape":
        return 3 #LOSS_SMAPE;
    else:
        return 0 #LOSS_L1;

def strToTonemapper(s):
    if s == "log_srgb":
        return 1 # TONEMAPPER_LOG_SRGB
    else:
        return 0 # TONEMAPPER_NONE

slang_loss = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "loss.slang"))

class _image_loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss, ctx.tonemapper = loss, tonemapper
        ctx.save_for_backward(img, target)
        out = slang_loss.loss_fwd(img, target, strToLoss(loss), strToTonemapper(tonemapper))
        return out

    @staticmethod
    def backward(ctx, dout):
        img, target = ctx.saved_variables
        return slang_loss.loss_bwd(img, target, strToLoss(ctx.loss), strToTonemapper(ctx.tonemapper), dout.contiguous()) + (None, None, None)

def image_loss(img, target, loss='l1', tonemapper='none', use_python=False):
    '''Compute HDR image loss. Combines tonemapping and loss into a single kernel for better perf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        img: Input image.
        target: Target (reference) image. 
        loss: Type of loss. Valid options are ['l1', 'mse', 'smape', 'relmse']
        tonemapper: Tonemapping operations. Valid options are ['none', 'log_srgb']
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Image space loss (scalar value).
    '''
    if use_python:
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _image_loss_func.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0]*img.shape[1]*img.shape[2])

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Transform points function

slang_mesh = slangpy.loadModule(os.path.join(os.path.dirname(__file__), "mesh.slang"))
class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return slang_mesh.xfm_fwd(points, matrix, isPoints)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (slang_mesh.xfm_bwd(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)[..., 0:3].contiguous()

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out



