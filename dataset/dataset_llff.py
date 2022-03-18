# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import glob

import torch
import numpy as np

from render import util

from .dataset import Dataset

def _load_mask(fn):
    img = torch.tensor(util.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def _load_img(fn):
    img = util.load_image_raw(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

###############################################################################
# LLFF datasets (real world camera lightfields)
###############################################################################

class DatasetLLFF(Dataset):
    def __init__(self, base_dir, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.base_dir = base_dir
        self.examples = examples

        # Enumerate all image files and get resolution
        all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        self.resolution = _load_img(all_img[0]).shape[0:2]

        # Load camera poses
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        
        poses        = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        poses        = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Taken from nerf, swizzles from LLFF to expected coordinate system
        poses        = np.moveaxis(poses, -1, 0).astype(np.float32)
        
        lcol         = np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(poses.shape[0], 0)
        self.imvs    = torch.tensor(np.concatenate((poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        self.aspect  = self.resolution[1] / self.resolution[0] # width / height
        self.fovy    = util.focal_length_to_fovy(poses[:, 2, 4], poses[:, 0, 4])

        # Recenter scene so lookat position is origin
        center                = util.lines_focal(self.imvs[..., :3, 3], -self.imvs[..., :3, 2])
        self.imvs[..., :3, 3] = self.imvs[..., :3, 3] - center[None, ...]

        if self.FLAGS.local_rank == 0:
            print("DatasetLLFF: %d images with shape [%d, %d]" % (len(all_img), self.resolution[0], self.resolution[1]))
            print("DatasetLLFF: auto-centering at %s" % (center.cpu().numpy()))

        # Pre-load from disc to avoid slow png parsing
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.imvs.shape[0]):
                self.preloaded_data += [self._parse_frame(i)]

    def _parse_frame(self, idx):
        all_img  = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_mask = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        assert len(all_img) == self.imvs.shape[0] and len(all_mask) == self.imvs.shape[0]

        # Load image+mask data
        img  = _load_img(all_img[idx])
        mask = _load_mask(all_mask[idx])
        img  = torch.cat((img, mask[..., 0:1]), dim=-1)

        # Setup transforms
        proj   = util.perspective(self.fovy[idx, ...], self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        mv     = torch.linalg.inv(self.imvs[idx, ...])
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension

    def __len__(self):
        return self.imvs.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.imvs.shape[0]]
        else:
            img, mv, mvp, campos = self._parse_frame(itr % self.imvs.shape[0])

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.resolution,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
