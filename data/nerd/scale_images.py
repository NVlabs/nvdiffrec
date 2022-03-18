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
import imageio
import shutil

import numpy as np
import torch

res = [512, 512]

datasets = ['ethiopianHead', 'moldGoldCape']
folders  = ['images', 'masks']

for dataset in datasets:
    dataset_rescaled = dataset + "_rescaled"
    os.makedirs(dataset_rescaled, exist_ok=True)
    shutil.copyfile(os.path.join(dataset, "poses_bounds.npy"), os.path.join(dataset_rescaled, "poses_bounds.npy"))
    for folder in folders:
        os.makedirs(os.path.join(dataset_rescaled, folder), exist_ok=True)
        files = glob.glob(os.path.join(dataset, folder, '*.jpg')) + glob.glob(os.path.join(dataset, folder, '*.JPG'))
        for file in files:
            print(file)
            img = torch.tensor(imageio.imread(file).astype(np.float32) / 255.0)
            img = img[None, ...].permute(0, 3, 1, 2)
            rescaled_img = torch.nn.functional.interpolate(img, res, mode='area')
            rescaled_img = rescaled_img.permute(0, 2, 3, 1)[0, ...]
            out_file = os.path.join(dataset_rescaled, folder, os.path.basename(file))
            imageio.imwrite(out_file, np.clip(np.rint(rescaled_img.numpy() * 255.0), 0, 255).astype(np.uint8))
