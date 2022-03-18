# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import io
import zipfile
import requests

import gdown
import imageio

import numpy as np
import torch

def download_nerf_synthetic():
    TMP_ARCHIVE = "nerf_synthetic.zip"
    print("------------------------------------------------------------")
    print(" Downloading NeRF synthetic dataset")
    print("------------------------------------------------------------")
    nerf_synthetic_url = "https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=sharing"
    gdown.download(url=nerf_synthetic_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting NeRF synthetic dataset")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('nerf_synthetic/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

def download_nerd():
    res = [512, 512]
    datasets = ['ethiopianHead', 'moldGoldCape']
    for dataset in datasets:
        print("------------------------------------------------------------")
        print(" Downloading & extracting NeRD %s dataset" % dataset)
        print("------------------------------------------------------------")

        # Create output directory
        dataset_rescaled = dataset + "_rescaled"
        os.makedirs(os.path.join("nerd", dataset_rescaled), exist_ok=True)

        # Download dataset 
        wget = requests.get("https://github.com/vork/%s/archive/refs/heads/master.zip" % dataset, stream=True)
        archive = zipfile.ZipFile(io.BytesIO(wget.content))

        # Unzip all files. Rescale images to desired resolution
        for zipinfo in archive.infolist():
            out_file = zipinfo.filename.replace("%s-master/" % dataset, "%s/" % dataset_rescaled)
            ext = os.path.splitext(out_file)[1][1:]
            if ext.lower() == 'jpg':
                folder = os.path.dirname(out_file)
                img = torch.tensor(imageio.imread(archive.read(zipinfo.filename), format=ext).astype(np.float32) / 255.0)
                img = img[None, ...].permute(0, 3, 1, 2)
                rescaled_img = torch.nn.functional.interpolate(img, res, mode='area')
                rescaled_img = rescaled_img.permute(0, 2, 3, 1)[0, ...]
                os.makedirs(os.path.join("nerd", folder), exist_ok=True)
                imageio.imwrite(os.path.join("nerd", out_file), np.clip(np.rint(rescaled_img.numpy() * 255.0), 0, 255).astype(np.uint8))
            else:
                zipinfo.filename = out_file
                archive.extract(zipinfo, path="nerd")
        archive.close()

download_nerf_synthetic()
download_nerd()
print("Completed")

