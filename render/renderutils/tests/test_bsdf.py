# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

import os
import sys
sys.path.insert(0, os.path.join(sys.path[0], '../..'))
import renderutils as ru

RES = 4

def relative_loss(name, ref, cuda):
	ref = ref.float()
	cuda = cuda.float()
	print(name, torch.max(torch.abs(ref - cuda) / torch.abs(ref + 1e-7)).item())

def test_normal():
	pos_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	pos_ref = pos_cuda.clone().detach().requires_grad_(True)
	view_pos_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	view_pos_ref = view_pos_cuda.clone().detach().requires_grad_(True)
	perturbed_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	perturbed_nrm_ref = perturbed_nrm_cuda.clone().detach().requires_grad_(True)
	smooth_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	smooth_nrm_ref = smooth_nrm_cuda.clone().detach().requires_grad_(True)
	smooth_tng_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	smooth_tng_ref = smooth_tng_cuda.clone().detach().requires_grad_(True)
	geom_nrm_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	geom_nrm_ref = geom_nrm_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda')

	ref = ru.prepare_shading_normal(pos_ref, view_pos_ref, perturbed_nrm_ref, smooth_nrm_ref, smooth_tng_ref, geom_nrm_ref, True, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.prepare_shading_normal(pos_cuda, view_pos_cuda, perturbed_nrm_cuda, smooth_nrm_cuda, smooth_tng_cuda, geom_nrm_cuda, True)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    bent normal")
	print("-------------------------------------------------------------")
	relative_loss("res:", ref, cuda)
	relative_loss("pos:", pos_ref.grad, pos_cuda.grad)
	relative_loss("view_pos:", view_pos_ref.grad, view_pos_cuda.grad)
	relative_loss("perturbed_nrm:", perturbed_nrm_ref.grad, perturbed_nrm_cuda.grad)
	relative_loss("smooth_nrm:", smooth_nrm_ref.grad, smooth_nrm_cuda.grad)
	relative_loss("smooth_tng:", smooth_tng_ref.grad, smooth_tng_cuda.grad)
	relative_loss("geom_nrm:", geom_nrm_ref.grad, geom_nrm_cuda.grad)

def test_pbr_bsdf():
	kd_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	kd_ref = kd_cuda.clone().detach().requires_grad_(True)
	arm_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	arm_ref = arm_cuda.clone().detach().requires_grad_(True)
	pos_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	pos_ref = pos_cuda.clone().detach().requires_grad_(True)
	nrm_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	nrm_ref = nrm_cuda.clone().detach().requires_grad_(True)
	view_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	view_ref = view_cuda.clone().detach().requires_grad_(True)
	light_cuda = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda', requires_grad=True)
	light_ref = light_cuda.clone().detach().requires_grad_(True)
	target = torch.rand(1, RES, RES, 3, dtype=torch.float32, device='cuda')

	ref = ru.pbr_bsdf(kd_ref, arm_ref, pos_ref, nrm_ref, view_ref, light_ref, use_python=True)
	ref_loss = torch.nn.MSELoss()(ref, target)
	ref_loss.backward()

	cuda = ru.pbr_bsdf(kd_cuda, arm_cuda, pos_cuda, nrm_cuda, view_cuda, light_cuda)
	cuda_loss = torch.nn.MSELoss()(cuda, target)
	cuda_loss.backward()

	print("-------------------------------------------------------------")
	print("    Pbr BSDF")
	print("-------------------------------------------------------------")

	relative_loss("res:", ref, cuda)
	if kd_ref.grad is not None:
		relative_loss("kd:", kd_ref.grad, kd_cuda.grad)
	if arm_ref.grad is not None:
		relative_loss("arm:", arm_ref.grad, arm_cuda.grad)
	if pos_ref.grad is not None:
		relative_loss("pos:", pos_ref.grad, pos_cuda.grad)
	if nrm_ref.grad is not None:
		relative_loss("nrm:", nrm_ref.grad, nrm_cuda.grad)
	if view_ref.grad is not None:
		relative_loss("view:", view_ref.grad, view_cuda.grad)
	if light_ref.grad is not None:
		relative_loss("light:", light_ref.grad, light_cuda.grad)

test_normal()
test_pbr_bsdf()
