// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction, 
// disclosure or distribution of this material and related documentation 
// without an express license agreement from NVIDIA CORPORATION or 
// its affiliates is strictly prohibited.

import utils;

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////

[CudaKernel]
void xfm_fwd_kernel(TensorView<float3> points, TensorView<float> matrix, bool is_point, TensorView<float4> out)
{
    uint3 threadIdx = cudaThreadIdx();
    uint3 blockIdx = cudaBlockIdx();
    uint3 blockDim = cudaBlockDim();

    uint px = blockIdx.x * blockDim.x + threadIdx.x;
    uint pz = blockIdx.z * blockDim.z + threadIdx.z;

    groupshared float4x4 mtx;
    if (threadIdx.x < 16)
    {
        mtx[threadIdx.x % 4][threadIdx.x / 4] = matrix[uint3(pz, threadIdx.x / 4, threadIdx.x % 4)];
    }
    GroupMemoryBarrierWithGroupSync();

    if (px >= points.size(1))
        return;

    float3 pos = broadcast_fetch(points, uint2(pz, px));
    let v = is_point ? float4(pos, 1.f) : float4(pos, 0.f);
    out[uint2( pz, px)] = mul(v, mtx);
}

[CudaKernel]
void xfm_bwd_kernel(TensorView<float3> points, TensorView<float> matrix, bool is_point, TensorView<float4> out_grad, TensorView<float3> points_grad)
{
    uint3 threadIdx = cudaThreadIdx();
    uint3 blockIdx = cudaBlockIdx();
    uint3 blockDim = cudaBlockDim();

    uint px = blockIdx.x * blockDim.x + threadIdx.x;
    uint pz = blockIdx.z * blockDim.z + threadIdx.z;

    groupshared float4x4 mtx;
    if (threadIdx.x < 16)
    {
        mtx[threadIdx.x % 4][threadIdx.x / 4] = matrix[uint3(pz, threadIdx.x / 4, threadIdx.x % 4)];
    }
    GroupMemoryBarrierWithGroupSync();

    if (px >= points.size(1))
        return;

    float3 pos = broadcast_fetch(points, uint2(pz, px));
    float4 d_out = out_grad[uint2(pz, px)];
    let v = is_point ? float4(pos, 1.f) : float4(pos, 0.f);

    var dp_v = diffPair(v);
    var dp_mtx = diffPair(mtx);
    __bwd_diff(mul)(dp_v, dp_mtx, d_out);

    broadcast_store(points_grad, uint2(pz, px), dp_v.d.xyz);
}

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////

[TorchEntryPoint]
TorchTensor<float4> xfm_fwd(
    TorchTensor<float3> points,
    TorchTensor<float> mymatrix,
    bool is_points)
{
    uint3 dims = uint3(points.size(1), 1, mymatrix.size(0));
    let blockSize = uint3(8 * 8, 1, 1);
    let warpSize = getWarpSize(blockSize);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var result = TorchTensor<float4>.alloc(mymatrix.size(0), points.size(1));
    __dispatch_kernel(xfm_fwd_kernel, blockCount, blockSize)(points, mymatrix, is_points, result);
    return result;
}

[TorchEntryPoint]
TorchTensor<float3> xfm_bwd(
    TorchTensor<float3> points,
    TorchTensor<float> mymatrix,
    TorchTensor<float4> grad_out,
    bool is_points)
{
    uint3 dims = uint3(points.size(1), 1, mymatrix.size(0));
    let blockSize = uint3(8 * 8, 1, 1);
    let warpSize = getWarpSize(blockSize);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var points_grad = TorchTensor<float3>.alloc(mymatrix.size(0), points.size(1));
    __dispatch_kernel(xfm_bwd_kernel, blockCount, blockSize)(points, mymatrix, is_points, grad_out, points_grad);
    return points_grad;
}