// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction, 
// disclosure or distribution of this material and related documentation 
// without an express license agreement from NVIDIA CORPORATION or 
// its affiliates is strictly prohibited.

import utils;

//------------------------------------------------------------------------
// Tonemapping

[BackwardDifferentiable]
float sRGB(float x)
{
    return x > 0.0031308f ? pow(max(x, 0.0031308f), 1.0f / 2.4f) * 1.055f - 0.055f : 12.92f * max(x, 0.0f);
}


[BackwardDifferentiable]
float3 tonemapLogSRGB(float3 x)
{
    return float3(sRGB(log(x.x + 1.0f)), sRGB(log(x.y + 1.0f)), sRGB(log(x.z + 1.0f)));
}

[BackwardDifferentiable]
float relMSE(float img, float target, float eps = 0.1f)
{
    return (img - target) * (img - target) / (img * img + target * target + eps);
}

[BackwardDifferentiable]
float SMAPE(float img, float target, float eps = 0.01f)
{
    return abs(img - target) / (img + target + eps);
}

enum TonemapperType
{
    TONEMAPPER_NONE = 0,
    TONEMAPPER_LOG_SRGB = 1
};

enum LossType
{
    LOSS_L1 = 0,
    LOSS_MSE = 1,
    LOSS_RELMSE = 2,
    LOSS_SMAPE = 3
};

[BackwardDifferentiable]
float loss_func(float3 img, float3 target, LossType loss, TonemapperType tonemapper)
{
    if (tonemapper == TonemapperType::TONEMAPPER_LOG_SRGB)
    {
        img = tonemapLogSRGB(clamp(img, 0.0f, 65535.0f));
        target = tonemapLogSRGB(clamp(target, 0.0f, 65535.0f));
    }

    var vloss = float3(0);

    if (loss == LossType::LOSS_MSE)
        vloss = (img - target) * (img - target);
    else if (loss == LossType::LOSS_RELMSE)
        vloss = float3(relMSE(img.x, target.x), relMSE(img.y, target.y), relMSE(img.z, target.z));
    else if (loss == LossType::LOSS_SMAPE)
        vloss = float3(SMAPE(img.x, target.x), SMAPE(img.y, target.y), SMAPE(img.z, target.z));
    else
        vloss = float3(abs(img.x - target.x), abs(img.y - target.y), abs(img.z - target.z));

    return (vloss.x + vloss.y + vloss.z) / 3.0f;
}

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////

[CudaKernel]
void loss_fwd_kernel(
    TensorView<float3> img,
    TensorView<float3> target,
    LossType loss,
    TonemapperType tonemapper,
    TensorView<float> output)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    float floss = 0.0f;

    if (idx.x < img.size(2) && idx.y < img.size(1) && idx.z < img.size(0))
    {
        let idxr = uint3(idx.z, idx.y, idx.x);
        floss = loss_func(img[idxr], target[idxr], loss, tonemapper);
    }

    floss = WaveActiveSum(floss);
    uint3 threadIdx = cudaThreadIdx();
    uint3 warpSize = getWarpSize(cudaBlockDim());

    if (idx.x < img.size(2) && idx.y < img.size(1) && idx.z < img.size(0)
        && threadIdx.x % warpSize.x == 0 
        && threadIdx.y % warpSize.y == 0 && threadIdx.z % warpSize.z == 0)
    {
        let sidx = uint3(idx.x / warpSize.x, idx.y / warpSize.y, idx.z / warpSize.z);
        output[sidx] = floss;
    }
}

[CudaKernel]
void loss_bwd_kernel(
    TensorView<float3> img,
    TensorView<float3> target,
    LossType loss,
    TonemapperType tonemapper,
    TensorView<float> grad_out,
    TensorView<float3> img_grad,
    TensorView<float3> target_grad)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    uint3 threadIdx = cudaThreadIdx();
    uint3 warpSize = getWarpSize(cudaBlockDim());

    if (idx.x < img.size(2) && idx.y < img.size(1) && idx.z < img.size(0))
    {
        let idxr = uint3(idx.z, idx.y, idx.x);
        let sidx = uint3(idx.x / warpSize.x, idx.y / warpSize.y, idx.z / warpSize.z);

        var dp_img = diffPair(img[idxr]);
        var dp_target = diffPair(target[idxr]);
        var dout = grad_out[sidx];

        __bwd_diff(loss_func)(dp_img, dp_target, loss, tonemapper, dout);

        img_grad[idxr] = dp_img.d;
        target_grad[idxr] = dp_target.d;
    }
}


/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////

[TorchEntryPoint]
TorchTensor<float> loss_fwd(
    TorchTensor<float3> img,
    TorchTensor<float3> target,
    LossType loss,
    TonemapperType tonemapper)
{
    uint3 dims = uint3(img.size(2), img.size(1), img.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    let warpSize = getWarpSize(blockSize);
    var result = TorchTensor<float>.alloc((dims.x - 1) / warpSize.x + 1, (dims.y - 1) / warpSize.y + 1, (dims.z - 1) / warpSize.z + 1);

    __dispatch_kernel(loss_fwd_kernel, blockCount, blockSize)(img, target, loss, tonemapper, result);
    return result;
}

struct MyReturnType
{
    TorchTensor<float3> img_grad;
    TorchTensor<float3> target_grad;
};

[TorchEntryPoint]
MyReturnType loss_bwd(
    TorchTensor<float3> img,
    TorchTensor<float3> target,
    LossType loss,
    TonemapperType tonemapper,
    TorchTensor<float> grad_out)
{
    MyReturnType rt;
    rt.img_grad = TorchTensor<float3>.emptyLike(img);
    rt.target_grad = TorchTensor<float3>.emptyLike(target);

    uint3 dims = uint3(img.size(2), img.size(1), img.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(loss_bwd_kernel, blockCount, blockSize)(img, target, loss, tonemapper, grad_out, rt.img_grad, rt.target_grad);
    return rt;
}
