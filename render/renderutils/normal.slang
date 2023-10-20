// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction, 
// disclosure or distribution of this material and related documentation 
// without an express license agreement from NVIDIA CORPORATION or 
// its affiliates is strictly prohibited.

import utils;

//////////////////////////////////////////////////////////////////////////
// Perturb shading normal by tangent frame
//////////////////////////////////////////////////////////////////////////

[BackwardDifferentiable]
float3 perturbNormal(const float3 perturbed_nrm, const float3 smooth_nrm, const float3 smooth_tng, no_diff bool opengl)
{
    float3 _smooth_bitng = cross(smooth_tng, smooth_nrm);
    float3 smooth_bitng = safeNormalize(_smooth_bitng);
    float3 _shading_nrm = smooth_tng * perturbed_nrm.x + smooth_bitng * perturbed_nrm.y * (opengl ? -1.0f : 1.0f) + smooth_nrm * max(perturbed_nrm.z, 0.0f);
    return safeNormalize(_shading_nrm);
}

//////////////////////////////////////////////////////////////////////////
// Bent normals
//////////////////////////////////////////////////////////////////////////

[BackwardDifferentiable]
float3 bendNormal(const float3 view_vec, const float3 smooth_nrm, const float3 geom_nrm)
{
    const float NORMAL_THRESHOLD = 0.1f;
    float dp = dot(view_vec, smooth_nrm);
    float t = clamp(dp / NORMAL_THRESHOLD, 0.0f, 1.0f);
    return geom_nrm * (1.0f - t) + smooth_nrm * t;
}

//////////////////////////////////////////////////////////////////////////
// Prepare shading normal
//////////////////////////////////////////////////////////////////////////

[BackwardDifferentiable]
float3 prepareShadingNormal(const float3 _smooth_nrm, const float3 _smooth_tng,
                            const float3 view_pos, const float3 pos,
                            const float3 geom_nrm, const float3 perturbed_nrm,
                            no_diff bool opengl, no_diff bool two_sided_shading)
{
    float3 smooth_nrm = safeNormalize(_smooth_nrm);
    float3 smooth_tng = safeNormalize(_smooth_tng);
    float3 view_vec = safeNormalize(view_pos - pos);
    float3 shading_nrm = perturbNormal(perturbed_nrm, smooth_nrm, smooth_tng, opengl);

    float3 res;
    if (two_sided_shading && dot(view_vec, geom_nrm) < 0.0f)
        res = bendNormal(view_vec, -shading_nrm, -geom_nrm);
    else
        res = bendNormal(view_vec, shading_nrm, geom_nrm);

    return res;
}

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////

[CudaKernel]
void prepareShadingNormal_fwd_kernel(
    TensorView<float3> smooth_nrm,
    TensorView<float3> smooth_tng,
    TensorView<float3> view_pos,
    TensorView<float3> pos,
    TensorView<float3> geom_nrm,
    TensorView<float3> perturbed_nrm,
    bool opengl, bool two_sided_shading,
    TensorView<float3> output)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > output.size(2) || idx.y > output.size(1) || idx.z > output.size(0))
        return;

    uint3 idxr = uint3(idx.z, idx.y, idx.x);
    float3 res = prepareShadingNormal(smooth_nrm[idxr], smooth_tng[idxr], broadcast_fetch(view_pos, idxr),
                                      pos[idxr], geom_nrm[idxr], broadcast_fetch(perturbed_nrm, idxr),
                                      opengl, two_sided_shading);
    output[idxr] = res;
}

[CudaKernel]
void prepareShadingNormal_bwd_kernel(
    TensorView<float3> smooth_nrm,
    TensorView<float3> smooth_tng,
    TensorView<float3> view_pos,
    TensorView<float3> pos,
    TensorView<float3> geom_nrm,
    TensorView<float3> perturbed_nrm,
    bool opengl, bool two_sided_shading,
    TensorView<float3> smooth_nrm_grad,
    TensorView<float3> smooth_tng_grad,
    TensorView<float3> view_pos_grad,
    TensorView<float3> pos_grad,
    TensorView<float3> geom_nrm_grad,
    TensorView<float3> perturbed_nrm_grad,
    TensorView<float3> grad_out)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x > grad_out.size(2) || idx.y > grad_out.size(1) || idx.z > grad_out.size(0))
        return;

    uint3 idxr = uint3(idx.z, idx.y, idx.x);

    var dp_smooth_nrm = diffPair(smooth_nrm[idxr]);
    var dp_smooth_tng = diffPair(smooth_tng[idxr]);
    var dp_view_pos = diffPair(broadcast_fetch(view_pos, idxr));
    var dp_pos = diffPair(pos[idxr]);
    var dp_geom_nrm = diffPair(geom_nrm[idxr]);
    var dp_perturbed_nrm = diffPair(broadcast_fetch(perturbed_nrm, idxr));
    var dout = grad_out[idxr];

    __bwd_diff(prepareShadingNormal)(dp_smooth_nrm, dp_smooth_tng, dp_view_pos, dp_pos,
                                     dp_geom_nrm, dp_perturbed_nrm, opengl, two_sided_shading, dout);

    smooth_nrm_grad[idxr] = dp_smooth_nrm.d;
    smooth_tng_grad[idxr] = dp_smooth_tng.d;
    broadcast_store(view_pos_grad, idxr, dp_view_pos.d); 
    pos_grad[idxr] = dp_pos.d;
    geom_nrm_grad[idxr] = dp_geom_nrm.d;
    broadcast_store(perturbed_nrm_grad, idxr, dp_perturbed_nrm.d);
}

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////

[TorchEntryPoint]
TorchTensor<float3> prepare_shading_normal_fwd(
    TorchTensor<float3> pos,
    TorchTensor<float3> view_pos,
    TorchTensor<float3> perturbed_nrm,
    TorchTensor<float3> smooth_nrm,
    TorchTensor<float3> smooth_tng,
    TorchTensor<float3> geom_nrm,
    const bool two_sided_shading,
    const bool opengl)
{
    var result = TorchTensor<float3>.emptyLike(smooth_nrm);
    uint3 dims = uint3(result.size(2), result.size(1), result.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(prepareShadingNormal_fwd_kernel, blockCount, blockSize)(smooth_nrm, smooth_tng, view_pos,
                                                                              pos, geom_nrm, perturbed_nrm, opengl, two_sided_shading, result);
    return result;
}

struct MyReturnType
{
    TorchTensor<float3> pos_grad;
    TorchTensor<float3> view_pos_grad;
    TorchTensor<float3> perturbed_nrm_grad;
    TorchTensor<float3> smooth_nrm_grad;
    TorchTensor<float3> smooth_tng_grad;
    TorchTensor<float3> geom_nrm_grad;
}

[TorchEntryPoint]
MyReturnType prepare_shading_normal_bwd(
        TorchTensor<float3> pos,
        TorchTensor<float3> view_pos,
        TorchTensor<float3> perturbed_nrm,
        TorchTensor<float3> smooth_nrm,
        TorchTensor<float3> smooth_tng,
        TorchTensor<float3> geom_nrm,
        TorchTensor<float3> grad_out,
    const bool two_sided_shading,
    const bool opengl)
{
    var smooth_nrm_grad = TorchTensor<float3>.emptyLike(smooth_nrm);
    var smooth_tng_grad = TorchTensor<float3>.emptyLike(smooth_tng);
    var view_pos_grad = TorchTensor<float3>.emptyLike(view_pos);
    var pos_grad = TorchTensor<float3>.emptyLike(pos);
    var geom_nrm_grad = TorchTensor<float3>.emptyLike(geom_nrm);
    var perturbed_nrm_grad = TorchTensor<float3>.emptyLike(perturbed_nrm);

    uint3 dims = uint3(grad_out.size(2), grad_out.size(1), grad_out.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);

    __dispatch_kernel(prepareShadingNormal_bwd_kernel, blockCount, blockSize)(smooth_nrm, smooth_tng, view_pos,
                                                             pos, geom_nrm, perturbed_nrm, opengl, two_sided_shading, 
                                                             smooth_nrm_grad, smooth_tng_grad, view_pos_grad, pos_grad, 
                                                             geom_nrm_grad, perturbed_nrm_grad, grad_out);

    MyReturnType rt;
    rt.smooth_nrm_grad = smooth_nrm_grad;
    rt.smooth_tng_grad = smooth_tng_grad;
    rt.view_pos_grad = view_pos_grad;
    rt.pos_grad = pos_grad;
    rt.geom_nrm_grad = geom_nrm_grad;
    rt.perturbed_nrm_grad = perturbed_nrm_grad;

    return rt;
}

