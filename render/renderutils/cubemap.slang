// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction, 
// disclosure or distribution of this material and related documentation 
// without an express license agreement from NVIDIA CORPORATION or 
// its affiliates is strictly prohibited.

import utils;

float3 cube_to_dir(uint x, uint y, uint side, uint N)
{
    float fx = 2.0f * (((float)x + 0.5f) / (float)N) - 1.0f;
    float fy = 2.0f * (((float)y + 0.5f) / (float)N) - 1.0f;
    switch (side)
    {
        case 0: return safeNormalize(float3(1.f, -fy, -fx));
        case 1: return safeNormalize(float3(-1.f, -fy, fx));
        case 2: return safeNormalize(float3(fx, 1.f, fy));
        case 3: return safeNormalize(float3(fx, -1.f, -fy));
        case 4: return safeNormalize(float3(fx, -fy, 1.f));
        case 5: return safeNormalize(float3(-fx, -fy, -1.f));
    }
    return float3(0); // Unreachable
}

float ndfGGX(const float alphaSqr, const float cosTheta)
{
    float _cosTheta = clamp(cosTheta, 0.0, 1.0f);
    float d = (_cosTheta * alphaSqr - _cosTheta) * _cosTheta + 1.0f;
    return alphaSqr / (d * d * 3.14159265f);
}

// https://cgvr.cs.uni-bremen.de/teaching/cg_literatur/Spherical,%20Cubic,%20and%20Parabolic%20Environment%20Mappings.pdf
float pixel_area(int x, int y, int N)
{
    if (N > 1)
    {
        int H = N / 2;
        x = abs(x - H);
        y = abs(y - H);
        float dx = atan((float)(x + 1) / (float)H) - atan((float)x / (float)H);
        float dy = atan((float)(y + 1) / (float)H) - atan((float)y / (float)H);
        return dx * dy;
    }
    else
        return 1;
}

/////////////////////////////////////////////////////////////////////////////////
// Cuda kernels
/////////////////////////////////////////////////////////////////////////////////

[CudaKernel]
void diffuse_cubemap_fwd_kernel(
    TensorView<float3> cubemap,
    TensorView<float3> output)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= cubemap.size(2) && idx.y >= cubemap.size(1) && idx.z >= cubemap.size(0))
        return;

    let Npx = cubemap.size(1);
    let N = cube_to_dir(idx.x, idx.y, idx.z, Npx);
    var col = float3(0);

    for (int y = 0; y < Npx; ++y)
    {
        for (int x = 0; x < Npx; ++x)
        {
            [ForceUnroll]
            for (int s = 0; s < 6; ++s)
            {
                let L = cube_to_dir(x, y, s, Npx);
                float costheta = min(max(dot(N, L), 0.0f), 0.999f);
                float w = costheta * pixel_area(x, y, int(Npx)) / 3.141592f; // pi = area of positive hemisphere
                col += cubemap[uint3(s, y, x)] * w;
            }
        }
    }
    output[uint3(idx.z, idx.y, idx.x)] = col;
}

[CudaKernel]
void diffuse_cubemap_bwd_kernel(
    TensorView<float3> cubemap,
    TensorView<float3> out_grad,
    TensorView<float3> cubemap_grad)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= cubemap.size(2) && idx.y >= cubemap.size(1) && idx.z >= cubemap.size(0))
        return;

    let Npx = cubemap.size(1);
    let N = cube_to_dir(idx.x, idx.y, idx.z, Npx);
    float3 grad = out_grad[uint3(idx.z, idx.y, idx.x)];

    for (int y = 0; y < Npx; ++y)
    {
        for (int x = 0; x < Npx; ++x)
        {
            [ForceUnroll]
            for (int s = 0; s < 6; ++s)
            {
                let L = cube_to_dir(x, y, s, Npx);
                float costheta = min(max(dot(N, L), 0.0f), 0.999f);
                float w = costheta * pixel_area(x, y, int(Npx)) / 3.141592f; // pi = area of positive hemisphere
                float3 oldval;
                cubemap_grad.InterlockedAdd(uint3(s, y, x), grad * w, oldval);
            }
        }
    }
}

[CudaKernel]
void specular_cubemap_fwd_kernel(
    TensorView<float3> cubemap,
    TensorView<int4> bounds,
    float roughness,
    float costheta_cutoff,
    TensorView<float4> output)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= cubemap.size(2) && idx.y >= cubemap.size(1) && idx.z >= cubemap.size(0))
        return;

    let Npx = cubemap.size(1);
    float3 VNR = cube_to_dir(idx.x, idx.y, idx.z, Npx);

    float alpha = roughness * roughness;
    float alphaSqr = alpha * alpha;

    float wsum = 0.0f;
    var col = float3(0);

    // Loop over cubemap faces
    for (int s = 0; s < 6; ++s)
    {
        int4 bound = bounds[uint4(idx.z, idx.y, idx.x, s)];
        let xmin = bound.x;
        let xmax = bound.y;
        let ymin = bound.z;
        let ymax = bound.w;

        if (xmin <= xmax)
        {
            for (int y = ymin; y <= ymax; ++y)
            {
                for (int x = xmin; x <= xmax; ++x)
                {
                    float3 L = cube_to_dir(x, y, s, Npx);
                    if (dot(L, VNR) >= costheta_cutoff)
                    {
                        float3 H = safeNormalize(L + VNR);
                        float wiDotN = max(dot(L, VNR), 0.0f);
                        float VNRDotH = max(dot(VNR, H), 0.0f);

                        float w = wiDotN * ndfGGX(alphaSqr, VNRDotH) * pixel_area(x, y, int(Npx)) / 4.0f;
                        col += cubemap[uint3(s,y,x)] * w; 
                        wsum += w;
                    }
                }
            }
        }
    }
    output[uint3(idx.z, idx.y, idx.x)] = float4(col, wsum);
}

[CudaKernel]
void specular_cubemap_bwd_kernel(
    TensorView<float3> cubemap,
    TensorView<int4> bounds,
    float roughness,
    float costheta_cutoff,
    TensorView<float4> out_grad,
    TensorView<float3> cubemap_grad)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    if (idx.x >= cubemap.size(2) && idx.y >= cubemap.size(1) && idx.z >= cubemap.size(0))
        return;

    let Npx = cubemap.size(1);
    float3 VNR = cube_to_dir(idx.x, idx.y, idx.z, Npx);

    float3 grad = out_grad[uint3(idx.z, idx.y, idx.x)].xyz;

    float alpha = roughness * roughness;
    float alphaSqr = alpha * alpha;

    var col = float3(0);
    for (int s = 0; s < 6; ++s)
    {
        int4 bound = bounds[uint4(idx.z, idx.y, idx.x, s)];
        let xmin = bound.x;
        let xmax = bound.y;
        let ymin = bound.z;
        let ymax = bound.w;

        if (xmin <= xmax)
        {
            for (int y = ymin; y <= ymax; ++y)
            {
                for (int x = xmin; x <= xmax; ++x)
                {
                    float3 L = cube_to_dir(x, y, s, Npx);
                    if (dot(L, VNR) >= costheta_cutoff)
                    {
                        float3 H = safeNormalize(L + VNR);
                        float wiDotN = max(dot(L, VNR), 0.f);
                        float VNRDotH = max(dot(VNR, H), 0.f);
                        float w = wiDotN * ndfGGX(alphaSqr, VNRDotH) * pixel_area(x, y, int(Npx)) / 4.0f;
                        float3 dummy;
                        cubemap_grad.InterlockedAdd(uint3(s, y, x), grad * w, dummy);
                    }
                }
            }
        }        
    }
}


[CudaKernel]
void specular_bounds_kernel(
    float costheta_cutoff,
    TensorView<int4> bounds)
{
    uint3 idx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    uint3 threadIdx = cudaThreadIdx();

    if (idx.x >= bounds.size(2) && idx.y >= bounds.size(1) && idx.z >= bounds.size(0))
        return;

    let Npx = bounds.size(1);
    float3 VNR = cube_to_dir(idx.x, idx.y, idx.z, Npx);
    const int TILE_SIZE = 16;

    // Brute force entire cubemap and compute bounds for the cone
    for (int s = 0; s < bounds.size(0); ++s) // loop over cube map faces
    {
        // Assume empty BBox
        int _min_x = int(bounds.size(2)) - 1, _max_x = 0;
        int _min_y = int(bounds.size(1)) - 1, _max_y = 0;

        // For each (8x8) tile
        for (int tx = 0; tx < (int(bounds.size(2)) + TILE_SIZE - 1) / TILE_SIZE; tx++)
        {
            for (int ty = 0; ty < (int(bounds.size(1)) + TILE_SIZE - 1) / TILE_SIZE; ty++)
            {
                // Compute tile extents
                int tsx = tx * TILE_SIZE, tsy = ty * TILE_SIZE;
                int tex = min((tx + 1) * TILE_SIZE, int(bounds.size(2)));
                int tey = min((ty + 1) * TILE_SIZE, int(bounds.size(1)));

                // Use some blunt interval arithmetics to cull tiles
                float3 L0 = cube_to_dir(tsx, tsy, s, Npx), L1 = cube_to_dir(tex, tsy, s, Npx);
                float3 L2 = cube_to_dir(tsx, tey, s, Npx), L3 = cube_to_dir(tex, tey, s, Npx);

                float minx = min(min(L0.x, L1.x), min(L2.x, L3.x)), maxx = max(max(L0.x, L1.x), max(L2.x, L3.x));
                float miny = min(min(L0.y, L1.y), min(L2.y, L3.y)), maxy = max(max(L0.y, L1.y), max(L2.y, L3.y));
                float minz = min(min(L0.z, L1.z), min(L2.z, L3.z)), maxz = max(max(L0.z, L1.z), max(L2.z, L3.z));

                float maxdp = max(minx * VNR.x, maxx * VNR.x) + max(miny * VNR.y, maxy * VNR.y) + max(minz * VNR.z, maxz * VNR.z);
                if (maxdp >= costheta_cutoff)
                {
                    // Test all pixels in tile.
                    for (int y = tsy; y < tey; ++y)
                    {
                        for (int x = tsx; x < tex; ++x)
                        {
                            float3 L = cube_to_dir(x, y, s, Npx);
                            if (dot(L, VNR) >= costheta_cutoff)
                            {
                                _min_x = min(_min_x, x);
                                _max_x = max(_max_x, x);
                                _min_y = min(_min_y, y);
                                _max_y = max(_max_y, y);
                            }
                        }
                    }
                }
            }
        }
        bounds[uint4(idx.z, idx.y, idx.x, s)] = int4(_min_x, _max_x, _min_y, _max_y);
    }
}

/////////////////////////////////////////////////////////////////////////////////
// Torch entry points
/////////////////////////////////////////////////////////////////////////////////

[TorchEntryPoint]
TorchTensor<float3> diffuse_cubemap_fwd(TorchTensor<float3> cubemap)
{
    uint3 dims = uint3(cubemap.size(2), cubemap.size(1), cubemap.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var out = TorchTensor<float3>.emptyLike(cubemap);
    __dispatch_kernel(diffuse_cubemap_fwd_kernel, blockCount, blockSize)(cubemap, out);
    return out;
}

[TorchEntryPoint]
TorchTensor<float3> diffuse_cubemap_bwd(TorchTensor<float3> cubemap,
                                        TorchTensor<float3> grad_out)
{
    uint3 dims = uint3(cubemap.size(2), cubemap.size(1), cubemap.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var cubemap_grad = TorchTensor<float3>.zerosLike(cubemap);

    __dispatch_kernel(diffuse_cubemap_bwd_kernel, blockCount, blockSize)(
        cubemap, grad_out, cubemap_grad);

    return cubemap_grad;
}

[TorchEntryPoint]
TorchTensor<int4> specular_bounds(uint resolution, float costheta_cutoff)
{
    var bounds = TorchTensor<int4>.alloc(6, resolution, resolution, 6);
    uint3 dims = uint3(resolution, resolution, 6);
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    __dispatch_kernel(specular_bounds_kernel, blockCount, blockSize)(costheta_cutoff, bounds);
    return bounds;
}

[TorchEntryPoint]
TorchTensor<float4> specular_cubemap_fwd(TorchTensor<float3> cubemap, TorchTensor<int4> bounds, 
                     float roughness, float costheta_cutoff)
{
    uint3 dims = uint3(cubemap.size(2), cubemap.size(1), cubemap.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var out = TorchTensor<float4>.alloc(cubemap.size(0), cubemap.size(1), cubemap.size(2));
    __dispatch_kernel(specular_cubemap_fwd_kernel, blockCount, blockSize)(cubemap, bounds, roughness, costheta_cutoff, out);
    return out;
}

[TorchEntryPoint]
TorchTensor<float3> specular_cubemap_bwd(TorchTensor<float3> cubemap, TorchTensor<int4> bounds,
                                         float roughness, float costheta_cutoff,
                                         TorchTensor<float4> grad_out)
{
    uint3 dims = uint3(cubemap.size(2), cubemap.size(1), cubemap.size(0));
    let blockSize = getLaunchBlockSize(8, 8, dims);
    let blockCount = getLaunchGridSize(blockSize, dims);
    var cubemap_grad = TorchTensor<float3>.zerosLike(cubemap);

    __dispatch_kernel(specular_cubemap_bwd_kernel, blockCount, blockSize)(
        cubemap, bounds, roughness, costheta_cutoff, grad_out, cubemap_grad);

    return cubemap_grad;
}
