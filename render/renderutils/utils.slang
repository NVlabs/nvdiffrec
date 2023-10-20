// Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction, 
// disclosure or distribution of this material and related documentation 
// without an express license agreement from NVIDIA CORPORATION or 
// its affiliates is strictly prohibited.

uint3 getLaunchGridSize(uint3 blockSize, uint3 dims)
{
    uint3 gridSize;
    gridSize.x = (dims.x - 1) / blockSize.x + 1;
    gridSize.y = (dims.y - 1) / blockSize.y + 1;
    gridSize.z = (dims.z - 1) / blockSize.z + 1;
    return gridSize;
}

//------------------------------------------------------------------------
// Block and grid size calculators for kernel launches.

uint3 getLaunchBlockSize(int maxWidth, int maxHeight, uint3 dims)
{
    int maxThreads = maxWidth * maxHeight;
    if (maxThreads <= 1 || (dims.x * dims.y) <= 1)
        return uint3(1, 1, 1); // Degenerate.

    // Start from max size.
    uint bw = maxWidth;
    uint bh = maxHeight;

    // Optimizations for weirdly sized buffers.
    if (dims.x < bw)
    {
        // Decrease block width to smallest power of two that covers the buffer width.
        while ((bw >> 1) >= dims.x)
            bw >>= 1;

        // Maximize height.
        bh = maxThreads / bw;
        if (bh > dims.y)
            bh = dims.y;
    }
    else if (dims.y < bh)
    {
        // Halve height and double width until fits completely inside buffer vertically.
        while (bh > dims.y)
        {
            bh >>= 1;
            if (bw < dims.x)
                bw <<= 1;
        }
    }

    // Done.
    return uint3(bw, bh, 1);
}

// returns the size of a block that can be reduced using horizontal SIMD operations (e.g. __shfl_xor_sync)
uint3 getWarpSize(uint3 blockSize)
{
    return uint3(
        min(blockSize.x, 32u),
        min(max(32u / blockSize.x, 1u), min(32u, blockSize.y)),
        min(max(32u / (blockSize.x * blockSize.y), 1u), min(32u, blockSize.z))
    );
}

[BackwardDifferentiable]
float3 safeNormalize(float3 v)
{
    float l = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return l > 0.0f ? (v / l) : float3(0.0f);
}

float3 broadcast_fetch(TensorView<float3> tensor, uint2 crd)
{
    let idx = uint2(
        tensor.size(0) == 1 ? 0 : crd.x,
        tensor.size(1) == 1 ? 0 : crd.y);

    return tensor[idx];
}

void broadcast_store(TensorView<float3> tensor, uint2 crd, float3 val)
{
    let idx = uint2(
        tensor.size(0) == 1 ? 0 : crd.x,
        tensor.size(1) == 1 ? 0 : crd.y);

    tensor[idx] = val;
}

float3 broadcast_fetch(TensorView<float3> tensor, uint3 crd)
{
    let idx = uint3(
        tensor.size(0) == 1 ? 0 : crd.x,
        tensor.size(1) == 1 ? 0 : crd.y,
        tensor.size(2) == 1 ? 0 : crd.z);

    return tensor[idx];
}

void broadcast_store(TensorView<float3> tensor, uint3 crd, float3 val)
{
    let idx = uint3(
        tensor.size(0) == 1 ? 0 : crd.x,
        tensor.size(1) == 1 ? 0 : crd.y,
        tensor.size(2) == 1 ? 0 : crd.z);

    tensor[idx] = val;
}