/*
 * fp8_quantize.cu
 * GPU kernels for FP16 ↔ FP8 (E4M3 / E5M2) quantisation and dequantisation.
 *
 * Provides:
 *   - Per-tensor scale computation  (max-abs)
 *   - Quantise: FP16 → FP8 with scale
 *   - Dequantise: FP8 → FP16 with scale
 *   - Fused cast + scale for both directions
 *
 * All kernels are vectorised (process 8 elements per thread using int4 loads).
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <cfloat>

#include "../include/compute_traits.cuh"

// Local warp helpers (avoid pulling in hopper_utils.cuh which has asm constraints
// that fail to compile in translation units where they are not inlined)
__device__ __forceinline__ int fp8_lane_id() { return threadIdx.x % 32; }
__device__ __forceinline__ int fp8_warp_id() { return threadIdx.x / 32; }

// ---------------------------------------------------------------------------
// Warp-level max reduction
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

// ---------------------------------------------------------------------------
// Per-tensor scale computation kernel
// Computes scale = max(|x|) / max_val_fp8
// One block per tensor; uses shared memory reduction across warps.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void compute_scale_kernel(
    const scalar_t* __restrict__ input,
    float*          __restrict__ scale_out,
    float           max_fp8,
    int             n_elements
) {
    extern __shared__ float smem[];   // one float per warp

    float local_max = 0.0f;
    int   idx       = blockIdx.x * blockDim.x + threadIdx.x;
    int   stride    = gridDim.x * blockDim.x;

    for (int i = idx; i < n_elements; i += stride)
        local_max = fmaxf(local_max, fabsf(static_cast<float>(input[i])));

    local_max = warp_reduce_max(local_max);

    if (fp8_lane_id() == 0)
        smem[fp8_warp_id()] = local_max;
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (fp8_warp_id() == 0) {
        float v = (threadIdx.x < blockDim.x / 32) ? smem[threadIdx.x] : 0.0f;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0)
            scale_out[blockIdx.x] = v / max_fp8;
    }
}

// ---------------------------------------------------------------------------
// Quantise kernel: FP16 → FP8 E4M3
// Vectorised: 8 fp16 elements per thread via int4 load.
// ---------------------------------------------------------------------------

__global__ void quantise_fp16_to_e4m3_kernel(
    const __half*        __restrict__ input,
    __nv_fp8_e4m3*       __restrict__ output,
    float                             inv_scale,   // 1.0 / scale
    int                               n_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n_elements) return;

    // Load 8 fp16 values in one 128-bit transaction
    int4 raw;
    raw = *reinterpret_cast<const int4*>(input + idx);

    __half h[8];
    static_assert(sizeof(h) == sizeof(raw));
    memcpy(h, &raw, sizeof(raw));

    __nv_fp8_e4m3 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float scaled = __half2float(h[i]) * inv_scale;
        // Clamp to E4M3 max before cast
        scaled = fminf(fmaxf(scaled, -448.0f), 448.0f);
        out[i] = __nv_fp8_e4m3(scaled);
    }

    // Store 8 fp8 values (8 bytes = one int2)
    *reinterpret_cast<int2*>(output + idx) = *reinterpret_cast<const int2*>(out);
}

// ---------------------------------------------------------------------------
// Quantise kernel: FP16 → FP8 E5M2
// ---------------------------------------------------------------------------

__global__ void quantise_fp16_to_e5m2_kernel(
    const __half*        __restrict__ input,
    __nv_fp8_e5m2*       __restrict__ output,
    float                             inv_scale,
    int                               n_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n_elements) return;

    int4 raw = *reinterpret_cast<const int4*>(input + idx);
    __half h[8];
    memcpy(h, &raw, sizeof(raw));

    __nv_fp8_e5m2 out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float scaled = __half2float(h[i]) * inv_scale;
        scaled = fminf(fmaxf(scaled, -57344.0f), 57344.0f);
        out[i] = __nv_fp8_e5m2(scaled);
    }

    *reinterpret_cast<int2*>(output + idx) = *reinterpret_cast<const int2*>(out);
}

// ---------------------------------------------------------------------------
// Dequantise kernel: FP8 E4M3 → FP16
// ---------------------------------------------------------------------------

__global__ void dequantise_e4m3_to_fp16_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    __half*              __restrict__ output,
    float                             scale,
    int                               n_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n_elements) return;

    int2 raw = *reinterpret_cast<const int2*>(input + idx);
    __nv_fp8_e4m3 in[8];
    memcpy(in, &raw, sizeof(raw));

    __half out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = __float2half(static_cast<float>(in[i]) * scale);

    *reinterpret_cast<int4*>(output + idx) = *reinterpret_cast<const int4*>(out);
}

// ---------------------------------------------------------------------------
// Dequantise kernel: FP8 E5M2 → FP16
// ---------------------------------------------------------------------------

__global__ void dequantise_e5m2_to_fp16_kernel(
    const __nv_fp8_e5m2* __restrict__ input,
    __half*              __restrict__ output,
    float                             scale,
    int                               n_elements
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= n_elements) return;

    int2 raw = *reinterpret_cast<const int2*>(input + idx);
    __nv_fp8_e5m2 in[8];
    memcpy(in, &raw, sizeof(raw));

    __half out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        out[i] = __float2half(static_cast<float>(in[i]) * scale);

    *reinterpret_cast<int4*>(output + idx) = *reinterpret_cast<const int4*>(out);
}

// ---------------------------------------------------------------------------
// Host-side launcher functions (called from bindings.cpp)
// ---------------------------------------------------------------------------

// Compute per-tensor scale for a FP16 tensor
float compute_fp8_scale(const torch::Tensor& input, float fp8_max_val) {
    auto   flat   = input.view(-1);
    int    n      = flat.numel();
    int    tpb    = 256;
    int    blocks = std::min((n + tpb - 1) / tpb, 1024);
    int    smem   = (tpb / 32) * sizeof(float);

    auto scale_buf = torch::zeros({blocks}, input.options().dtype(torch::kFloat32));

    // Launch with FP16 input
    compute_scale_kernel<__half><<<blocks, tpb, smem>>>(
        reinterpret_cast<const __half*>(flat.data_ptr<at::Half>()),
        scale_buf.data_ptr<float>(),
        fp8_max_val,
        n
    );

    // One more reduction on CPU (blocks <= 1024, trivial)
    auto   host_scales = scale_buf.cpu();
    float* ptr         = host_scales.data_ptr<float>();
    float  max_scale   = 0.0f;
    for (int i = 0; i < blocks; ++i)
        max_scale = std::max(max_scale, ptr[i]);

    return max_scale > 0.0f ? max_scale : 1.0f;
}

// FP16 tensor → FP8 E4M3 tensor  +  computed scale
std::pair<torch::Tensor, float> quantise_to_e4m3(const torch::Tensor& fp16) {
    float scale     = compute_fp8_scale(fp16, 448.0f);
    float inv_scale = 1.0f / scale;

    auto out = torch::empty_like(fp16, fp16.options().dtype(torch::kUInt8));
    int  n   = fp16.numel();
    int  tpb = 256;
    int  blk = (n / 8 + tpb - 1) / tpb;

    quantise_fp16_to_e4m3_kernel<<<blk, tpb>>>(
        reinterpret_cast<const __half*>(fp16.data_ptr<at::Half>()),
        reinterpret_cast<__nv_fp8_e4m3*>(out.data_ptr<uint8_t>()),
        inv_scale,
        n
    );
    return {out, scale};
}

// FP16 tensor → FP8 E5M2 tensor  +  computed scale
std::pair<torch::Tensor, float> quantise_to_e5m2(const torch::Tensor& fp16) {
    float scale     = compute_fp8_scale(fp16, 57344.0f);
    float inv_scale = 1.0f / scale;

    auto out = torch::empty_like(fp16, fp16.options().dtype(torch::kUInt8));
    int  n   = fp16.numel();
    int  tpb = 256;
    int  blk = (n / 8 + tpb - 1) / tpb;

    quantise_fp16_to_e5m2_kernel<<<blk, tpb>>>(
        reinterpret_cast<const __half*>(fp16.data_ptr<at::Half>()),
        reinterpret_cast<__nv_fp8_e5m2*>(out.data_ptr<uint8_t>()),
        inv_scale,
        n
    );
    return {out, scale};
}

// FP8 E4M3 → FP16
torch::Tensor dequantise_e4m3(const torch::Tensor& fp8, float scale,
                               const std::vector<int64_t>& shape) {
    auto out = torch::empty(shape, torch::dtype(torch::kFloat16).device(fp8.device()));
    int  n   = fp8.numel();
    int  tpb = 256;
    int  blk = (n / 8 + tpb - 1) / tpb;

    dequantise_e4m3_to_fp16_kernel<<<blk, tpb>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(fp8.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        scale,
        n
    );
    return out;
}

// FP8 E5M2 → FP16
torch::Tensor dequantise_e5m2(const torch::Tensor& fp8, float scale,
                               const std::vector<int64_t>& shape) {
    auto out = torch::empty(shape, torch::dtype(torch::kFloat16).device(fp8.device()));
    int  n   = fp8.numel();
    int  tpb = 256;
    int  blk = (n / 8 + tpb - 1) / tpb;

    dequantise_e5m2_to_fp16_kernel<<<blk, tpb>>>(
        reinterpret_cast<const __nv_fp8_e5m2*>(fp8.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        scale,
        n
    );
    return out;
}
