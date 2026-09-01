/*
 * layernorm_fused.cu
 * Fused residual-add + LayerNorm kernel for H100.
 *
 * Computes: output = LayerNorm(x + residual) * gamma + beta
 *
 * Design:
 *   - One block per row (one normalisation instance)
 *   - Warp-level reductions using __shfl_xor_sync (no shared memory needed
 *     for reduction when hidden_dim fits in registers; fallback for large dims)
 *   - Vectorised FP16 loads (8 elements per thread via int4)
 *   - Output is FP16 (avoids extra cast kernel downstream)
 *   - epsilon hardcoded to 1e-5 (Whisper default)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Warp-level reduction helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

// ---------------------------------------------------------------------------
// Fused residual-add + LayerNorm kernel
//
// Grid:  (batch * seq_len,)      — one block per token
// Block: (threads_per_row,)      — should be >= hidden_dim / 8
//
// All tensors are [batch * seq_len, hidden_dim] in row-major FP16.
// gamma and beta are [hidden_dim] FP16.
// ---------------------------------------------------------------------------

__global__ void layernorm_fused_kernel(
    const __half* __restrict__ x,          // input hidden state
    const __half* __restrict__ residual,   // residual to add; nullptr = plain LayerNorm
    const __half* __restrict__ gamma,      // LN scale
    const __half* __restrict__ beta,       // LN bias
    __half*       __restrict__ output,
    __half*       __restrict__ residual_out,  // optional: writes x + residual
    int           hidden_dim,
    float         epsilon
) {
    extern __shared__ float smem[];  // 2 * (blockDim.x / 32) floats

    int row    = blockIdx.x;
    int tid    = threadIdx.x;
    int stride = blockDim.x;

    // Pre-norm transformers (Whisper included) apply LayerNorm on its own and
    // add the residual separately, so the add is optional here. Skipping it
    // avoids reading a zero tensor across all three passes.
    const bool has_res = (residual != nullptr);

    const __half* row_x   = x        + row * hidden_dim;
    const __half* row_res = has_res ? residual + row * hidden_dim : nullptr;
    __half*       row_out = output    + row * hidden_dim;
    __half*       row_sum = residual_out ? residual_out + row * hidden_dim : nullptr;

    // ------------------------------------------------------------------
    // Pass 1: compute mean via vectorised accumulation
    // ------------------------------------------------------------------
    float sum = 0.0f;

    for (int i = tid * 8; i < hidden_dim; i += stride * 8) {
        // Load 8 fp16 from x and residual in one 128-bit read each
        int4 vx  = *reinterpret_cast<const int4*>(row_x   + i);

        __half hx[8], hr[8];
        memcpy(hx, &vx, 16);
        if (has_res) {
            int4 vr = *reinterpret_cast<const int4*>(row_res + i);
            memcpy(hr, &vr, 16);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j)
            sum += __half2float(hx[j]) + (has_res ? __half2float(hr[j]) : 0.0f);
    }

    // Warp reduce
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0)
        smem[tid / 32] = sum;
    __syncthreads();

    // First warp reduces across all warps
    if (tid < 32) {
        sum = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        smem[0] = sum;
    }
    __syncthreads();
    float mean = smem[0] / static_cast<float>(hidden_dim);

    // ------------------------------------------------------------------
    // Pass 2: compute variance
    // ------------------------------------------------------------------
    float var = 0.0f;

    for (int i = tid * 8; i < hidden_dim; i += stride * 8) {
        int4 vx = *reinterpret_cast<const int4*>(row_x   + i);

        __half hx[8], hr[8];
        memcpy(hx, &vx, 16);
        if (has_res) {
            int4 vr = *reinterpret_cast<const int4*>(row_res + i);
            memcpy(hr, &vr, 16);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float val = __half2float(hx[j])
                      + (has_res ? __half2float(hr[j]) : 0.0f) - mean;
            var += val * val;
        }
    }

    var = warp_reduce_sum(var);
    if (tid % 32 == 0)
        smem[tid / 32] = var;
    __syncthreads();

    if (tid < 32) {
        var = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        var = warp_reduce_sum(var);
        smem[0] = var;
    }
    __syncthreads();

    float inv_std = rsqrtf(smem[0] / static_cast<float>(hidden_dim) + epsilon);

    // ------------------------------------------------------------------
    // Pass 3: normalise, scale, bias, write output
    // ------------------------------------------------------------------

    for (int i = tid * 8; i < hidden_dim; i += stride * 8) {
        int4 vx = *reinterpret_cast<const int4*>(row_x   + i);
        int4 vg = *reinterpret_cast<const int4*>(gamma   + i);
        int4 vb = *reinterpret_cast<const int4*>(beta    + i);

        __half hx[8], hr[8], hg[8], hb[8];
        memcpy(hx, &vx, 16);
        memcpy(hg, &vg, 16);
        memcpy(hb, &vb, 16);
        if (has_res) {
            int4 vr = *reinterpret_cast<const int4*>(row_res + i);
            memcpy(hr, &vr, 16);
        }

        __half out[8], sum_out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float s   = __half2float(hx[j])
                      + (has_res ? __half2float(hr[j]) : 0.0f);
            float val = (s - mean) * inv_std;
            val = val * __half2float(hg[j]) + __half2float(hb[j]);
            out[j]     = __float2half(val);
            sum_out[j] = __float2half(s);
        }

        *reinterpret_cast<int4*>(row_out + i) = *reinterpret_cast<const int4*>(out);
        // Pre-norm blocks need the un-normalised sum for the next residual.
        if (row_sum)
            *reinterpret_cast<int4*>(row_sum + i) =
                *reinterpret_cast<const int4*>(sum_out);
    }
}

// ---------------------------------------------------------------------------
// RMS Norm variant (Whisper large-v3 uses RMSNorm in some layers)
// Computes: output = x / rms(x) * gamma   (no mean subtraction, no beta)
// ---------------------------------------------------------------------------

__global__ void rmsnorm_fused_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    const __half* __restrict__ gamma,
    __half*       __restrict__ output,
    int           hidden_dim,
    float         epsilon
) {
    extern __shared__ float smem[];

    int row    = blockIdx.x;
    int tid    = threadIdx.x;
    int stride = blockDim.x;

    const __half* row_x   = x        + row * hidden_dim;
    const __half* row_res = residual  + row * hidden_dim;
    __half*       row_out = output    + row * hidden_dim;

    float ss = 0.0f;

    for (int i = tid * 8; i < hidden_dim; i += stride * 8) {
        int4 vx = *reinterpret_cast<const int4*>(row_x   + i);
        int4 vr = *reinterpret_cast<const int4*>(row_res + i);

        __half hx[8], hr[8];
        memcpy(hx, &vx, 16);
        memcpy(hr, &vr, 16);

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = __half2float(hx[j]) + __half2float(hr[j]);
            ss += v * v;
        }
    }

    ss = warp_reduce_sum(ss);
    if (tid % 32 == 0) smem[tid / 32] = ss;
    __syncthreads();

    if (tid < 32) {
        ss = (tid < blockDim.x / 32) ? smem[tid] : 0.0f;
        ss = warp_reduce_sum(ss);
        smem[0] = ss;
    }
    __syncthreads();

    float inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden_dim) + epsilon);

    for (int i = tid * 8; i < hidden_dim; i += stride * 8) {
        int4 vx = *reinterpret_cast<const int4*>(row_x   + i);
        int4 vr = *reinterpret_cast<const int4*>(row_res + i);
        int4 vg = *reinterpret_cast<const int4*>(gamma   + i);

        __half hx[8], hr[8], hg[8];
        memcpy(hx, &vx, 16);
        memcpy(hr, &vr, 16);
        memcpy(hg, &vg, 16);

        __half out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = (__half2float(hx[j]) + __half2float(hr[j])) * inv_rms;
            out[j] = __float2half(v * __half2float(hg[j]));
        }

        *reinterpret_cast<int4*>(row_out + i) = *reinterpret_cast<const int4*>(out);
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

// Returns {normalised, sum}. `residual` may be undefined (plain LayerNorm);
// `sum` is only materialised when want_sum is true.
std::vector<torch::Tensor> layernorm_fused_impl(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float                epsilon,
    bool                 want_sum
) {
    int rows       = x.numel() / x.size(-1);
    int hidden_dim = x.size(-1);

    // threads: enough to cover hidden_dim / 8 elements, capped at 1024
    int tpb  = std::min(1024, (hidden_dim / 8 + 31) / 32 * 32);
    int smem = (tpb / 32) * sizeof(float);

    auto output = torch::empty_like(x);
    auto sum    = want_sum ? torch::empty_like(x) : torch::Tensor();

    const __half* res_ptr = nullptr;
    if (residual.has_value() && residual->defined()) {
        res_ptr = reinterpret_cast<const __half*>(residual->data_ptr<at::Half>());
    }

    layernorm_fused_kernel<<<rows, tpb, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        res_ptr,
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        want_sum ? reinterpret_cast<__half*>(sum.data_ptr<at::Half>()) : nullptr,
        hidden_dim,
        epsilon
    );
    return {output, sum};
}

torch::Tensor layernorm_fused(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float                epsilon = 1e-5f
) {
    return layernorm_fused_impl(x, residual, gamma, beta, epsilon, false)[0];
}

torch::Tensor rmsnorm_fused(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const torch::Tensor& gamma,
    float                epsilon = 1e-5f
) {
    int rows       = x.numel() / x.size(-1);
    int hidden_dim = x.size(-1);

    int tpb  = std::min(1024, (hidden_dim / 8 + 31) / 32 * 32);
    int smem = (tpb / 32) * sizeof(float);

    auto output = torch::empty_like(x);

    rmsnorm_fused_kernel<<<rows, tpb, smem>>>(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        hidden_dim,
        epsilon
    );
    return output;
}
