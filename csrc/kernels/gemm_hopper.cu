/*
 * gemm_hopper.cu
 * WGMMA + TMA GEMM kernel for H100 Hopper.
 *
 * Computes: D = A × B  (+ optional bias)
 *
 * Supports (via template dispatch):
 *   FP16  × FP16  → FP16   (full precision)
 *   FP8_E4M3 × FP8_E4M3 → FP16   (max throughput)
 *   FP8_E4M3 × FP8_E5M2 → FP16   (mixed, wide-range acts)
 *
 * Tile size: M=64, N=128, K=wgmma_k (16 for fp16, 32 for fp8)
 * Pipeline:  2-stage double buffering (producer TMA + consumer WGMMA)
 *
 * Each thread block:
 *   - Contains one warpgroup (128 threads = 4 warps)
 *   - Handles a 64×128 output tile
 *   - Loops over K dimension with 2-stage pipelined TMA loads
 *
 * Requires: -arch=sm_90a, CUDA 12.0+
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <torch/extension.h>

#include "../include/compute_traits.cuh"
#include "../include/hopper_utils.cuh"
#include "../include/wgmma_utils.cuh"

// ---------------------------------------------------------------------------
// Tile dimensions
// ---------------------------------------------------------------------------

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 128;
static constexpr int NUM_STAGES = 2;

// ---------------------------------------------------------------------------
// Shared memory layout for one pipeline stage
// Holds one A tile (TILE_M × K_TILE) and one B tile (K_TILE × TILE_N)
// ---------------------------------------------------------------------------

template <int K_TILE, int ELEM_BYTES>
struct SmemStage {
    // 128B-aligned smem tiles
    alignas(128) char A[TILE_M * K_TILE * ELEM_BYTES];
    alignas(128) char B[K_TILE * TILE_N * ELEM_BYTES];
};

// ---------------------------------------------------------------------------
// FP16 GEMM kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(WARPGROUP_SIZE)
gemm_fp16_kernel(
    const __grid_constant__ CUtensorMap A_desc,
    const __grid_constant__ CUtensorMap B_desc,
    __half*        __restrict__ output,
    const __half*  __restrict__ bias,    // nullptr if no bias
    int M, int N, int K,
    bool has_bias
) {
    static constexpr int K_TILE     = PrecisionTraits<ComputePrecision::FP16>::wgmma_k; // 16
    static constexpr int ELEM_BYTES = sizeof(__half);

    // Shared memory: 2 stages of A+B tiles + 2 barriers
    __shared__ SmemStage<K_TILE, ELEM_BYTES> stages[NUM_STAGES];
    __shared__ uint64_t mbar[NUM_STAGES];

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    int tid     = threadIdx.x;

    // Init barriers
    if (tid == 0) {
        mbar_init(&mbar[0], TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES);
        mbar_init(&mbar[1], TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES);
    }
    __syncthreads();

    // Accumulator registers: 64 floats for TILE_M=64, TILE_N=128
    WgmmaAccum128 acc;

    int k_tiles = K / K_TILE;
    int stage   = 0;

    // ------------------------------------------------------------------
    // Prefetch stage 0
    // ------------------------------------------------------------------
    if (tid == 0) {
        mbar_arrive_expect_tx(&mbar[0],
            TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES);
        tma_load_2d(A_desc, stages[0].A, 0,        block_m, &mbar[0]);
        tma_load_2d(B_desc, stages[0].B, block_n,  0,       &mbar[0]);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Main loop
    // ------------------------------------------------------------------
    for (int k = 0; k < k_tiles; ++k) {
        int next_k = k + 1;

        // Prefetch next stage while computing current
        if (next_k < k_tiles) {
            int next_stage = next_k % NUM_STAGES;
            if (tid == 0) {
                mbar_arrive_expect_tx(&mbar[next_stage],
                    TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES);
                tma_load_2d(A_desc, stages[next_stage].A,
                            next_k * K_TILE, block_m, &mbar[next_stage]);
                tma_load_2d(B_desc, stages[next_stage].B,
                            block_n, next_k * K_TILE,  &mbar[next_stage]);
            }
        }

        // Wait for current stage data to arrive
        mbar_wait(&mbar[stage], (k / NUM_STAGES) % 2);
        fence_proxy_async();

        // Build WGMMA descriptors from shared memory
        uint64_t desc_a = make_wgmma_desc(
            stages[stage].A, K_TILE * ELEM_BYTES);
        uint64_t desc_b = make_wgmma_desc(
            stages[stage].B, TILE_N * ELEM_BYTES);

        // Issue WGMMA: accumulate into acc
        if (k == 0)
            wgmma_m64n128k16_f16<0>(acc, desc_a, desc_b);
        else
            wgmma_m64n128k16_f16<1>(acc, desc_a, desc_b);
        wgmma_commit_group();
        wgmma_wait_group<0>();

        // Re-init barrier for reuse
        if (tid == 0)
            mbar_init(&mbar[stage], TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES);

        stage = (stage + 1) % NUM_STAGES;
    }

    // ------------------------------------------------------------------
    // Write accumulator to global memory
    // Each thread owns 64/128 * TILE_N = 64 registers covering a 64×128 tile.
    // Threads in a warpgroup cover the tile in a specific pattern (8×16 per warp).
    // ------------------------------------------------------------------

    // Each warp of 32 threads covers an 8×16 strip of the 64×128 output tile.
    // Row offset within tile = warp_id * 16; col offset = lane_id / 2
    int warp   = warp_id();
    int lane   = lane_id();
    int row_offset = warp * 16 + (lane / 4);
    int col_offset = (lane % 4) * 2;

    // 64 registers span 8 rows × 16 cols per warp, tiled 4× across N
    for (int reg = 0; reg < 64; ++reg) {
        int local_row = row_offset + (reg / 16) * 8;
        int local_col = col_offset + (reg % 16);

        int global_row = block_m + local_row;
        int global_col = block_n + local_col;

        if (global_row < M && global_col < N) {
            float val = acc.reg[reg];
            if (has_bias)
                val += __half2float(bias[global_col]);
            output[global_row * N + global_col] = __float2half(val);
        }
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3×E4M3 GEMM kernel (same structure, different WGMMA)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(WARPGROUP_SIZE)
gemm_fp8_e4m3_kernel(
    const __grid_constant__ CUtensorMap A_desc,
    const __grid_constant__ CUtensorMap B_desc,
    __half*       __restrict__ output,
    const __half* __restrict__ bias,
    float         w_scale,
    float         a_scale,
    int M, int N, int K,
    bool has_bias
) {
    static constexpr int K_TILE     = PrecisionTraits<ComputePrecision::FP8_E4M3>::wgmma_k; // 32
    static constexpr int ELEM_BYTES = sizeof(__nv_fp8_e4m3);

    __shared__ SmemStage<K_TILE, ELEM_BYTES> stages[NUM_STAGES];
    __shared__ uint64_t mbar[NUM_STAGES];

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    int tid     = threadIdx.x;

    if (tid == 0) {
        int tx_bytes = TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES;
        mbar_init(&mbar[0], tx_bytes);
        mbar_init(&mbar[1], tx_bytes);
    }
    __syncthreads();

    WgmmaAccum128 acc;
    int k_tiles = K / K_TILE;
    int stage   = 0;
    int tx_bytes = TILE_M * K_TILE * ELEM_BYTES + K_TILE * TILE_N * ELEM_BYTES;

    if (tid == 0) {
        mbar_arrive_expect_tx(&mbar[0], tx_bytes);
        tma_load_2d(A_desc, stages[0].A, 0,       block_m, &mbar[0]);
        tma_load_2d(B_desc, stages[0].B, block_n, 0,       &mbar[0]);
    }
    __syncthreads();

    for (int k = 0; k < k_tiles; ++k) {
        int next_k = k + 1;
        if (next_k < k_tiles) {
            int ns = next_k % NUM_STAGES;
            if (tid == 0) {
                mbar_arrive_expect_tx(&mbar[ns], tx_bytes);
                tma_load_2d(A_desc, stages[ns].A,
                            next_k * K_TILE, block_m, &mbar[ns]);
                tma_load_2d(B_desc, stages[ns].B,
                            block_n, next_k * K_TILE,  &mbar[ns]);
            }
        }

        mbar_wait(&mbar[stage], (k / NUM_STAGES) % 2);
        fence_proxy_async();

        uint64_t desc_a = make_wgmma_desc(stages[stage].A, K_TILE * ELEM_BYTES);
        uint64_t desc_b = make_wgmma_desc(stages[stage].B, TILE_N * ELEM_BYTES);

        if (k == 0)
            wgmma_m64n128k32_e4m3e4m3<0>(acc, desc_a, desc_b);
        else
            wgmma_m64n128k32_e4m3e4m3<1>(acc, desc_a, desc_b);
        wgmma_commit_group();
        wgmma_wait_group<0>();

        if (tid == 0)
            mbar_init(&mbar[stage], tx_bytes);

        stage = (stage + 1) % NUM_STAGES;
    }

    float descale = w_scale * a_scale;
    int warp = warp_id(), lane = lane_id();
    int row_offset = warp * 16 + (lane / 4);
    int col_offset = (lane % 4) * 2;

    for (int reg = 0; reg < 64; ++reg) {
        int local_row = row_offset + (reg / 16) * 8;
        int local_col = col_offset + (reg % 16);
        int global_row = block_m + local_row;
        int global_col = block_n + local_col;

        if (global_row < M && global_col < N) {
            float val = acc.reg[reg] * descale;
            if (has_bias)
                val += __half2float(bias[global_col]);
            output[global_row * N + global_col] = __float2half(val);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

torch::Tensor gemm_fp16(
    const torch::Tensor& A,      // [M, K] fp16
    const torch::Tensor& B,      // [K, N] fp16
    const torch::Tensor& bias,   // [N] fp16, optional (empty = no bias)
    CUtensorMap* A_tma,
    CUtensorMap* B_tma
) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat16).device(A.device()));

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_fp16_kernel<<<grid, WARPGROUP_SIZE>>>(
        *A_tma, *B_tma,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        bias.numel() ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()) : nullptr,
        M, N, K,
        bias.numel() > 0
    );
    return output;
}

torch::Tensor gemm_fp8_e4m3(
    const torch::Tensor& A_fp8,   // [M, K] uint8 (e4m3 bits)
    const torch::Tensor& B_fp8,   // [K, N] uint8 (e4m3 bits)
    float                w_scale,
    float                a_scale,
    const torch::Tensor& bias,
    CUtensorMap*         A_tma,
    CUtensorMap*         B_tma
) {
    int M = A_fp8.size(0), K = A_fp8.size(1), N = B_fp8.size(1);
    auto output = torch::empty({M, N},
        torch::dtype(torch::kFloat16).device(A_fp8.device()));

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    gemm_fp8_e4m3_kernel<<<grid, WARPGROUP_SIZE>>>(
        *A_tma, *B_tma,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        bias.numel() ? reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()) : nullptr,
        w_scale, a_scale,
        M, N, K,
        bias.numel() > 0
    );
    return output;
}
