/*
 * hopper_utils.cuh
 * Hopper (SM90) hardware utilities:
 *   - TMA descriptor construction & async bulk copy
 *   - Async transaction barriers (cuda::barrier)
 *   - Thread Block Cluster helpers
 *   - Warp group ID helpers
 *
 * Requires: CUDA 12.0+, -arch=sm_90a
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cstdint>
#include <cassert>

// ---------------------------------------------------------------------------
// Compile-time guard
// ---------------------------------------------------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "hopper_utils.cuh requires sm_90 or higher (H100)"
#endif

// ---------------------------------------------------------------------------
// Warp / Warpgroup helpers
// ---------------------------------------------------------------------------

// Number of threads per warpgroup (4 warps)
static constexpr int WARPGROUP_SIZE = 128;
static constexpr int WARP_SIZE      = 32;

__device__ __forceinline__ int warp_id() {
    return threadIdx.x / WARP_SIZE;
}

__device__ __forceinline__ int warpgroup_id() {
    return threadIdx.x / WARPGROUP_SIZE;
}

__device__ __forceinline__ int lane_id() {
    return threadIdx.x % WARP_SIZE;
}

__device__ __forceinline__ int thread_id_in_warpgroup() {
    return threadIdx.x % WARPGROUP_SIZE;
}

// ---------------------------------------------------------------------------
// TMA data types (mirrors CUtensorMapDataType)
// ---------------------------------------------------------------------------

enum class TMADtype : uint32_t {
    F16    = CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    BF16   = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    F32    = CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    // FP8 types are 1-byte; TMA treats them as UINT8 for bulk copy.
    // The hardware copies raw bytes — no format conversion occurs in TMA.
    FP8_E4M3 = CU_TENSOR_MAP_DATA_TYPE_UINT8,
    FP8_E5M2 = CU_TENSOR_MAP_DATA_TYPE_UINT8,
};

// ---------------------------------------------------------------------------
// TMA 2D descriptor construction (host-side)
// Call once per weight/activation tensor before the kernel launch.
// ---------------------------------------------------------------------------

inline void tma_create_descriptor_2d(
    CUtensorMap*  out_map,          // output descriptor
    void*         global_ptr,       // base pointer of global tensor
    uint64_t      rows,             // total rows
    uint64_t      cols,             // total cols
    uint32_t      tile_rows,        // shared memory tile rows
    uint32_t      tile_cols,        // shared memory tile cols
    TMADtype      dtype
) {
    const uint64_t global_dims[2]    = {cols, rows};   // TMA is col-major dims
    const uint64_t global_strides[1] = {cols * /* elem bytes */ [&]() -> uint64_t {
        uint32_t d = static_cast<uint32_t>(dtype);
        if (d == static_cast<uint32_t>(TMADtype::F32))  return 4;
        if (d == static_cast<uint32_t>(TMADtype::F16)  ||
            d == static_cast<uint32_t>(TMADtype::BF16))  return 2;
        /* FP8_E4M3, FP8_E5M2 (both mapped to UINT8) */  return 1;
    }()};

    const uint32_t box_dims[2]    = {tile_cols, tile_rows};
    const uint32_t elem_strides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        out_map,
        static_cast<CUtensorMapDataType>(dtype),
        2,                                // tensor rank
        global_ptr,
        global_dims,
        global_strides,                   // stride of dim-1 in bytes
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,       // 128B swizzle for bank-conflict-free access
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    assert(res == CUDA_SUCCESS && "cuTensorMapEncodeTiled failed");
}

// ---------------------------------------------------------------------------
// TMA async copy (device-side): global → shared
// Issues an async bulk copy using TMA hardware.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void tma_load_2d(
    const CUtensorMap& desc,
    void*              smem_ptr,
    int                coord0,     // column index in global tensor
    int                coord1,     // row index in global tensor
    uint64_t*          mbar        // mbarrier address in shared memory
) {
    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr))),
          "l"(&desc),
          "r"(coord0),
          "r"(coord1),
          "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
        : "memory"
    );
}

// ---------------------------------------------------------------------------
// Async transaction barrier helpers
// Wraps cuda::barrier for the producer/consumer pipeline pattern.
// ---------------------------------------------------------------------------

// Initialize a shared-memory mbarrier for `expected_tx` bytes of transactions
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t expected_tx) {
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
          "r"(expected_tx)
        : "memory"
    );
}

// Arrive on the barrier (signals one thread has finished producing)
__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    asm volatile (
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar)))
        : "memory"
    );
}

// Wait until all expected transactions have completed
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile (
        "{\n"
        ".reg .pred p;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAIT;\n"
        "}\n"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
          "r"(phase)
        : "memory"
    );
}

// Arrive-and-expect: producer announces N bytes will arrive via TMA
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    asm volatile (
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
          "r"(tx_bytes)
        : "memory"
    );
}

// ---------------------------------------------------------------------------
// Thread Block Cluster helpers (Hopper new feature)
// Allows SMs in the same cluster to access each other's shared memory (DSMEM)
// ---------------------------------------------------------------------------

// Get the cluster-local block index
__device__ __forceinline__ uint32_t cluster_block_rank() {
    uint32_t rank;
    asm ("mov.u32 %0, %cluster_ctaid.x;" : "=r"(rank));
    return rank;
}

// Fence to ensure DSMEM visibility across blocks in the cluster
__device__ __forceinline__ void cluster_arrive() {
    asm volatile ("barrier.cluster.arrive;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait() {
    asm volatile ("barrier.cluster.wait;" ::: "memory");
}

// ---------------------------------------------------------------------------
// Warpgroup commit/wait for WGMMA async groups
// ---------------------------------------------------------------------------

__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile ("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int N = 0>
__device__ __forceinline__ void wgmma_wait_group() {
    // Wait until at most N wgmma groups are outstanding
    asm volatile (
        "wgmma.wait_group.sync.aligned %0;"
        :
        : "n"(N)
        : "memory"
    );
}

// ---------------------------------------------------------------------------
// Shared memory fence
// ---------------------------------------------------------------------------

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile ("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------------------
// Pipeline stage buffer helper
// Simple N-stage ping-pong buffer for K/V tiles in shared memory.
// ---------------------------------------------------------------------------

template <int NUM_STAGES>
__device__ __forceinline__ int next_stage(int current) {
    return (current + 1) % NUM_STAGES;
}
