/*
 * wgmma_utils.cuh
 * WGMMA (Warpgroup Matrix Multiply-Accumulate) instruction wrappers for H100.
 *
 * Supports:
 *   - m64n128k16  FP16  accumulating to FP32
 *   - m64n64k16   FP16  accumulating to FP32
 *   - m64n128k32  FP8 E4M3×E4M3 → FP32
 *   - m64n128k32  FP8 E4M3×E5M2 → FP32  (mixed)
 *   - m64n64k32   FP8 E4M3×E4M3 → FP32
 *
 * All instructions take shared-memory descriptors (uint64_t) for A and B.
 * The accumulator (D) lives in registers — 64 floats for n=128, 32 for n=64.
 *
 * Requires: -arch=sm_90a
 */

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Accumulator fragments
// ---------------------------------------------------------------------------

// 64×128 tile: 64 FP32 accumulators per thread in a warpgroup
struct WgmmaAccum128 {
    float reg[64] = {};
};

// 64×64 tile: 32 FP32 accumulators per thread
struct WgmmaAccum64 {
    float reg[32] = {};
};

// ---------------------------------------------------------------------------
// Descriptor construction
// Build a wgmma matrix descriptor from a shared-memory pointer.
//
// Layout (64 bits):
//   [13: 0] smem base address >> 4
//   [29:16] leading-dim byte offset >> 4
//   [45:32] stride-dim byte offset >> 4  (same as leading for row-major)
//   [63:62] swizzle mode: 3 = 128B
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t make_wgmma_desc(
    const void* smem_ptr,
    int         leading_byte_offset   // bytes between rows (= K * sizeof(element))
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0ULL;
    desc |= (static_cast<uint64_t>(addr >> 4) & 0x3FFFULL);           // bits [13:0]
    desc |= (static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16); // [29:16]
    desc |= (static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 32); // [45:32]
    desc |= (static_cast<uint64_t>(0x3) << 62);                        // 128B swizzle
    return desc;
}

// ---------------------------------------------------------------------------
// WGMMA  m64 n128 k16  FP16 → FP32
// ---------------------------------------------------------------------------

template <int scale_d = 1>  // 1 = accumulate into d; 0 = overwrite
__device__ __forceinline__ void wgmma_m64n128k16_f16(
    WgmmaAccum128& d,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64, %65, %66, 1, 1, 0, 0;"
        : "+f"(d.reg[ 0]),"+f"(d.reg[ 1]),"+f"(d.reg[ 2]),"+f"(d.reg[ 3]),
          "+f"(d.reg[ 4]),"+f"(d.reg[ 5]),"+f"(d.reg[ 6]),"+f"(d.reg[ 7]),
          "+f"(d.reg[ 8]),"+f"(d.reg[ 9]),"+f"(d.reg[10]),"+f"(d.reg[11]),
          "+f"(d.reg[12]),"+f"(d.reg[13]),"+f"(d.reg[14]),"+f"(d.reg[15]),
          "+f"(d.reg[16]),"+f"(d.reg[17]),"+f"(d.reg[18]),"+f"(d.reg[19]),
          "+f"(d.reg[20]),"+f"(d.reg[21]),"+f"(d.reg[22]),"+f"(d.reg[23]),
          "+f"(d.reg[24]),"+f"(d.reg[25]),"+f"(d.reg[26]),"+f"(d.reg[27]),
          "+f"(d.reg[28]),"+f"(d.reg[29]),"+f"(d.reg[30]),"+f"(d.reg[31]),
          "+f"(d.reg[32]),"+f"(d.reg[33]),"+f"(d.reg[34]),"+f"(d.reg[35]),
          "+f"(d.reg[36]),"+f"(d.reg[37]),"+f"(d.reg[38]),"+f"(d.reg[39]),
          "+f"(d.reg[40]),"+f"(d.reg[41]),"+f"(d.reg[42]),"+f"(d.reg[43]),
          "+f"(d.reg[44]),"+f"(d.reg[45]),"+f"(d.reg[46]),"+f"(d.reg[47]),
          "+f"(d.reg[48]),"+f"(d.reg[49]),"+f"(d.reg[50]),"+f"(d.reg[51]),
          "+f"(d.reg[52]),"+f"(d.reg[53]),"+f"(d.reg[54]),"+f"(d.reg[55]),
          "+f"(d.reg[56]),"+f"(d.reg[57]),"+f"(d.reg[58]),"+f"(d.reg[59]),
          "+f"(d.reg[60]),"+f"(d.reg[61]),"+f"(d.reg[62]),"+f"(d.reg[63])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}

// ---------------------------------------------------------------------------
// WGMMA  m64 n64 k16  FP16 → FP32
// ---------------------------------------------------------------------------

template <int scale_d = 1>
__device__ __forceinline__ void wgmma_m64n64k16_f16(
    WgmmaAccum64& d,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32, %33, %34, 1, 1, 0, 0;"
        : "+f"(d.reg[ 0]),"+f"(d.reg[ 1]),"+f"(d.reg[ 2]),"+f"(d.reg[ 3]),
          "+f"(d.reg[ 4]),"+f"(d.reg[ 5]),"+f"(d.reg[ 6]),"+f"(d.reg[ 7]),
          "+f"(d.reg[ 8]),"+f"(d.reg[ 9]),"+f"(d.reg[10]),"+f"(d.reg[11]),
          "+f"(d.reg[12]),"+f"(d.reg[13]),"+f"(d.reg[14]),"+f"(d.reg[15]),
          "+f"(d.reg[16]),"+f"(d.reg[17]),"+f"(d.reg[18]),"+f"(d.reg[19]),
          "+f"(d.reg[20]),"+f"(d.reg[21]),"+f"(d.reg[22]),"+f"(d.reg[23]),
          "+f"(d.reg[24]),"+f"(d.reg[25]),"+f"(d.reg[26]),"+f"(d.reg[27]),
          "+f"(d.reg[28]),"+f"(d.reg[29]),"+f"(d.reg[30]),"+f"(d.reg[31])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}

// ---------------------------------------------------------------------------
// WGMMA  m64 n128 k32  FP8 E4M3 × E4M3 → FP32
// ---------------------------------------------------------------------------

template <int scale_d = 1>
__device__ __forceinline__ void wgmma_m64n128k32_e4m3e4m3(
    WgmmaAccum128& d,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64, %65, %66, 1, 1;"
        : "+f"(d.reg[ 0]),"+f"(d.reg[ 1]),"+f"(d.reg[ 2]),"+f"(d.reg[ 3]),
          "+f"(d.reg[ 4]),"+f"(d.reg[ 5]),"+f"(d.reg[ 6]),"+f"(d.reg[ 7]),
          "+f"(d.reg[ 8]),"+f"(d.reg[ 9]),"+f"(d.reg[10]),"+f"(d.reg[11]),
          "+f"(d.reg[12]),"+f"(d.reg[13]),"+f"(d.reg[14]),"+f"(d.reg[15]),
          "+f"(d.reg[16]),"+f"(d.reg[17]),"+f"(d.reg[18]),"+f"(d.reg[19]),
          "+f"(d.reg[20]),"+f"(d.reg[21]),"+f"(d.reg[22]),"+f"(d.reg[23]),
          "+f"(d.reg[24]),"+f"(d.reg[25]),"+f"(d.reg[26]),"+f"(d.reg[27]),
          "+f"(d.reg[28]),"+f"(d.reg[29]),"+f"(d.reg[30]),"+f"(d.reg[31]),
          "+f"(d.reg[32]),"+f"(d.reg[33]),"+f"(d.reg[34]),"+f"(d.reg[35]),
          "+f"(d.reg[36]),"+f"(d.reg[37]),"+f"(d.reg[38]),"+f"(d.reg[39]),
          "+f"(d.reg[40]),"+f"(d.reg[41]),"+f"(d.reg[42]),"+f"(d.reg[43]),
          "+f"(d.reg[44]),"+f"(d.reg[45]),"+f"(d.reg[46]),"+f"(d.reg[47]),
          "+f"(d.reg[48]),"+f"(d.reg[49]),"+f"(d.reg[50]),"+f"(d.reg[51]),
          "+f"(d.reg[52]),"+f"(d.reg[53]),"+f"(d.reg[54]),"+f"(d.reg[55]),
          "+f"(d.reg[56]),"+f"(d.reg[57]),"+f"(d.reg[58]),"+f"(d.reg[59]),
          "+f"(d.reg[60]),"+f"(d.reg[61]),"+f"(d.reg[62]),"+f"(d.reg[63])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}

// ---------------------------------------------------------------------------
// WGMMA  m64 n128 k32  FP8 E4M3 × E5M2 → FP32  (mixed)
// Useful for decoder cross-attention: weights in E4M3, wide-range acts in E5M2
// ---------------------------------------------------------------------------

template <int scale_d = 1>
__device__ __forceinline__ void wgmma_m64n128k32_e4m3e5m2(
    WgmmaAccum128& d,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
        "%64, %65, %66, 1, 1;"
        : "+f"(d.reg[ 0]),"+f"(d.reg[ 1]),"+f"(d.reg[ 2]),"+f"(d.reg[ 3]),
          "+f"(d.reg[ 4]),"+f"(d.reg[ 5]),"+f"(d.reg[ 6]),"+f"(d.reg[ 7]),
          "+f"(d.reg[ 8]),"+f"(d.reg[ 9]),"+f"(d.reg[10]),"+f"(d.reg[11]),
          "+f"(d.reg[12]),"+f"(d.reg[13]),"+f"(d.reg[14]),"+f"(d.reg[15]),
          "+f"(d.reg[16]),"+f"(d.reg[17]),"+f"(d.reg[18]),"+f"(d.reg[19]),
          "+f"(d.reg[20]),"+f"(d.reg[21]),"+f"(d.reg[22]),"+f"(d.reg[23]),
          "+f"(d.reg[24]),"+f"(d.reg[25]),"+f"(d.reg[26]),"+f"(d.reg[27]),
          "+f"(d.reg[28]),"+f"(d.reg[29]),"+f"(d.reg[30]),"+f"(d.reg[31]),
          "+f"(d.reg[32]),"+f"(d.reg[33]),"+f"(d.reg[34]),"+f"(d.reg[35]),
          "+f"(d.reg[36]),"+f"(d.reg[37]),"+f"(d.reg[38]),"+f"(d.reg[39]),
          "+f"(d.reg[40]),"+f"(d.reg[41]),"+f"(d.reg[42]),"+f"(d.reg[43]),
          "+f"(d.reg[44]),"+f"(d.reg[45]),"+f"(d.reg[46]),"+f"(d.reg[47]),
          "+f"(d.reg[48]),"+f"(d.reg[49]),"+f"(d.reg[50]),"+f"(d.reg[51]),
          "+f"(d.reg[52]),"+f"(d.reg[53]),"+f"(d.reg[54]),"+f"(d.reg[55]),
          "+f"(d.reg[56]),"+f"(d.reg[57]),"+f"(d.reg[58]),"+f"(d.reg[59]),
          "+f"(d.reg[60]),"+f"(d.reg[61]),"+f"(d.reg[62]),"+f"(d.reg[63])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}

// ---------------------------------------------------------------------------
// WGMMA  m64 n64 k32  FP8 E4M3 × E4M3 → FP32
// ---------------------------------------------------------------------------

template <int scale_d = 1>
__device__ __forceinline__ void wgmma_m64n64k32_e4m3e4m3(
    WgmmaAccum64& d,
    uint64_t desc_a,
    uint64_t desc_b
) {
    asm volatile (
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
        "%32, %33, %34, 1, 1;"
        : "+f"(d.reg[ 0]),"+f"(d.reg[ 1]),"+f"(d.reg[ 2]),"+f"(d.reg[ 3]),
          "+f"(d.reg[ 4]),"+f"(d.reg[ 5]),"+f"(d.reg[ 6]),"+f"(d.reg[ 7]),
          "+f"(d.reg[ 8]),"+f"(d.reg[ 9]),"+f"(d.reg[10]),"+f"(d.reg[11]),
          "+f"(d.reg[12]),"+f"(d.reg[13]),"+f"(d.reg[14]),"+f"(d.reg[15]),
          "+f"(d.reg[16]),"+f"(d.reg[17]),"+f"(d.reg[18]),"+f"(d.reg[19]),
          "+f"(d.reg[20]),"+f"(d.reg[21]),"+f"(d.reg[22]),"+f"(d.reg[23]),
          "+f"(d.reg[24]),"+f"(d.reg[25]),"+f"(d.reg[26]),"+f"(d.reg[27]),
          "+f"(d.reg[28]),"+f"(d.reg[29]),"+f"(d.reg[30]),"+f"(d.reg[31])
        : "l"(desc_a), "l"(desc_b), "n"(scale_d)
    );
}
