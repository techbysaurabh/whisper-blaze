/*
 * compute_traits.cuh
 * Type traits for FP16 and FP8 (E4M3 / E5M2) precisions.
 * Controls storage types, accumulator types, WGMMA K-tile size, and scaling.
 *
 * Usage:
 *   using Traits = PrecisionTraits<ComputePrecision::FP8_E4M3>;
 *   Traits::scalar_t* ptr = ...;          // __nv_fp8_e4m3
 *   Traits::accum_t  acc = 0.0f;          // float
 *   constexpr int K = Traits::wgmma_k;    // 32 for FP8, 16 for FP16
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Precision enum
// ---------------------------------------------------------------------------

enum class ComputePrecision : uint8_t {
    FP16     = 0,
    FP8_E4M3 = 1,
    FP8_E5M2 = 2,
};

// ---------------------------------------------------------------------------
// Primary template — undefined; only specialisations below are valid
// ---------------------------------------------------------------------------

template <ComputePrecision P>
struct PrecisionTraits;

// ---------------------------------------------------------------------------
// FP16
// ---------------------------------------------------------------------------

template <>
struct PrecisionTraits<ComputePrecision::FP16> {
    // Storage type
    using scalar_t = __half;
    // Accumulator type (always FP32 on H100)
    using accum_t  = float;

    // WGMMA K-tile for FP16 is 16
    static constexpr int wgmma_k = 16;

    // FP8 needs per-tensor scale; FP16 does not
    static constexpr bool needs_scale = false;

    // Max representable value (used to clamp before quantisation)
    static constexpr float max_val = 65504.0f;

    // PTX type string tokens used when constructing PTX strings
    static constexpr const char* ptx_type = "f16";
};

// ---------------------------------------------------------------------------
// FP8 E4M3  — higher precision, narrower range
//             Best for: weights, forward activations
// ---------------------------------------------------------------------------

template <>
struct PrecisionTraits<ComputePrecision::FP8_E4M3> {
    using scalar_t = __nv_fp8_e4m3;
    using accum_t  = float;

    static constexpr int wgmma_k = 32;   // FP8 WGMMA processes 32 elements per step

    static constexpr bool needs_scale = true;

    // Max value for E4M3 is 448.0
    static constexpr float max_val = 448.0f;

    static constexpr const char* ptx_type = "e4m3";
};

// ---------------------------------------------------------------------------
// FP8 E5M2  — lower precision, wider range
//             Best for: attention scores, gradients, decoder cross-attn
// ---------------------------------------------------------------------------

template <>
struct PrecisionTraits<ComputePrecision::FP8_E5M2> {
    using scalar_t = __nv_fp8_e5m2;
    using accum_t  = float;

    static constexpr int wgmma_k = 32;

    static constexpr bool needs_scale = true;

    // Max value for E5M2 is 57344.0
    static constexpr float max_val = 57344.0f;

    static constexpr const char* ptx_type = "e5m2";
};

// ---------------------------------------------------------------------------
// Helper: mixed-precision WGMMA traits (A type, B type)
// ---------------------------------------------------------------------------

template <ComputePrecision A, ComputePrecision B>
struct MixedWgmmaTraits {
    using A_traits = PrecisionTraits<A>;
    using B_traits = PrecisionTraits<B>;
    using A_scalar = typename A_traits::scalar_t;
    using B_scalar = typename B_traits::scalar_t;
    using accum_t  = float;

    // K tile is max of the two (both must match for valid wgmma)
    static constexpr int wgmma_k = A_traits::wgmma_k;
    static_assert(A_traits::wgmma_k == B_traits::wgmma_k,
                  "A and B precisions must have matching wgmma_k");
};

// ---------------------------------------------------------------------------
// Runtime precision descriptor (for Python dispatch)
// ---------------------------------------------------------------------------

struct PrecisionDesc {
    ComputePrecision weight_prec;
    ComputePrecision act_prec;

    bool operator==(const PrecisionDesc& o) const {
        return weight_prec == o.weight_prec && act_prec == o.act_prec;
    }
};

// Preset configs
namespace PrecisionPresets {
    // All FP16 — maximum quality
    constexpr PrecisionDesc FULL_FP16 = {
        ComputePrecision::FP16,
        ComputePrecision::FP16
    };
    // FP8 weights + FP8 activations (E4M3) — max throughput, good quality
    constexpr PrecisionDesc FP8_E4M3_BOTH = {
        ComputePrecision::FP8_E4M3,
        ComputePrecision::FP8_E4M3
    };
    // FP8 weights (E4M3) + FP8 acts (E5M2) — wide-range activations
    constexpr PrecisionDesc FP8_MIXED = {
        ComputePrecision::FP8_E4M3,
        ComputePrecision::FP8_E5M2
    };
}
