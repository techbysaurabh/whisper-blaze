/*
 * bindings.cpp
 * PyTorch custom op registration for all whisper_blaze kernels.
 *
 * Exposes to Python:
 *   whisper_blaze_kernels.quantise_e4m3(fp16_tensor)  → (fp8_tensor, scale)
 *   whisper_blaze_kernels.quantise_e5m2(fp16_tensor)  → (fp8_tensor, scale)
 *   whisper_blaze_kernels.dequantise_e4m3(fp8, scale, shape) → fp16_tensor
 *   whisper_blaze_kernels.dequantise_e5m2(fp8, scale, shape) → fp16_tensor
 *   whisper_blaze_kernels.layernorm_fused(x, res, g, b, eps)  → fp16_tensor
 *   whisper_blaze_kernels.rmsnorm_fused(x, res, g, eps)        → fp16_tensor
 *   whisper_blaze_kernels.mel_spectrogram(audio_f32)            → fp16_tensor [1,128,T]
 *   whisper_blaze_kernels.encoder_self_attn(Q, K, V)            → fp16_tensor
 *   whisper_blaze_kernels.decoder_cross_attn(Q, K, V)           → fp16_tensor
 *   whisper_blaze_kernels.decoder_self_attn(Q, K, V)            → fp16_tensor
 */

#include <torch/extension.h>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations (defined in their respective .cu files)
// ---------------------------------------------------------------------------

// fp8_quantize.cu
std::pair<torch::Tensor, float> quantise_to_e4m3(const torch::Tensor& fp16);
std::pair<torch::Tensor, float> quantise_to_e5m2(const torch::Tensor& fp16);
torch::Tensor dequantise_e4m3(const torch::Tensor& fp8, float scale,
                               const std::vector<int64_t>& shape);
torch::Tensor dequantise_e5m2(const torch::Tensor& fp8, float scale,
                               const std::vector<int64_t>& shape);

// layernorm_fused.cu
torch::Tensor layernorm_fused(const torch::Tensor& x,
                               const torch::Tensor& residual,
                               const torch::Tensor& gamma,
                               const torch::Tensor& beta,
                               float epsilon);
std::vector<torch::Tensor> layernorm_fused_impl(
    const torch::Tensor& x,
    const c10::optional<torch::Tensor>& residual,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float epsilon,
    bool want_sum);
torch::Tensor rmsnorm_fused(const torch::Tensor& x,
                             const torch::Tensor& residual,
                             const torch::Tensor& gamma,
                             float epsilon);

// mel_spectrogram.cu
torch::Tensor compute_mel_spectrogram(const torch::Tensor& audio_cpu);

// flash_attn_hopper.cu
torch::Tensor encoder_self_attn(const torch::Tensor& Q,
                                 const torch::Tensor& K,
                                 const torch::Tensor& V);
torch::Tensor decoder_cross_attn(const torch::Tensor& Q,
                                  const torch::Tensor& K,
                                  const torch::Tensor& V);
torch::Tensor decoder_self_attn(const torch::Tensor& Q,
                                 const torch::Tensor& K,
                                 const torch::Tensor& V);

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

#define CHECK_CUDA(x)   TORCH_CHECK((x).is_cuda(),  #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FP16(x)   TORCH_CHECK((x).dtype() == torch::kFloat16, #x " must be fp16")
#define CHECK_FP32(x)   TORCH_CHECK((x).dtype() == torch::kFloat32, #x " must be fp32")
#define CHECK_INPUT(x)  CHECK_CUDA(x); CHECK_CONTIG(x)

// ---------------------------------------------------------------------------
// Wrapped launchers with validation
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, float> py_quantise_e4m3(const torch::Tensor& fp16) {
    CHECK_INPUT(fp16); CHECK_FP16(fp16);
    return quantise_to_e4m3(fp16);
}

std::pair<torch::Tensor, float> py_quantise_e5m2(const torch::Tensor& fp16) {
    CHECK_INPUT(fp16); CHECK_FP16(fp16);
    return quantise_to_e5m2(fp16);
}

torch::Tensor py_dequantise_e4m3(const torch::Tensor& fp8, float scale,
                                   const std::vector<int64_t>& shape) {
    CHECK_INPUT(fp8);
    return dequantise_e4m3(fp8, scale, shape);
}

torch::Tensor py_dequantise_e5m2(const torch::Tensor& fp8, float scale,
                                   const std::vector<int64_t>& shape) {
    CHECK_INPUT(fp8);
    return dequantise_e5m2(fp8, scale, shape);
}

torch::Tensor py_layernorm_fused(const torch::Tensor& x,
                                  const c10::optional<torch::Tensor>& residual,
                                  const torch::Tensor& gamma,
                                  const torch::Tensor& beta,
                                  float epsilon = 1e-5f) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma); CHECK_INPUT(beta);
    CHECK_FP16(x);
    if (residual.has_value() && residual->defined()) {
        CHECK_INPUT(residual.value());
        TORCH_CHECK(residual->sizes() == x.sizes(),
                    "residual must have the same shape as x");
    }
    return layernorm_fused_impl(x, residual, gamma, beta, epsilon, false)[0];
}

// Same kernel, but also returns the un-normalised (x + residual) sum, which
// pre-norm transformers need to carry forward as the next residual.
std::tuple<torch::Tensor, torch::Tensor> py_layernorm_fused_residual(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& residual,
        const torch::Tensor& gamma,
        const torch::Tensor& beta,
        float epsilon = 1e-5f) {
    CHECK_INPUT(x);
    CHECK_INPUT(gamma); CHECK_INPUT(beta);
    CHECK_FP16(x);
    if (residual.has_value() && residual->defined()) {
        CHECK_INPUT(residual.value());
        TORCH_CHECK(residual->sizes() == x.sizes(),
                    "residual must have the same shape as x");
    }
    auto out = layernorm_fused_impl(x, residual, gamma, beta, epsilon, true);
    return std::make_tuple(out[0], out[1]);
}

torch::Tensor py_rmsnorm_fused(const torch::Tensor& x,
                                const torch::Tensor& residual,
                                const torch::Tensor& gamma,
                                float epsilon = 1e-5f) {
    CHECK_INPUT(x); CHECK_INPUT(residual); CHECK_INPUT(gamma);
    CHECK_FP16(x);
    return rmsnorm_fused(x, residual, gamma, epsilon);
}

torch::Tensor py_mel_spectrogram(const torch::Tensor& audio) {
    TORCH_CHECK(audio.dtype() == torch::kFloat32, "audio must be float32");
    TORCH_CHECK(audio.is_contiguous(), "audio must be contiguous");
    return compute_mel_spectrogram(audio);
}

torch::Tensor py_encoder_self_attn(const torch::Tensor& Q,
                                    const torch::Tensor& K,
                                    const torch::Tensor& V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_FP16(Q); CHECK_FP16(K); CHECK_FP16(V);
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D: [B, H, Sq, D]");
    return encoder_self_attn(Q, K, V);
}

torch::Tensor py_decoder_cross_attn(const torch::Tensor& Q,
                                     const torch::Tensor& K,
                                     const torch::Tensor& V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_FP16(Q);
    return decoder_cross_attn(Q, K, V);
}

torch::Tensor py_decoder_self_attn(const torch::Tensor& Q,
                                    const torch::Tensor& K,
                                    const torch::Tensor& V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);
    CHECK_FP16(Q);
    return decoder_self_attn(Q, K, V);
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "whisper_blaze: Hopper-native CUDA kernels for Whisper large-v3";

    // FP8 quantisation
    m.def("quantise_e4m3",   &py_quantise_e4m3,
          "Quantise FP16 tensor to FP8 E4M3. Returns (fp8_tensor, scale).",
          py::arg("fp16"));

    m.def("quantise_e5m2",   &py_quantise_e5m2,
          "Quantise FP16 tensor to FP8 E5M2. Returns (fp8_tensor, scale).",
          py::arg("fp16"));

    m.def("dequantise_e4m3", &py_dequantise_e4m3,
          "Dequantise FP8 E4M3 tensor to FP16.",
          py::arg("fp8"), py::arg("scale"), py::arg("shape"));

    m.def("dequantise_e5m2", &py_dequantise_e5m2,
          "Dequantise FP8 E5M2 tensor to FP16.",
          py::arg("fp8"), py::arg("scale"), py::arg("shape"));

    // Normalisation
    m.def("layernorm_fused", &py_layernorm_fused,
          "Fused residual-add + LayerNorm → FP16. residual=None gives a plain "
          "LayerNorm (no residual read).",
          py::arg("x"), py::arg("residual") = py::none(),
          py::arg("gamma"), py::arg("beta"),
          py::arg("epsilon") = 1e-5f);

    m.def("layernorm_fused_residual", &py_layernorm_fused_residual,
          "Fused residual-add + LayerNorm → (normalised, x+residual). The "
          "second output is the residual stream for pre-norm blocks.",
          py::arg("x"), py::arg("residual") = py::none(),
          py::arg("gamma"), py::arg("beta"),
          py::arg("epsilon") = 1e-5f);

    m.def("rmsnorm_fused",   &py_rmsnorm_fused,
          "Fused residual-add + RMSNorm → FP16.",
          py::arg("x"), py::arg("residual"),
          py::arg("gamma"),
          py::arg("epsilon") = 1e-5f);

    // Mel spectrogram
    m.def("mel_spectrogram",  &py_mel_spectrogram,
          "Compute Whisper mel spectrogram on GPU. Input: float32 audio. "
          "Output: float16 [1, 128, T].",
          py::arg("audio"));

    // Attention
    m.def("encoder_self_attn",  &py_encoder_self_attn,
          "Encoder self-attention (FA3, no causal mask). [B,H,Sq,D] fp16.",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    m.def("decoder_cross_attn", &py_decoder_cross_attn,
          "Decoder cross-attention (FA3). Q from decoder, K/V from encoder.",
          py::arg("Q"), py::arg("K"), py::arg("V"));

    m.def("decoder_self_attn",  &py_decoder_self_attn,
          "Decoder self-attention (FA3, causal mask). [B,H,Sq,D] fp16.",
          py::arg("Q"), py::arg("K"), py::arg("V"));
}
