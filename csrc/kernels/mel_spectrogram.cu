/*
 * mel_spectrogram.cu
 * GPU-native mel spectrogram computation for Whisper large-v3.
 *
 * Parameters (matching Whisper exactly):
 *   sample_rate = 16000
 *   n_fft       = 400  (25 ms window)
 *   hop_length  = 160  (10 ms hop)
 *   n_mels      = 128  (large-v3)
 *   f_min       = 0 Hz
 *   f_max       = 8000 Hz
 *
 * Pipeline:
 *   1. hann_window_kernel   — precompute 400-point Hann window (once)
 *   2. stft_kernel          — windowed STFT frames using cuFFT (512-pt, zero-padded)
 *   3. mel_filterbank_kernel — apply 128 mel filters to power spectrum
 *   4. log_norm_kernel      — log10(max(spec, 1e-10)) then normalise to [-1, 1] range
 *
 * Output: float16 tensor of shape [1, 128, 3000] (30-second chunk)
 *
 * Note: cuFFT is used for the FFT step (radix-2, 512-pt).
 *       The mel filterbank is applied as a sparse matrix-vector product.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <torch/extension.h>
#include <cmath>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int   N_FFT       = 400;
static constexpr int   FFT_SIZE    = 512;   // zero-pad to next power of 2
static constexpr int   HOP_LENGTH  = 160;
static constexpr int   N_MELS      = 128;
static constexpr int   N_FRAMES    = 3000;  // for 30s audio
static constexpr int   SAMPLE_RATE = 16000;
static constexpr float F_MIN       = 0.0f;
static constexpr float F_MAX       = 8000.0f;

// ---------------------------------------------------------------------------
// Hann window precomputation (host, called once)
// ---------------------------------------------------------------------------

static std::vector<float> make_hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (n - 1)));
    return w;
}

// ---------------------------------------------------------------------------
// Mel filterbank construction (host, called once)
// Returns a [n_mels, fft_bins] float matrix where fft_bins = FFT_SIZE/2 + 1
// ---------------------------------------------------------------------------

static std::vector<float> make_mel_filterbank(
    int n_mels, int fft_size, int sample_rate, float f_min, float f_max
) {
    int fft_bins = fft_size / 2 + 1;

    auto hz_to_mel = [](float hz) {
        return 2595.0f * log10f(1.0f + hz / 700.0f);
    };
    auto mel_to_hz = [](float mel) {
        return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    };

    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);

    // n_mels + 2 equally spaced mel points
    std::vector<float> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i)
        mel_pts[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));

    // Map mel points to FFT bin indices
    std::vector<float> freq_bins(fft_bins);
    for (int i = 0; i < fft_bins; ++i)
        freq_bins[i] = static_cast<float>(i) * sample_rate / fft_size;

    std::vector<float> fb(n_mels * fft_bins, 0.0f);

    for (int m = 0; m < n_mels; ++m) {
        float f_left   = mel_pts[m];
        float f_center = mel_pts[m + 1];
        float f_right  = mel_pts[m + 2];

        for (int k = 0; k < fft_bins; ++k) {
            float f = freq_bins[k];
            if (f >= f_left && f <= f_center)
                fb[m * fft_bins + k] = (f - f_left) / (f_center - f_left);
            else if (f > f_center && f <= f_right)
                fb[m * fft_bins + k] = (f_right - f) / (f_right - f_center);
        }
    }
    return fb;
}

// ---------------------------------------------------------------------------
// STFT framing + windowing kernel
// Extracts overlapping windowed frames from raw audio into a padded FFT buffer.
//
// Grid:  (n_frames, 1)
// Block: (FFT_SIZE, 1)
//
// Input:  audio[n_samples]  float32
// Output: frames[n_frames * FFT_SIZE]  float32  (zero-padded to FFT_SIZE)
// ---------------------------------------------------------------------------

__global__ void stft_frame_kernel(
    const float* __restrict__ audio,
    const float* __restrict__ window,
    cufftComplex*             frames,   // interleaved complex (im=0 before FFT)
    int                       n_samples,
    int                       n_frames
) {
    int frame_idx = blockIdx.x;
    int tid       = threadIdx.x;

    if (frame_idx >= n_frames) return;

    int sample_start = frame_idx * HOP_LENGTH;

    float val = 0.0f;
    if (tid < N_FFT) {
        int sample_idx = sample_start + tid;
        if (sample_idx < n_samples)
            val = audio[sample_idx] * window[tid];
    }
    // Positions [N_FFT, FFT_SIZE) are zero (zero-padding for power-of-2 FFT)

    frames[frame_idx * FFT_SIZE + tid] = make_cuFloatComplex(val, 0.0f);
}

// ---------------------------------------------------------------------------
// Power spectrum kernel
// Converts complex FFT output → power spectrum (magnitude squared).
// Only the first FFT_SIZE/2 + 1 bins are kept (Hermitian symmetry).
//
// Grid:  (n_frames, 1)
// Block: (FFT_SIZE/2+1, 1)
// ---------------------------------------------------------------------------

__global__ void power_spectrum_kernel(
    const cufftComplex* __restrict__ fft_out,
    float*                           power,    // [n_frames, FFT_SIZE/2+1]
    int                              n_frames
) {
    int frame = blockIdx.x;
    int bin   = threadIdx.x;   // 0..FFT_SIZE/2

    if (frame >= n_frames || bin > FFT_SIZE / 2) return;

    cufftComplex c = fft_out[frame * FFT_SIZE + bin];
    power[frame * (FFT_SIZE / 2 + 1) + bin] = c.x * c.x + c.y * c.y;
}

// ---------------------------------------------------------------------------
// Mel filterbank application kernel
// Applies [n_mels × fft_bins] filter matrix to power spectrum.
//
// Grid:  (n_frames,)
// Block: (N_MELS,)        — one thread per mel bin
// ---------------------------------------------------------------------------

__global__ void mel_filterbank_kernel(
    const float* __restrict__ power,         // [n_frames, fft_bins]
    const float* __restrict__ filterbank,    // [n_mels,   fft_bins]
    float*                    mel_spec,      // [n_frames, n_mels]
    int                       n_frames,
    int                       fft_bins
) {
    int frame   = blockIdx.x;
    int mel_bin = threadIdx.x;

    if (frame >= n_frames || mel_bin >= N_MELS) return;

    const float* pw  = power      + frame   * fft_bins;
    const float* flt = filterbank + mel_bin * fft_bins;

    float acc = 0.0f;
#pragma unroll 4
    for (int k = 0; k < fft_bins; ++k)
        acc += pw[k] * flt[k];

    mel_spec[frame * N_MELS + mel_bin] = acc;
}

// ---------------------------------------------------------------------------
// Log compression + normalisation kernel
// log10(max(x, 1e-10)) then:
//   1. Subtract (max - 8.0) to compress dynamic range
//   2. Add 4.0, divide by 4.0  → [-1, 1] ish range
// Writes FP16 output transposed to [n_mels, n_frames] (Whisper convention).
//
// Grid:  (n_frames / 256 + 1, n_mels)
// Block: (256, 1)
// ---------------------------------------------------------------------------

__global__ void log_norm_kernel(
    const float* __restrict__ mel_spec,   // [n_frames, n_mels]  float32
    __half*                   output,     // [n_mels,   n_frames] float16
    float                     log_max,    // pre-computed log max across whole spectrogram
    int                       n_frames
) {
    int frame   = blockIdx.x * blockDim.x + threadIdx.x;
    int mel_bin = blockIdx.y;

    if (frame >= n_frames || mel_bin >= N_MELS) return;

    float v = mel_spec[frame * N_MELS + mel_bin];
    v = log10f(fmaxf(v, 1e-10f));
    v = fmaxf(v, log_max - 8.0f);
    v = (v + 4.0f) / 4.0f;

    // Transpose: output[mel_bin, frame]
    output[mel_bin * n_frames + frame] = __float2half(v);
}

// Per-spectrogram log max reduction kernel (one block)
__global__ void reduce_log_max_kernel(
    const float* __restrict__ mel_spec,
    float*                    log_max_out,
    int                       n_elements
) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float local_max = -1e38f;
    for (int i = tid; i < n_elements; i += blockDim.x) {
        float v = log10f(fmaxf(mel_spec[i], 1e-10f));
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    if (tid % 32 == 0) smem[tid / 32] = local_max;
    __syncthreads();

    if (tid < blockDim.x / 32) {
        local_max = smem[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    if (tid == 0) *log_max_out = local_max;
}

// ---------------------------------------------------------------------------
// Host launcher: full mel spectrogram pipeline
//
// input: float32 CPU or CUDA tensor of shape [n_samples]
// returns: float16 CUDA tensor of shape [1, 128, n_frames]
// ---------------------------------------------------------------------------

torch::Tensor compute_mel_spectrogram(const torch::Tensor& audio_cpu) {
    static float*        d_window      = nullptr;
    static float*        d_filterbank  = nullptr;
    static cufftHandle   fft_plan      = 0;
    static bool          initialised   = false;

    int n_samples = audio_cpu.numel();
    int n_frames  = (n_samples - N_FFT) / HOP_LENGTH + 1;
    int fft_bins  = FFT_SIZE / 2 + 1;

    // ------------------------------------------------------------------
    // One-time initialisation: upload window + filterbank, create plan
    // ------------------------------------------------------------------
    if (!initialised) {
        auto hann  = make_hann_window(N_FFT);
        auto fb    = make_mel_filterbank(N_MELS, FFT_SIZE, SAMPLE_RATE, F_MIN, F_MAX);

        cudaMalloc(&d_window,     N_FFT         * sizeof(float));
        cudaMalloc(&d_filterbank, N_MELS * fft_bins * sizeof(float));
        cudaMemcpy(d_window,     hann.data(), N_FFT * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filterbank, fb.data(),   N_MELS * fft_bins * sizeof(float), cudaMemcpyHostToDevice);

        // Batched 1D FFT plan: n_frames transforms of size FFT_SIZE
        cufftPlan1d(&fft_plan, FFT_SIZE, CUFFT_C2C, n_frames);
        initialised = true;
    }

    // ------------------------------------------------------------------
    // Upload audio to GPU
    // ------------------------------------------------------------------
    auto audio_gpu = audio_cpu.to(torch::Device(torch::kCUDA), torch::kFloat32);

    // ------------------------------------------------------------------
    // Allocate intermediate buffers
    // ------------------------------------------------------------------
    cufftComplex* d_frames;
    float*        d_power;
    float*        d_mel;
    float*        d_log_max;

    cudaMalloc(&d_frames,  n_frames * FFT_SIZE * sizeof(cufftComplex));
    cudaMalloc(&d_power,   n_frames * fft_bins * sizeof(float));
    cudaMalloc(&d_mel,     n_frames * N_MELS   * sizeof(float));
    cudaMalloc(&d_log_max, sizeof(float));

    // ------------------------------------------------------------------
    // Step 1: framing + windowing
    // ------------------------------------------------------------------
    stft_frame_kernel<<<n_frames, FFT_SIZE>>>(
        audio_gpu.data_ptr<float>(),
        d_window,
        d_frames,
        n_samples,
        n_frames
    );

    // ------------------------------------------------------------------
    // Step 2: batched FFT (in-place)
    // ------------------------------------------------------------------
    cufftExecC2C(fft_plan, d_frames, d_frames, CUFFT_FORWARD);

    // ------------------------------------------------------------------
    // Step 3: power spectrum
    // ------------------------------------------------------------------
    power_spectrum_kernel<<<n_frames, fft_bins>>>(d_frames, d_power, n_frames);

    // ------------------------------------------------------------------
    // Step 4: mel filterbank
    // ------------------------------------------------------------------
    mel_filterbank_kernel<<<n_frames, N_MELS>>>(
        d_power, d_filterbank, d_mel, n_frames, fft_bins
    );

    // ------------------------------------------------------------------
    // Step 5: global log max (for normalisation)
    // ------------------------------------------------------------------
    int tpb = 256;
    int smem = (tpb / 32) * sizeof(float);
    reduce_log_max_kernel<<<1, tpb, smem>>>(d_mel, d_log_max, n_frames * N_MELS);

    // ------------------------------------------------------------------
    // Step 6: log + normalise → FP16, transpose to [n_mels, n_frames]
    // ------------------------------------------------------------------
    auto output = torch::empty({1, N_MELS, n_frames},
                               torch::dtype(torch::kFloat16).device(torch::kCUDA));

    float h_log_max;
    cudaMemcpy(&h_log_max, d_log_max, sizeof(float), cudaMemcpyDeviceToHost);

    dim3 grid((n_frames + 255) / 256, N_MELS);
    log_norm_kernel<<<grid, 256>>>(
        d_mel,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()) ,
        h_log_max,
        n_frames
    );

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cudaFree(d_frames);
    cudaFree(d_power);
    cudaFree(d_mel);
    cudaFree(d_log_max);

    return output;
}
