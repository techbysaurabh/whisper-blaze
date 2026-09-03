"""
setup.py
Build whisper_blaze_kernels CUDA extension.

Usage:
    pip install . --no-build-isolation
    pip install -e . --no-build-isolation
"""

import os
from setuptools import setup, find_packages

# Use relative paths so sdist installs work from any directory
CSRC    = os.path.join("csrc")
INCLUDE = os.path.join(CSRC, "include")
KERNELS = os.path.join(CSRC, "kernels")
OPS     = os.path.join(CSRC, "ops")


def get_ext_modules():
    """Build the CUDA extension, or return [] if torch/nvcc are unavailable."""
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    except ImportError:
        print(
            "WARNING: torch not found — skipping CUDA kernel compilation.\n"
            "Install PyTorch first, then reinstall with:\n"
            "  pip install . --no-build-isolation"
        )
        return [], {}

    cuda_sources = [
        os.path.join(OPS,     "bindings.cpp"),
        os.path.join(KERNELS, "fp8_quantize.cu"),
        os.path.join(KERNELS, "layernorm_fused.cu"),
        os.path.join(KERNELS, "mel_spectrogram.cu"),
        os.path.join(KERNELS, "gemm_hopper.cu"),
        os.path.join(KERNELS, "flash_attn_hopper.cu"),
    ]

    nvcc_flags = [
        "-arch=sm_90a",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "--ptxas-options=-v",
        "--ptxas-options=-warn-lmem-usage",
        "-lineinfo",
        "-DCUDA_HAS_FP8",
    ]

    ext = CUDAExtension(
        name="whisper_blaze_kernels",
        sources=cuda_sources,
        include_dirs=[INCLUDE],
        extra_compile_args={
            "cxx":  ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
        extra_link_args=["-lcufft"],
    )

    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = get_ext_modules()

setup(
    name="whisper-blaze",
    version="0.1.17",
    description="High-throughput batched Whisper large-v3 serving on H100, with VRAM capping and a fused Hopper LayerNorm kernel",
    long_description=open("README.md").read()
    if os.path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "numpy",
    ],
    extras_require={
        "audio": ["torchaudio>=2.1.0", "librosa>=0.10.0"],
        "dev":   ["pytest", "pytest-benchmark"],
    },
)
