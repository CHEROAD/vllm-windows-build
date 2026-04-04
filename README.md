# vllm-windows-build

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue)
![vLLM: v0.17.1](https://img.shields.io/badge/vLLM-v0.17.1-orange)
![CUDA: 12.6](https://img.shields.io/badge/CUDA-12.6-76B900)
![Python: 3.10](https://img.shields.io/badge/Python-3.10-3776AB)
![Triton: 3.6](https://img.shields.io/badge/Triton-3.6-red)
![TurboQuant: KV Cache](https://img.shields.io/badge/TurboQuant-KV%20Cache%20Compression-purple)

**Native Windows build of vLLM — no WSL, no Docker, no Linux VM.** Now with **Triton support**, **Qwen 3.5** (Gated Delta Networks), and **TurboQuant KV cache compression** (per-attention-head FP8 quantization).

vLLM is the most popular open-source LLM serving engine, but it officially only supports Linux. This repo provides both a **pre-built wheel** (just download and install) and a complete patchset for compiling vLLM natively on Windows with full CUDA acceleration.

## Releases

| Release | vLLM | PyTorch | Triton | Models | Download |
|---------|------|---------|--------|--------|----------|
| **v0.17.1-win (NEW)** | 0.17.1 | 2.10.0+cu126 | 3.6.0 | Qwen 3.5, Qwen 3, Llama 4, all v0.14 models | [Download](https://github.com/rookiemann/vllm-windows-build/releases/tag/v0.17.1-win) |
| v0.14.2-win | 0.14.2 | 2.9.1+cu126 | N/A | Qwen 2.5, Llama 3.x, Phi-4, xLAM | [Download](https://github.com/rookiemann/vllm-windows-build/releases/tag/v0.14.2-win) |

### What's new in v0.17.1

- **Triton on Windows** — first native Windows build with Triton kernel support via [triton-windows](https://github.com/triton-lang/triton-windows)
- **Qwen 3.5 support** — Gated Delta Network (GDN) linear attention layers run via Triton kernels
- **FlashAttention 2 + 4** — FA2 compiled, FA4 CuteDSL support
- **PyTorch 2.10.0** — latest stable with CUDA 12.6
- **253 CUDA kernels compiled** — all passing on MSVC 2022
- **TurboQuant KV cache compression** — per-attention-head FP8 quantization Triton kernels ported to SM 8.6, integrated via compressed-tensors/LLM-Compressor framework

---

## Quick Start — Pre-built Wheel (v0.17.1)

### 1. Install

Download **[vllm-0.17.1+cu126-cp310-cp310-win_amd64.whl](https://github.com/rookiemann/vllm-windows-build/releases/tag/v0.17.1-win)** from the Releases page, then:

```batch
:: Create a Python 3.10 virtual environment
py -3.10 -m venv venv
venv\Scripts\activate

:: Install PyTorch 2.10.0 with CUDA 12.6
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

:: Install Triton for Windows
pip install triton-windows

:: Install the pre-built vLLM wheel
pip install vllm-0.17.1+cu126-cp310-cp310-win_amd64.whl

:: Install optional structured output deps
pip install "llguidance>=1.3.0,<1.4.0" "xgrammar==0.1.29"
```

### 2. Run

```python
import os
os.environ['VLLM_HOST_IP'] = '127.0.0.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import LLM, SamplingParams

llm = LLM(
    model='E:\\models\\Qwen3.5-4B',
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    trust_remote_code=True,
)

params = SamplingParams(max_tokens=256, temperature=0.7)
output = llm.generate(['Hello!'], sampling_params=params)
print(output[0].outputs[0].text)
```

### 3. Test it

```batch
curl http://127.0.0.1:8100/health
:: Returns: {"status":"ok"}

curl http://127.0.0.1:8100/v1/models
:: Returns: {"object":"list","data":[{"id":"Qwen3.5-4B",...}]}
```

> **Note:** First inference after loading is slow (1-2 min) because Triton JIT-compiles the GDN kernels. Subsequent requests use cached kernels and are fast.

---

## Quick Start — v0.14.2 One-Click Installer (Legacy)

For the older v0.14.2 release with the one-click `launch.bat` installer, see the [v0.14.2-win release](https://github.com/rookiemann/vllm-windows-build/releases/tag/v0.14.2-win).

---

## Build from Source (v0.17.1)

For developers who want to compile vLLM themselves (30-45 min build time).

### Step 1: Clone this repo and vLLM source

```batch
git clone https://github.com/rookiemann/vllm-windows-build.git
cd vllm-windows-build
git clone --depth 1 --branch v0.17.1 https://github.com/vllm-project/vllm.git vllm-source
cd vllm-source
git apply ..\vllm-windows-v2.patch
```

### Step 2: Create a Python environment

```batch
py -3.10 -m venv venv
venv\Scripts\activate

:: Install PyTorch 2.10.0 with CUDA 12.6
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

:: Install Triton for Windows and build dependencies
pip install triton-windows wheel "setuptools>=77.0.3,<81.0.0" setuptools-scm cmake ninja packaging numpy
```

### Step 3: Patch PyTorch header (required for PyTorch 2.10 + MSVC)

Windows SDK defines `small` as `char` in `rpcndr.h`, which breaks a PyTorch CUDA header:

```batch
:: Add #undef small to the top of this file (after #pragma once):
:: venv\Lib\site-packages\torch\include\c10\cuda\CUDACachingAllocator.h
```

Add this block after `#pragma once`:
```cpp
#ifdef _MSC_VER
#ifdef small
#undef small
#endif
#endif
```

### Step 4: Build vLLM

Open a **Visual Studio Developer Command Prompt** (or run `vcvars64.bat`), then:

```batch
venv\Scripts\activate

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set TORCH_CUDA_ARCH_LIST=8.6
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=8
set SETUPTOOLS_SCM_PRETEND_VERSION=0.17.1

cd vllm-source
pip install . --no-build-isolation -v
```

Change `TORCH_CUDA_ARCH_LIST` to match your GPU (see table below). Compiles 253 CUDA/C++ files — expect 30-45 minutes with `MAX_JOBS=8`.

### Step 5: Run

```python
import os
os.environ['VLLM_HOST_IP'] = '127.0.0.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import LLM, SamplingParams
llm = LLM(model='E:\\models\\Qwen3.5-4B', max_model_len=4096,
           gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True)
output = llm.generate(['Hello!'], sampling_params=SamplingParams(max_tokens=100))
print(output[0].outputs[0].text)
```

---

## Usage

### Why the Launcher?

vLLM's built-in `vllm.entrypoints.openai.api_server` uses `AsyncMPClient` with ZMQ multiprocessing, which doesn't work on Windows (no `fork`, no IPC sockets). The included `vllm_launcher.py` works around this by:

1. Stubbing `uvloop` (not available on Windows)
2. Using vLLM's synchronous `LLM` class (which uses `InprocClient` — the patched in-process engine)
3. Wrapping it in a lightweight FastAPI server with OpenAI-compatible `/v1/chat/completions` endpoint
4. Supporting both streaming (SSE) and non-streaming responses

### Running the Server

```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

python vllm_launcher.py ^
    --model E:\models\Qwen2.5-1.5B-Instruct ^
    --port 8100 ^
    --gpu-memory-utilization 0.90 ^
    --max-num-seqs 128 ^
    --enforce-eager
```

| Flag | Default | Description |
|------|---------|------------|
| `--model` | (interactive) | HuggingFace model path or ID. Omit for interactive selector |
| `--models-dir` | (auto-scan) | Additional directory to scan for models |
| `--port` | 8100 | Server port |
| `--host` | 127.0.0.1 | Bind address |
| `--gpu-memory-utilization` | 0.6 | Fraction of GPU VRAM to pre-allocate (0.1 - 1.0). Safe to use 0.90-0.92 on dedicated GPU servers (display driver reserves ~80 MB) |
| `--max-model-len` | (auto) | Max sequence length |
| `--max-num-seqs` | 64 | Max concurrent sequences in the KV cache. 128 is viable at 0.90+ memory utilization |
| `--enforce-eager` | off | **Required on Windows.** Disables CUDA graphs (needs Triton, which is Linux-only) |
| `--tensor-parallel-size` | 1 | Number of GPUs (must be 1 on Windows) |
| `--gpu-id` | (all) | GPU device index (e.g. `1` for second GPU) |
| `--enable-prefix-caching` | off | Reuse KV cache for shared prefixes |
| `--num-scheduler-steps` | 1 | Multi-step scheduling (decode N tokens before CPU sync) |
| `--max-num-batched-tokens` | (auto) | Max tokens per scheduler iteration |
| `--task` | generate | `generate` for text, `embed` for embeddings |
| `--trust-remote-code` | off | Allow custom model code (needed for some models) |

### API Endpoints

Once running, the server exposes:

```
GET  /health                    # Returns {"status": "ok"}
GET  /v1/models                 # List loaded models
POST /v1/chat/completions       # OpenAI-compatible chat completions (with tool calling)
POST /v1/embeddings             # Text embeddings (start with --task embed)
```

### Multi-GPU: Running Two Instances

On multi-GPU systems, you can run generation and embedding servers simultaneously — each pinned to a different GPU:

**Terminal 1 — Text generation (GPU 0):**
```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

python vllm_launcher.py ^
    --model E:\models\xLAM-2-1b-fc-r ^
    --port 8100 ^
    --gpu-id 0 ^
    --gpu-memory-utilization 0.92 ^
    --max-num-seqs 128 ^
    --enforce-eager
```

**Terminal 2 — Embeddings (GPU 1):**
```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1

python vllm_launcher.py ^
    --model E:\models\nomic-embed-text-v1.5 ^
    --task embed ^
    --port 8101 ^
    --gpu-id 1 ^
    --trust-remote-code ^
    --enforce-eager
```

Both servers run independently. Use `--gpu-id` (not `CUDA_VISIBLE_DEVICES`) to avoid environment variable conflicts between terminals.

> **Note:** Embedding models must be in FP16 safetensors format — GGUF is not supported for BERT-based embedding models like nomic-embed.

### Calling the API

```python
import requests

response = requests.post("http://127.0.0.1:8100/v1/chat/completions", json={
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": False,
})
print(response.json()["choices"][0]["message"]["content"])
```

Works with any OpenAI-compatible client library (openai-python, LangChain, LiteLLM, etc.) — just point the `base_url` at `http://127.0.0.1:8100/v1`.

### VRAM and Memory

vLLM pre-allocates a large KV cache on startup using PagedAttention. This is by design — it's how vLLM achieves high concurrent throughput. The tradeoff is that even a small model uses significant VRAM:

| Model Size | Weights | KV Cache (0.6 util) | KV Cache (0.92 util) | Card |
|-----------|---------|-------------------|---------------------|------|
| 1.5B FP16 | ~3 GB | ~12 GB | ~19 GB | 24 GB RTX 3090 |
| 1.5B Q8 | ~1.8 GB | ~12 GB | ~20 GB | 24 GB RTX 3090 |
| 7B Q4 | ~4 GB | ~12 GB | ~18 GB | 24 GB RTX 3090 |

On a dedicated inference machine (no desktop workload on the GPU), you can safely use `--gpu-memory-utilization 0.92` — the Windows display driver reserves ~80 MB regardless. Lower the value if you hit OOM during sampler warmup. Lower `--max-num-seqs` to reduce KV cache size.

---

## What's in the Patch

26 files modified, +254 lines / -121 lines. Every change is guarded behind `#ifdef _MSC_VER`, `sys.platform == "win32"`, or similar checks — nothing breaks on Linux.

### Build System Fixes

| File | What Changed |
|------|-------------|
| `CMakeLists.txt` | Force CUDA toolkit from `CUDA_HOME` (prevents wrong toolkit when multiple versions installed), add MSVC-specific include paths, `/Zc:preprocessor` flag, link `CUDA::cublas` |
| `cmake/utils.cmake` | Quote paths with spaces for `file(REAL_PATH)` |
| `setup.py` | Allow Windows CUDA builds (was hardcoded to `"empty"`), backslash→forward-slash paths for CMake, `nvcc.exe` detection |

### CUDA Kernel Fixes (MSVC Compatibility)

| File | What Changed |
|------|-------------|
| `csrc/quantization/utils.cuh` | Variable template → function template (MSVC can't apply `__host__`/`__device__` to variable templates) |
| `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | Replaced nested `BOOL_SWITCH` lambdas with explicit template dispatch (MSVC doesn't propagate constexpr through lambda captures) |
| `csrc/quantization/activation_kernels.cu` | Designated initializers `.x = val` → positional `{val}`, `__int128_t` → `int4` |
| `csrc/core/math.hpp` | `__builtin_clz` → portable bit-twiddling replacement |
| `csrc/quantization/awq/gemm_kernels.cu` | `__asm__ __volatile__` → `asm volatile` (MSVC PTX inline assembly syntax) |
| `csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu` | `or` keyword → `\|\|` operator |
| `csrc/quantization/fused_kernels/layernorm_utils.cuh` | `quant_type_max_v<T>` → `quant_type_max_v<T>()` (variable template → function template call) |
| `csrc/quantization/fused_kernels/quant_conversions.cuh` | Same variable template fix |
| `csrc/quantization/w8a8/fp8/common.cu`, `common.cuh` | Same variable template fix |
| `csrc/quantization/marlin/sparse/common/base.h` | `typedef unsigned int uint;` for MSVC |
| `csrc/attention/merge_attn_states.cu` | `std::isinf` → `isinf` (CUDA device context), add `uint` typedef |
| `csrc/cumem_allocator.cpp` | `ssize_t` → `SSIZE_T` via `BaseTsd.h` |
| `csrc/fused_qknorm_rope_kernel.cu` | Add `uint` typedef for MSVC |
| `csrc/gptq_marlin/generate_kernels.py` | Flat `if` chains instead of `else if` (MSVC C1061 "blocks nested too deeply" with 700+ branches) |
| `csrc/moe/marlin_moe_wna16/generate_kernels.py` | Same flat `if` fix |

### Runtime Fixes (Windows Platform)

| File | What Changed |
|------|-------------|
| `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `fcntl.flock()` → `msvcrt.locking()` (Unix file locking API doesn't exist on Windows) |
| `vllm/utils/network_utils.py` | ZMQ IPC transport → `tcp://127.0.0.1` (Unix domain sockets not available) |
| `vllm/utils/system_utils.py` | Force `spawn` multiprocessing (`fork` doesn't exist on Windows) |
| `vllm/v1/engine/core_client.py` | Force `InprocClient` (multiprocess ZMQ fails with `spawn` context) |
| `vllm/distributed/parallel_state.py` | **Biggest fix.** Gloo TCP/UV transport isn't compiled in PyTorch Windows builds. Uses PyTorch's `FakeProcessGroup` backend with `FileStore` for single-GPU operation. |
| `vllm/entrypoints/openai/api_server.py` | Guard `SO_REUSEPORT` (doesn't exist on Windows) |
| `requirements/cuda.txt` | Comment out `flashinfer-python` (not available on Windows) |

---

## Required Environment Variables

```batch
:: v0.17.1 — only VLLM_HOST_IP is required
set VLLM_HOST_IP=127.0.0.1

:: Optional — pin to a specific GPU
set CUDA_VISIBLE_DEVICES=0

:: v0.14.2 also requires:
:: set VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

> **Warning:** Avoid setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` on multi-GPU systems — it can reorder GPU indices and cause vLLM to target the wrong device. Use `--gpu-id` or `CUDA_VISIBLE_DEVICES` instead.

---

## System Requirements

| Component | v0.17.1 | v0.14.2 |
|-----------|---------|---------|
| OS | Windows 10 or 11 (64-bit) | Same |
| GPU | NVIDIA with Compute Capability 7.5+ | 7.0+ |
| CUDA Toolkit | 12.6 | 12.6 |
| Compiler | Visual Studio 2022 with C++ Desktop workload | Same |
| Python | 3.10.x | 3.10.x |
| PyTorch | 2.10.0+cu126 | 2.9.1+cu126 |
| Triton | 3.6.0 (triton-windows) | N/A |
| RAM | 32 GB recommended | Same |
| Disk | ~20 GB for build artifacts | Same |

---

## Performance Tips

### Use FP16 Safetensors Instead of GGUF

This is the single biggest performance optimization. vLLM's continuous batching and PagedAttention are designed for native tensor formats. GGUF works but bypasses these optimizations:

| Format | Throughput (32 concurrent) | Speedup |
|--------|---------------------------|---------|
| GGUF Q8 | ~2.0 decisions/s | 1x (baseline) |
| FP16 safetensors | ~11.9 decisions/s | **~6x faster** |

Benchmarked with Salesforce xLAM-2-1B on RTX 3090, 64 concurrent requests, `--gpu-memory-utilization 0.92`.

**How to get FP16 models:** Most HuggingFace model repos include both safetensors and GGUF. Download the safetensors version (the repo with `config.json`, `model.safetensors`, and `tokenizer.json` — not the GGUF quant repo).

### Optimal Launch Command

For maximum throughput on a dedicated RTX 3090:

```batch
set VLLM_ATTENTION_BACKEND=FLASH_ATTN
set VLLM_HOST_IP=127.0.0.1
set CUDA_VISIBLE_DEVICES=0

python vllm_launcher.py ^
    --model E:\models\your-model-fp16 ^
    --port 8100 ^
    --gpu-memory-utilization 0.92 ^
    --max-num-seqs 128 ^
    --max-model-len 4096 ^
    --enable-prefix-caching ^
    --enforce-eager
```

### Enable Prefix Caching

`--enable-prefix-caching` reuses KV cache for shared prompt prefixes. If your application sends similar system prompts across requests (common for agents and chatbots), this significantly reduces redundant computation.

---

## TurboQuant — KV Cache Compression

This build includes [TurboQuant](https://arxiv.org/abs/2501.06725)-style per-attention-head FP8 KV cache quantization, which compresses the KV cache to reduce VRAM usage and increase concurrency. The implementation is integrated two ways:

### 1. Triton Kernels (standalone, `turboquant/`)

Three Triton kernels from the TurboQuant paper, ported to SM 8.6 (RTX 3090 / Ampere):

| Kernel | File | Purpose |
|--------|------|---------|
| KV Update | `turboquant/triton_turboquant_kv_update.py` | Quantizes K/V to FP8 with per-head scales during cache insertion |
| Decode Attention | `turboquant/triton_turboquant_decode.py` | FP8 paged attention with per-head dequantization during decode |
| Modified Attention | `turboquant/triton_attn_modified.py` | Full attention with integrated FP8 KV quantization/dequantization |

Supporting files: `turboquant_kv_cache.py` (cache manager), `turboquant_metadata.py` (scale storage), `generate_turboquant_metadata.py` (calibration).

### 2. vLLM Integration (compressed-tensors framework)

Inside `vllm-source/`, the KV cache compression concepts are integrated through vLLM's compressed-tensors / LLM-Compressor framework:

- **`CompressedTensorsKVCacheMethod`** — per-attention-head FP8 quantization with TP-aware scale loading
- **Triton reshape-and-cache kernels** — FP8 quantization during KV cache insertion
- **Automatic detection** — `kv_cache_scheme` in model configs enables per-head quant scales

To use with a calibrated model checkpoint (LLM-Compressor format):

```python
llm = LLM(
    model='path/to/calibrated-model',
    kv_cache_dtype='fp8',
    enforce_eager=True,
)
```

---

## Known Limitations

- **Single GPU only** — no NCCL on Windows means no multi-GPU tensor parallelism. `world_size` must be 1.
- **No FlashInfer** — `flashinfer-python` doesn't publish Windows wheels.
- **No FA3 (Hopper)** — FlashAttention 3 has MSVC-incompatible nested macros. FA2 works for all SM < 90 GPUs. FA3 is only needed for H100/H200.
- **Triton JIT cold start** — First inference after loading a model with GDN layers (Qwen 3.5) takes 1-2 minutes while Triton compiles kernels. Cached after first run.
- **VRAM pre-allocation** — vLLM's PagedAttention aggressively pre-allocates KV cache. Lower `gpu_memory_utilization` or `max_num_seqs` if you hit OOM.

---

## Build Notes

The build compiles 370 CUDA/C++ source files. On an RTX 3090 system with 64 GB RAM and `MAX_JOBS=8`, expect roughly 30-45 minutes.

### Multiple CUDA Toolkits

If you have multiple CUDA versions installed (common if you use both PyTorch cu126 and the CUDA 13.1 compiler), the CMake patch forces toolkit selection from `CUDA_HOME`. Make sure this points at the version matching your PyTorch build:

```batch
:: Check which CUDA your PyTorch was built with
python -c "import torch; print(torch.version.cuda)"

:: Set CUDA_HOME to match
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

### GPU Compute Capability

Change `TORCH_CUDA_ARCH_LIST` to match your GPU:

| GPU Series | Compute Capability |
|-----------|-------------------|
| RTX 20xx (Turing) | 7.5 |
| RTX 30xx (Ampere) | 8.6 |
| RTX 40xx (Ada Lovelace) | 8.9 |
| RTX 50xx (Blackwell) | 12.0 |

---

## Directory Structure

**Pre-built release (what you download from Releases):**

```
vllm-windows-build/
├── launch.bat            # One-click launcher (start here!)
├── install.bat           # Auto-installs Python, PyTorch, vLLM
├── vllm_launcher.py      # OpenAI-compatible server with model selector
├── build_wheel.py        # Re-package vLLM into a wheel (advanced)
├── dist/
│   └── vllm-*.whl        # Pre-built vLLM wheel (380 MB)
├── vllm-windows.patch    # Complete git patch (for building from source)
├── build.bat             # Source build script
├── PATCHES.md            # Detailed per-file patch reference
├── LICENSE
└── README.md
```

**After first launch (auto-created by install.bat):**

```
vllm-windows-build/
├── python/               # Portable Python 3.10.11 (auto-downloaded)
│   ├── python.exe
│   ├── Scripts/pip.exe
│   └── Lib/site-packages/
├── launch.bat
├── install.bat
├── vllm_launcher.py
├── dist/
│   └── vllm-*.whl
└── ...
```

**Build from source layout:**

```
vllm-windows-build/
├── vllm-source/          # Patched vLLM source (git clone + git apply)
│   ├── csrc/             #   CUDA/C++ kernels (patched for MSVC)
│   ├── vllm/             #   Python package (patched for Windows)
│   ├── .deps/            #   Build dependencies (flash-attn source, etc.)
│   ├── build/            #   CMake build output (generated)
│   ├── setup.py          #   Patched build script
│   └── ...
├── venv/                 # Python virtual environment
│   └── Scripts/python.exe
├── turboquant/           # TurboQuant Triton kernels (SM 8.6)
│   ├── triton_turboquant_kv_update.py
│   ├── triton_turboquant_decode.py
│   ├── triton_attn_modified.py
│   ├── turboquant_kv_cache.py
│   ├── turboquant_metadata.py
│   └── generate_turboquant_metadata.py
├── vllm-windows.patch
├── vllm_launcher.py      # <-- run this to start the server
├── build.bat
└── ...
```

---

## Tested With

**Hardware:** RTX 3090 24 GB + RTX 3060 12 GB, Windows 10 Pro 10.0.19045, MSVC 19.43, CUDA 12.6

### v0.17.1 (PyTorch 2.10.0+cu126, Triton 3.6.0)

| Model | Format | Util | KV Cache | Max Concurrency | Notes |
|-------|--------|------|----------|----------------|-------|
| Qwen3.5-4B (huihui) | BF16 safetensors | 0.90 | 91,872 tokens | 63x @ 4096 ctx | GDN Triton kernels working |

### v0.14.2 (PyTorch 2.9.1+cu126)

| Model | Format | Util | Seqs | Throughput | Notes |
|-------|--------|------|------|-----------|-------|
| Salesforce xLAM-2-1B-fc-r | FP16 safetensors | 0.92 | 128 | 11.91 dec/s, 16,350 tok/s | Best balanced (tool calling) |
| Qwen2.5-1.5B-Instruct | GGUF Q8 | 0.6 | 64 | 202.8 tok/s | Baseline GGUF test |
| nomic-embed-text-v1.5 | FP16 safetensors | 0.92 | — | Embedding server | RTX 3060, `--task embed` |

Full benchmark results at [AI Character Engine](https://github.com/rookiemann/ai-player-engine).

---

## Credits

| Project | Description | License |
|---------|------------|---------|
| [vLLM](https://github.com/vllm-project/vllm) | High-throughput LLM serving engine | Apache 2.0 |
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework | BSD-3 |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | NVIDIA GPU computing | NVIDIA EULA |
| [Flash Attention](https://github.com/Dao-AILab/flash-attention) | Fast attention implementation | BSD-3 |
| [triton-windows](https://github.com/triton-lang/triton-windows) | Triton compiler for Windows | MIT |

Built with [Claude Opus 4.6](https://claude.ai) as pair-programming partner.

## License

MIT License. See [LICENSE](LICENSE) for details.

The patch modifies vLLM source code which is licensed under Apache 2.0. This repo contains only the patch (derivative work) and build tooling, not the full vLLM source.
