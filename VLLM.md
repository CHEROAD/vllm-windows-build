# vLLM v0.17.1 Windows — Qwen 3.5 on RTX 3090

## One-Click Server

Double-click to start:

```
E:\vllm-windows-build-v2\start_server.bat
```

This loads Qwen3.5-4B on `http://127.0.0.1:8100` with:
- 2048 context, 0.92 VRAM utilization, prefix caching
- Thinking disabled at template level (no wasted tokens)
- 96,096 token KV cache, ~73x concurrency

## Quick Start (manual)

```batch
E:\vllm-windows-build-v2\venv\Scripts\activate.bat
set VLLM_HOST_IP=127.0.0.1
set CUDA_VISIBLE_DEVICES=0

python E:\vllm-windows-build-v2\serve.py
```

## Loading Qwen 3.5-4B on RTX 3090 (24 GB)

### As a Python LLM object

```python
import os
os.environ['VLLM_HOST_IP'] = '127.0.0.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import LLM, SamplingParams

llm = LLM(
    model='E:/AI_game/huihui',
    max_model_len=2048,
    gpu_memory_utilization=0.92,
    enforce_eager=True,
    trust_remote_code=True,
    enable_prefix_caching=True,
)

# Use llm.chat() with enable_thinking=False to prevent think tokens
params = SamplingParams(max_tokens=256, temperature=0.7)
output = llm.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    sampling_params=params,
    chat_template_kwargs={"enable_thinking": False},
)
print(output[0].outputs[0].text)
```

### As an OpenAI-compatible server (serve.py)

Server runs on `http://127.0.0.1:8100` with these endpoints:

```
GET  /health                  → {"status": "ok"}
GET  /v1/models               → model list
POST /v1/chat/completions     → OpenAI-compatible chat (thinking disabled)
```

Query it:

```batch
curl http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"huihui\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":256}"
```

## Current Server Settings (serve.py)

| Setting | Value |
|---------|-------|
| Model | Qwen3.5-4B (BF16 safetensors) `E:\AI_game\huihui` |
| max_model_len | 2048 |
| gpu_memory_utilization | 0.92 |
| max_num_seqs | 64 |
| enforce_eager | True |
| enable_prefix_caching | True |
| enable_thinking | False (at chat template level — no wasted compute) |
| Port | 8100 |

## RTX 3090 Performance Profile

| Metric | Value |
|--------|-------|
| Weights VRAM | 8.61 GiB |
| KV cache VRAM | ~11.74 GiB |
| KV cache tokens | 96,096 |
| KV block size | 528 tokens (aligned for GDN mamba cache) |
| Max concurrency @ 2048 tokens | ~73x |
| Max concurrency @ 1024 tokens | ~146x |
| Max concurrency @ 512 tokens | ~292x |

## Thinking Mode

Qwen 3.5 has a `<think>` mode that generates chain-of-thought before answering. For agent workloads this wastes tokens and time. `serve.py` disables it at the template level via `chat_template_kwargs={"enable_thinking": False}` — the model never generates think tokens, so no compute is wasted.

If you need thinking for a specific use case, remove that parameter.

## Prefix Caching

Enabled by default. Reuses KV cache for shared prompt prefixes across requests. If your application sends the same system prompt with every request (agents, chatbots), this avoids recomputing that prefix each time.

With Qwen 3.5's GDN layers, vLLM uses "align" mode for the mamba-style cache, which sets block sizes to 528 tokens. Experimental but tested and working.

## First Run Note

First inference after loading is slow (1-2 min) because Triton JIT-compiles the Gated Delta Network kernels. Cached after first run — subsequent requests are fast.

## RTX 3060 (12 GB)

BF16 Qwen3.5-4B uses 8.61 GiB for weights alone — no room for KV cache on 12 GB. Use:
- A quantized (AWQ/GPTQ) version of Qwen3.5-4B
- Qwen3.5-0.8B
- A smaller model under ~4 GB weights

## Key Files

| File | Purpose |
|------|---------|
| `E:\vllm-windows-build-v2\start_server.bat` | One-click server launcher |
| `E:\vllm-windows-build-v2\serve.py` | FastAPI server (edit settings here) |
| `E:\vllm-windows-build-v2\venv\` | Python 3.10 venv with vLLM + Triton |
| `E:\AI_game\huihui\` | Qwen3.5-4B model weights |

## Environment

- vLLM: 0.17.1+cu126
- PyTorch: 2.10.0+cu126
- Triton: 3.6.0 (triton-windows)
- Python: 3.10
- CUDA: 12.6
- OS: Windows 10
