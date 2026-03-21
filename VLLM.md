# vLLM v0.17.1 Windows — Qwen 3.5 on RTX 3090

## Quick Start

Open a terminal and run:

```batch
E:\vllm-windows-build-v2\venv\Scripts\activate.bat
set VLLM_HOST_IP=127.0.0.1
set CUDA_VISIBLE_DEVICES=0

python -c "from vllm import LLM; print('vLLM ready')"
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
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
    trust_remote_code=True,
)

params = SamplingParams(max_tokens=256, temperature=0.7)
output = llm.generate(['Hello, how are you?'], sampling_params=params)
print(output[0].outputs[0].text)
```

### As an OpenAI-compatible server

Copy `vllm_launcher.py` from the v1 build or run vLLM's built-in server:

```batch
E:\vllm-windows-build-v2\venv\Scripts\activate.bat
set VLLM_HOST_IP=127.0.0.1
set CUDA_VISIBLE_DEVICES=0

python -m vllm.entrypoints.openai.api_server ^
    --model E:\AI_game\huihui ^
    --port 8100 ^
    --max-model-len 4096 ^
    --gpu-memory-utilization 0.90 ^
    --enforce-eager ^
    --trust-remote-code
```

Then query it:

```batch
curl http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"huihui\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":256}"
```

## RTX 3090 Performance Profile

| Setting | Value |
|---------|-------|
| Model | Qwen3.5-4B (BF16 safetensors) |
| Weights VRAM | 8.61 GiB |
| KV cache VRAM | ~11.26 GiB (at 0.90 util) |
| KV cache tokens | 91,872 |
| Max concurrency @ 4096 tokens | ~63x |
| Max concurrency @ 2048 tokens | ~126x |
| Max concurrency @ 512 tokens | ~504x |

## Tuning Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--max-model-len` | 4096 | Max sequence length per request. Lower = more concurrency |
| `--gpu-memory-utilization` | 0.90 | Safe max for dedicated GPU. Use 0.92 if no display connected |
| `--max-num-seqs` | 64 | Max concurrent sequences. Increase to 128 for high throughput |
| `--enforce-eager` | required | Disables CUDA graphs (Triton JIT handles this instead) |
| `--enable-prefix-caching` | off | Enable for shared system prompts (agents, chatbots) |
| `--trust-remote-code` | off | Required for some models with custom code |

## First Run Note

The first inference after loading will be slow (1-2 minutes) because Triton JIT-compiles the Gated Delta Network kernels. Subsequent requests use cached kernels and are fast.

## RTX 3060 (12 GB)

The BF16 Qwen3.5-4B model uses 8.61 GiB for weights alone, leaving virtually no room for KV cache on a 12 GB card. For the 3060, use either:
- A quantized (AWQ/GPTQ) version of Qwen3.5-4B
- The smaller Qwen3.5-0.8B model
- A different model under ~4 GB weights (e.g. Qwen2.5-1.5B)

## Environment

- vLLM: 0.17.1+cu126
- PyTorch: 2.10.0+cu126
- Triton: 3.6.0 (triton-windows)
- Python: 3.10
- CUDA: 12.6
- OS: Windows 10
