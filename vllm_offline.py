# vLLM v0.17.1 Windows — Qwen 3.5 on RTX 3090
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
    enable_auto_tool_choice=True,
    tool_call_parser="qwen3_coder"
)

# Use llm.chat() with enable_thinking=False to prevent think tokens
params = SamplingParams(max_tokens=256, temperature=0.7)
output = llm.chat(
    messages=[{"role": "user", "content": "Hello!"}],
    sampling_params=params,
    chat_template_kwargs={"enable_thinking": False},
)
print(output[0].outputs[0].text)