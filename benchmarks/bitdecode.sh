export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

# Now run your benchmark
CUDA_VISIBLE_DEVICES=1 vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path ./sonnet.txt \
  --num-prompts 5 \
  --input-len 8192 \
  --output-len 8192 \
  --max-model-len 25000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50