export VLLM_ATTENTION_BACKEND=BITDECODE
export VLLM_USE_V1=1

# Now run your benchmark
CUDA_VISIBLE_DEVICES=0 vllm bench throughput \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --dataset-name sonnet \
  --dataset-path ./sonnet.txt \
  --num-prompts 5 \
  --input-len 16384 \
  --output-len 8192 \
  --max-model-len 25000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.75