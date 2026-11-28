#!/usr/bin/env bash

set -o pipefail

# Host-wide tuning; requires appropriate privileges.
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# sysctl -w vm.swappiness=0
# sysctl -w kernel.numa_balancing=0

export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy http_proxy all_proxy
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/cann-9.1.0/set_env.sh || exit 1
source /usr/local/Ascend/nnal/atb/9.1.0/atb/set_env.sh || exit 1

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export ASCEND_USE_FIA=1

export PYTHONPATH="/home/chentong/flashnext-test/kernel-test.eMi0zb/python:/home/chentong/flashnext-test/sglang-main/python${PYTHONPATH:+:$PYTHONPATH}"

MODEL_PATH="/mnt/paas/weights/Qwen3.8-Flash-Next"

LOG_DIR="logs_serve"
mkdir -p -- "${LOG_DIR}" || exit 1
RUN_TIME="$(date +%Y%m%d_%H%M%S)"

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port 30000 \
    --tp-size 8 \
    --nnodes 1 \
    --attention-backend ascend \
    --device npu \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 52000 \
    --disable-radix-cache \
    --trust-remote-code \
    --max-running-requests 16 \
    --mem-fraction-static 0.85 \
    --enable-multimodal \
    --mm-attention-backend ascend_attn \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --mm-enable-dp-encoder \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    "$@" \
    2>&1 | tee "${LOG_DIR}/${RUN_TIME}.log"