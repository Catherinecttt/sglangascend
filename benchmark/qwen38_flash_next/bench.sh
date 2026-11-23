#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
results=$(mktemp -d)
echo "Benchmark results: $results"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

run_bench() {
  local name=$1 output_tokens=$2 warmup_requests=$3
  echo "Running $name"
  python3 -m sglang.benchmark.serving \
    --backend sglang --base-url "http://127.0.0.1:${PORT:-30000}" \
    --dataset-name custom --dataset-path "$here/prompts.jsonl" \
    --num-prompts 32 --max-concurrency 32 --request-rate inf \
    --temperature 0 --seed 42 --sharegpt-output-len "$output_tokens" \
    --warmup-requests "$warmup_requests" --output-details --output-file "$results/$name.jsonl"
}

run_bench warmup-16 16 2
run_bench warmup-256 256 0
for round in 1 2 3; do
  run_bench "round-$round" 4096 8
done

python3 - "$results" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean

rounds = []
for i in (1, 2, 3):
    path = Path(sys.argv[1]) / f"round-{i}.jsonl"
    row = json.loads(path.read_text())
    counts = row["completed"], row["total_input_tokens"], row["total_output_tokens"]
    if counts != (32, 336000, 131072) or any(row["errors"]):
        raise SystemExit(f"Round {i} workload mismatch or failed requests: {counts}. See {path}")
    print(f"Round {i}: {row['duration']:.2f} s")
    rounds.append(row)

def warm_mean(key):
    return mean(row[key] for row in rounds[1:])

print(f"Rounds 2-3 mean: batch {warm_mean('duration'):.2f} s, "
      f"TTFT {warm_mean('mean_ttft_ms') / 1000:.2f} s, TPOT {warm_mean('mean_tpot_ms'):.2f} ms, "
      f"output {warm_mean('output_throughput'):.1f} tok/s")
PY
