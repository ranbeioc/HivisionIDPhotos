#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 IMAGE INPUT REPORT [ITERATIONS] [BENCHMARK_SCRIPT]" >&2
  exit 64
fi

image=$1
input=$2
report=$3
iterations=${4:-5}
benchmark_script=${5:-}
container_name="xhalo-hivision-benchmark-$$"

if [[ ! -f "$input" || "$input" != /tmp/xhalo-hivision-benchmark-*/* ]]; then
  echo "input must be an existing file in a dedicated /tmp/xhalo-hivision-benchmark-* directory" >&2
  exit 64
fi
if [[ "$report" != /tmp/xhalo-hivision-benchmark-*/* ]]; then
  echo "report must stay in the dedicated benchmark directory" >&2
  exit 64
fi
if [[ ! "$iterations" =~ ^[0-9]+$ ]] || (( iterations < 2 || iterations > 20 )); then
  echo "iterations must be an integer between 2 and 20" >&2
  exit 64
fi

benchmark_mount=()
if [[ -n "$benchmark_script" ]]; then
  if [[ ! -f "$benchmark_script" || "$benchmark_script" != /tmp/xhalo-hivision-benchmark-*/* ]]; then
    echo "benchmark script override must stay in the dedicated benchmark directory" >&2
    exit 64
  fi
  benchmark_mount=(--volume "$benchmark_script:/app/scripts/benchmark_models.py:ro")
fi

sudo -n docker image inspect "$image" --format 'image={{.Id}} size={{.Size}} architecture={{.Architecture}}'
echo "before=$(date -Is)"
free -h
uptime

set +e
sudo -n timeout 300 docker run --rm \
  --name "$container_name" \
  --network none \
  --cpus 2 \
  --memory 6g \
  --pids-limit 256 \
  --read-only \
  --tmpfs /tmp:size=512m,mode=1777 \
  --security-opt no-new-privileges:true \
  --cap-drop ALL \
  --volume "$input:/fixtures/input.png:ro" \
  "${benchmark_mount[@]}" \
  "$image" \
  python scripts/benchmark_models.py \
    --image /fixtures/input.png \
    --mime image/png \
    --matting-model modnet_photographic_portrait_matting \
    --face-model mtcnn \
    --face-model retinaface-resnet50 \
    --iterations "$iterations" > "$report"
status=$?
set -e

cat "$report"
echo "benchmark_exit=$status"
echo "after=$(date -Is)"
free -h
uptime

if sudo -n docker ps --quiet --filter "name=$container_name" | grep -q .; then
  echo "benchmark container still running unexpectedly" >&2
  exit 70
fi
exit "$status"
