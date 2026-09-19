#!/usr/bin/env bash
# Phase 0.5 -- overnight dense-model scaling sweep.
#
# Phase 0.4 established that MoE trains dense (active-parameter count does not predict
# training throughput) and that the 35B-A3B cannot reach seq 4096 on 64GB. This sweep
# measures dense models across parameter classes so the base-model choice can be made
# against a throughput/memory curve rather than a single point -- and so a future model
# of a given size can be slotted in without re-measuring.
#
# Robust by design: a model that OOMs or fails to download is recorded and the sweep
# continues. Nothing here should require attention overnight.

set -u  # deliberately NOT -e: one failed config must not kill the sweep

cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
OUT=artifacts/sweep
LOG=$OUT/sweep.log
mkdir -p "$OUT"

: > "$LOG"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# model_id | short label | seq lengths to try (space separated)
CONFIGS=(
  "mlx-community/Qwen3-14B-4bit|qwen3-14b|4096 8192"
  "mlx-community/Ministral-3-14B-Base-2512-4bit|ministral-14b-BASE|4096 8192"
  "mlx-community/Mistral-Small-3.1-Text-24B-Instruct-2503-4bit|mistral-24b-text|4096"
  "mlx-community/Qwen3-32B-4bit|qwen3-32b|4096"
)

log "=== ARGUS Phase 0.5 dense scaling sweep ==="
log "machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 ))GB"
log ""

# ---------- stage 1: download everything first ----------
# Downloading up front means the benchmark phase never stalls on network, and a
# network failure surfaces before any GPU time is spent.
for cfg in "${CONFIGS[@]}"; do
  MODEL="${cfg%%|*}"; rest="${cfg#*|}"; LABEL="${rest%%|*}"
  log "downloading $LABEL ($MODEL)"
  HF_HUB_ENABLE_HF_TRANSFER=1 $PY - "$MODEL" <<'PYEOF' >>"$LOG" 2>&1
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1],
                      allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*'])
    print("  download OK")
except Exception as e:
    print(f"  DOWNLOAD FAILED: {type(e).__name__}: {e}")
PYEOF
done

log ""
log "=== downloads done; starting benchmarks ==="
log ""

# ---------- stage 2: benchmark ----------
for cfg in "${CONFIGS[@]}"; do
  MODEL="${cfg%%|*}"; rest="${cfg#*|}"; LABEL="${rest%%|*}"; SEQS="${rest#*|}"
  for S in $SEQS; do
    TAG="${LABEL}_seq${S}"
    log "--- $TAG ---"
    # 10-minute sustain: long enough to expose thermal drift, short enough that the
    # whole sweep finishes overnight.
    caffeinate -i $PY -u scripts/bench_mlx.py \
        --model "$MODEL" \
        --seq-len "$S" \
        --warmup-steps 2 \
        --steady-steps 8 \
        --sustain-minutes 10 \
        --no-compile \
        --out "$OUT/${TAG}.json" >"$OUT/${TAG}.log" 2>&1
    RC=$?
    if [ $RC -eq 0 ]; then
      grep -E "steady-state|sustained|peak memory|projected" "$OUT/${TAG}.log" \
        | sed 's/^/    /' | tee -a "$LOG"
    else
      REASON=$(grep -oE "Resource limit \([0-9]+\) exceeded|RuntimeError.*|OutOfMemory.*" \
               "$OUT/${TAG}.log" | head -1)
      log "    FAILED (rc=$RC) ${REASON:-see $OUT/${TAG}.log}"
      # Record the failure as data -- an OOM at a given size/context is a result.
      $PY -c "
import json,sys
json.dump({'model':'$MODEL','label':'$LABEL','seq_len':$S,'failed':True,
           'reason':'''${REASON:-unknown}'''}, open('$OUT/${TAG}.json','w'), indent=2)
"
    fi
    log ""
  done
done

log "=== sweep complete ==="
log "results: $OUT/*.json"
