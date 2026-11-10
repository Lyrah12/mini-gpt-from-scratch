#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-pilot}"
data_dir="${MINIGPT_DATA:-data/processed}"
run_root="${MINIGPT_RUNS:-runs/cloud}"
mkdir -p "$run_root"
if [[ ! -f "$data_dir/manifest.json" ]]; then
  echo "Missing prepared data: $data_dir/manifest.json" >&2
  exit 1
fi

train_stage() {
  local config="$1" destination="$2" initial="${3:-}"
  local train_args=(-m minigpt.train --config "$config" --data "$data_dir" --output "$destination" --device cuda)
  if [[ -f "$destination/last.pt" ]]; then
    train_args+=(--resume "$destination/last.pt")
  elif [[ -n "$initial" ]]; then
    train_args+=(--init "$initial")
  fi
  if [[ "$mode" == "pilot" ]]; then
    train_args+=(--stop-after 30)
  fi
  python "${train_args[@]}" 2>&1 | tee -a "$run_root/console.log"
}

case "$mode" in
  pilot)
    train_stage configs/pretrain.json "$run_root/pretrain"
    echo "Pilot checkpoint saved. Review speed and memory before the full run."
    ;;
  full)
    train_stage configs/pretrain.json "$run_root/pretrain"
    train_stage configs/sft.json "$run_root/sft" "$run_root/pretrain/best.pt"
    train_stage configs/scratch_sft.json "$run_root/scratch"
    ;;
  val-eval|test-eval)
    split="val"; limit=200
    if [[ "$mode" == "test-eval" ]]; then split="test"; limit=500; fi
    python -m minigpt.evaluate --data "$data_dir" --split "$split" --limit "$limit" --output "$run_root/eval_${split}_lead"
    for name in pretrain sft scratch; do
      python -m minigpt.evaluate --data "$data_dir" --split "$split" --limit "$limit" \
        --checkpoint "$run_root/$name/best.pt" --output "$run_root/eval_${split}_$name" --device cuda
    done
    ;;
  export)
    if [[ ! -f "$run_root/continuations.json" ]]; then
      python -m scripts.collect_continuations --runs "$run_root" --data "$data_dir"
    fi
    python -m scripts.export_report --runs "$run_root" --data "$data_dir" --output report/generated
    ;;
  *) echo "Usage: bash scripts/run_cloud.sh {pilot|full|val-eval|test-eval|export}" >&2; exit 2 ;;
esac
