#!/usr/bin/env bash
# 在云端按阶段执行试跑、正式训练、评测和结果导出。
set -euo pipefail
# 切换到项目目录，并读取运行模式与可选的数据路径设置。
cd "$(dirname "$0")/.."
mode="${1:-pilot}"
data_dir="${MINIGPT_DATA:-data/processed}"
run_root="${MINIGPT_RUNS:-runs/cloud}"
mkdir -p "$run_root"
# 训练依赖已准备的数据清单，缺失时给出对应路径。
if [[ ! -f "$data_dir/manifest.json" ]]; then
  echo "Missing prepared data: $data_dir/manifest.json" >&2
  exit 1
fi

# 为单个阶段组装训练参数，优先从最近检查点恢复。
train_stage() {
  local config="$1" destination="$2" initial="${3:-}"
  local train_args=(-m minigpt.train --config "$config" --data "$data_dir" --output "$destination" --device cuda)
  # 已有检查点用于续训，首次微调则加载预训练权重。
  if [[ -f "$destination/last.pt" ]]; then
    train_args+=(--resume "$destination/last.pt")
  elif [[ -n "$initial" ]]; then
    train_args+=(--init "$initial")
  fi
  # 试跑只改变本次停止步数，学习率仍按完整训练计划计算。
  if [[ "$mode" == "pilot" ]]; then
    train_args+=(--stop-after 30)
  fi
  # 将训练输出同时显示到终端并追加到流水线日志。
  python "${train_args[@]}" 2>&1 | tee -a "$run_root/console.log"
}

# 按命令行模式选择对应阶段，训练与评测可以分开运行。
case "$mode" in
  # 短时预训练用于测量速度和显存。
  pilot)
    train_stage configs/pretrain.json "$run_root/pretrain"
    echo "Pilot checkpoint saved. Review speed and memory before the full run."
    ;;
  # 正式流程依次运行预训练、任务微调和随机初始化对照。
  full)
    train_stage configs/pretrain.json "$run_root/pretrain"
    train_stage configs/sft.json "$run_root/sft" "$run_root/pretrain/best.pt"
    train_stage configs/scratch_sft.json "$run_root/scratch"
    ;;
  # 各组使用相同的数据分区、样本数和抽取方法进行比较。
  val-eval|test-eval)
    split="val"; limit=200
    if [[ "$mode" == "test-eval" ]]; then split="test"; limit=500; fi
    python -m minigpt.evaluate --data "$data_dir" --split "$split" --limit "$limit" --output "$run_root/eval_${split}_lead"
    # 各组权重分别保存预测文件，便于按文档核对结果。
    for name in pretrain sft scratch; do
      python -m minigpt.evaluate --data "$data_dir" --split "$split" --limit "$limit" \
        --checkpoint "$run_root/$name/best.pt" --output "$run_root/eval_${split}_$name" --device cuda
    done
    ;;
  # 导出前先收集训练前后续写，再核对数据与检查点来源。
  export)
    if [[ ! -f "$run_root/continuations.json" ]]; then
      python -m scripts.collect_continuations --runs "$run_root" --data "$data_dir"
    fi
    python -m scripts.export_report --runs "$run_root" --data "$data_dir" --output report/generated
    ;;
  *) echo "Usage: bash scripts/run_cloud.sh {pilot|full|val-eval|test-eval|export}" >&2; exit 2 ;;
esac
