# 从零构建大模型

基于 PyTorch 基础组件实现中文 GPT 的分词、模型搭建、预训练、标题生成微调和推理。模型从随机权重开始训练，包含 **49,957,120** 个独立参数。实验在单张 RTX 4060 Laptop 8GB 上完成。

## 模型和实际结果

| 配置 | 设置 |
|---|---|
| Transformer 层数 | 8 |
| 注意力头数 | 10 |
| 隐藏维度 / 前馈维度 | 640 / 2560 |
| 上下文长度 | 512 token |
| 词表 | 训练集学习的 16000 大小字节级 BPE |
| 位置编码 | 学习式绝对位置嵌入 |
| 权重共享 | 输入嵌入与输出投影共享 |
| 训练精度 | BF16 |
| 随机种子 | 42 |

下表使用同一批 500 条测试新闻。ROUGE 为字符级 F1 的宏平均乘以 100，保留 Unicode 字母和数字，英文转为小写。首句抽取与模型读取相同的新闻前缀，并保留首句前 32 个字符。

| 模型与新闻输入 | 解码 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| 首句抽取，正文 | 前32字符 | 26.28 | 14.98 | 22.92 |
| 仅预训练，正文 | 贪心 | 2.89 | 1.25 | 2.77 |
| 随机初始化微调，正文 | 贪心 | 9.56 | 3.71 | 9.43 |
| 预训练后微调，正文 | 贪心 | 30.81 | 17.99 | 28.77 |
| 原微调权重，正文 | 重复惩罚1.1 | 31.72 | 18.64 | 29.37 |
| 原微调权重，摘要加正文 | 重复惩罚1.1 | 34.47 | 20.70 | 31.80 |
| 重新微调，摘要加正文 | 重复惩罚1.1 | **38.77** | **24.44** | **35.86** |
| 首句抽取，摘要加正文 | 前32字符 | 33.34 | 20.24 | 29.41 |

最终方案加入数据集提供的人工摘要，并重新微调 1500 步。相同输入和解码条件下，重新微调相对原权重提高 4.06 分。实验使用一个随机种子，补充实验沿用了此前已查看的测试新闻。输出仍有数字和事件关系错误，例如将新闻中的12名嫌疑人写成7名。

## 目录

```text
minigpt/       模型、分词、数据处理、训练、评测和推理
configs/       预训练、微调及随机初始化对照的配置
scripts/       数据下载、训练流水线、评测与辅助工具
tests/         因果遮罩、梯度、损失、解码和数据相关测试
```

运行后，数据写入 `data/`，检查点、日志和预测结果写入 `runs/`，这些本地产物由 `.gitignore` 排除。

## 复现环境

实际环境为 Windows 11、Python 3.12.14、PyTorch 2.6.0+cu124、NumPy 2.2.6、Tokenizers 0.21.0 和 Matplotlib 3.10.3。以下命令使用 Python 3.12，在 PowerShell 中执行。

```powershell
git clone https://github.com/Lyrah12/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dev.txt
```

确认 PyTorch 能访问 GPU。

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

CPU 数值验证可在独立虚拟环境安装 CPU 版 PyTorch。

```powershell
py -3.12 -m venv .venv_cpu
.\.venv_cpu\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m scripts.smoke --output work/smoke_repro
deactivate
.\.venv\Scripts\Activate.ps1
```

`scripts.smoke` 使用明确标记的合成数据验证训练、恢复和评测流程。重复执行时为 `--output` 选择新的目录。

测试覆盖因果遮罩、注意力梯度、标签移位、损失计算、生成约束和中断恢复。

## 数据准备

正式数据来自 XL-Sum 简体中文 BBC 新闻。数据集与项目说明分别见下列链接。

- [数据集](https://huggingface.co/datasets/csebuetnlp/xlsum)
- [XL-Sum 项目](https://github.com/csebuetnlp/xl-sum)

固定版本为 `30fece425f9a3866e04321773ca7a80056d55ca6`。原始压缩包 SHA-256 为 `eaa26651ca379a6f689cecd459eede68c3a67b0de64ce9e9bea62889c252fb2e`。

```powershell
python -X utf8 -m scripts.download_xlsum
```

脚本按固定版本下载、校验并处理数据。初始实验取 `text` 为正文，`title` 为目标，保留官方分区。过滤与跨分区去重后，训练、验证、测试分别为 **36762、4585、4662** 篇，训练 token 流长度为 **26312816**。词表只在训练分区学习。

处理结果写入 `data/processed_xlsum/`，`manifest.json` 保存来源、参数和文件校验值。准备另一份数据时，用 `--output` 指定新目录。

## 初始预训练与微调

先执行 30 步预训练试跑，再按完整计划继续。

```powershell
python -X utf8 -u -m scripts.run_local --pilot
python -X utf8 -u -m scripts.run_local
```

流水线执行以下阶段。

1. 从随机权重预训练 3000 步。
2. 从最佳预训练权重微调标题任务 1500 步。
3. 随机初始化直接微调 1500 步，作为初始化对照。
4. 在固定验证和测试新闻上比较模型与首句抽取方法。
5. 收集固定提示下的预训练前后续写，导出原始结果素材。

| 阶段 | 配置文件 | 微批量 × 累积 | 最大学习率 | 预热步数 |
|---|---|---:|---:|---:|
| 预训练 | `configs/local_pretrain.json` | 4 × 16 | 0.0003 | 100 |
| 标题微调 | `configs/local_sft.json` | 4 × 8 | 0.00005 | 50 |
| 随机初始化微调 | `configs/local_scratch_sft.json` | 4 × 8 | 0.00005 | 50 |

状态写入 `runs/local_4060/pipeline.json`，各阶段的 `run.json`、`metrics.jsonl` 和 `status.json` 分别保存实际配置、逐步指标与完成状态，`best.pt` 和 `last.pt` 保存最优及最近检查点。

```powershell
Get-Content runs/local_4060/pipeline.json
Get-Content runs/local_4060/logs/pretrain.log -Tail 5
```

## 摘要输入与解码比较

完成初始实验后执行补充流水线。

```powershell
python -X utf8 -u -m scripts.run_refinement
```

该流程从同一原始压缩包读取 `summary` 人工摘要，放在 `text` 正文之前。原文档 ID、排序、划分和分词器沿用初始实验，数据另存 `data/processed_xlsum_lead/`。

新增微调从原预训练的 `best.pt` 开始，使用相同的 1500 步微调配置。三组输入与权重组合分别比较贪心、重复惩罚 1.1、禁止重复 token 三元组。根据固定 200 条验证新闻选择解码后，再评估所选设置的 500 条测试新闻。

补充记录写入 `runs/refinement_20261104/`。

- `policy.json` 记录候选设置和选择规则。
- `selection.json` 记录测试前冻结的选择。
- `summary.json` 汇总实际训练和评测结果。
- `val_*/` 保存各候选的验证预测。
- `test_*/` 保存所选设置的测试预测。
- `sft/best.pt` 保存摘要加正文条件下的最佳微调权重。

## 暂停与恢复

在训练中创建对应流水线的暂停请求文件，程序会完成当前优化器步并保存。

```powershell
New-Item -ItemType File -Path runs/local_4060/pause.request
```

等待 `pipeline.json` 显示 `paused` 后，可在下一次运行前移除请求文件，再执行同一流水线。

```powershell
Remove-Item -LiteralPath runs/local_4060/pause.request
python -X utf8 -u -m scripts.run_local
```

补充实验使用 `runs/refinement_20261104/pause.request`，恢复命令为 `python -X utf8 -u -m scripts.run_refinement`。恢复时需匹配训练配置、数据、分词器和核心源码指纹。

本机四个训练阶段累计用时约 **62 分钟**，验证和存盘另计。

## 单独评测

以下命令以贪心解码评测初始微调模型，将 500 条测试新闻的预测和分数写入指定目录。

```powershell
python -X utf8 -m minigpt.evaluate --data data/processed_xlsum --checkpoint runs/local_4060/sft/best.pt --split test --limit 500 --output runs/local_4060/eval_test_sft_manual
```

再次评测时更换 `--output`，即可保存新一轮结果。

## 本地推理

预训练模型的中文续写。

```powershell
python -X utf8 -m minigpt.generate --checkpoint runs/local_4060/pretrain/best.pt --tokenizer data/processed_xlsum/tokenizer.json --text "人工智能技术的发展" --temperature 0.8 --max-new-tokens 64
```

最终标题模型接收新闻摘要与正文。

```powershell
python -X utf8 -m scripts.title_decoding --checkpoint runs/refinement_20261104/sft/best.pt --tokenizer data/processed_xlsum/tokenizer.json --mode penalty_1p1 --lead "数据集提供的新闻摘要" --text "新闻正文"
```

`--lead` 用于传入摘要，可省略。标题最多生成 64 个 token，遇到 EOS 结束。

## 资料与许可

架构与训练流程学习参考 [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)，单卡工程组织参考 [MiniMind](https://github.com/jingyaogong/minimind)。模型使用 [PyTorch](https://docs.pytorch.org/docs/stable/) 基础层、自动求导、AdamW 和 SDPA，字节级 BPE 使用 [Tokenizers](https://huggingface.co/docs/tokenizers/)。

项目源码采用 MIT 许可，见 [LICENSE](LICENSE)。XL-Sum 数据采用 CC-BY-NC-SA-4.0。数据、分词器、检查点、虚拟环境和运行日志通过 `.gitignore` 排除，复现时按上述步骤生成本地产物。
