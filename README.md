# 从零训练中文 Mini GPT

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

下表使用同一固定的 500 条测试新闻。ROUGE 为字符级 F1 的宏平均乘以 100，保留 Unicode 字母和数字，英文转为小写。

| 模型与新闻输入 | 解码 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| 首句抽取，正文 | 前32字符 | 26.28 | 14.98 | 22.92 |
| 仅预训练，正文 | 贪心 | 2.89 | 1.25 | 2.77 |
| 随机初始化微调，正文 | 贪心 | 9.56 | 3.71 | 9.43 |
| 预训练后微调，正文 | 贪心 | 30.81 | 17.99 | 28.77 |
| 原微调权重，正文 | 重复惩罚1.1 | 31.72 | 18.64 | 29.37 |
| 原微调权重，导语加正文 | 重复惩罚1.1 | 34.47 | 20.70 | 31.80 |
| 重新微调，导语加正文 | 重复惩罚1.1 | **38.77** | **24.44** | **35.86** |
| 首句抽取，导语加正文 | 前32字符 | 33.34 | 20.24 | 29.41 |

最终方案增加了原始导语信息，并重新执行同预算微调。相同输入和解码条件下，重新微调相对原权重提高 4.06 个 ROUGE-L 分值点。实验只使用一个种子，补充实验沿用了此前已查看的测试子集。输出仍有数字和事件关系错误，具体样例见报告与[补充实验记录](docs/REFINEMENT.md)。

## 目录

```text
minigpt/       模型、分词、数据处理、训练、评测和推理
configs/       预训练、微调及随机初始化对照的配置
scripts/       数据获取、实验流水线、报告素材生成与检查
tests/         因果遮罩、梯度、损失、解码和数据相关测试
docs/          环境记录、实验说明和来源
report/        单份作业报告的 Overleaf 源文件
data/          本地数据与分词器
runs/          本地检查点、训练日志和逐条预测
```

## 复现环境

实际环境为 Windows 11、Python 3.12.14、PyTorch 2.6.0+cu124、NumPy 2.2.6、Tokenizers 0.21.0 和 Matplotlib 3.10.3。以下命令在仓库根目录的 PowerShell 中执行，使用 Python 3.12。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dev.txt
```

确认 PyTorch 能访问 GPU。

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

完整依赖记录见[CUDA 环境](docs/local_cuda_environment.json)和[依赖快照](docs/local_cuda_freeze.txt)。CPU 数值验证可在独立虚拟环境安装 CPU 版 PyTorch。

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

代码在关键的张量变换、因果遮罩、标签移位、损失归一化和随机状态保存处提供中文注释。验证包括单元测试、合成数据上的完整训练流程，以及中断恢复后模型与优化器状态的一致性检查。

## 数据准备

正式数据来自 XL-Sum 简体中文 BBC 新闻。数据集与项目说明分别见下列链接。

- [数据集](https://huggingface.co/datasets/csebuetnlp/xlsum)
- [XL-Sum 项目](https://github.com/csebuetnlp/xl-sum)

固定版本为 `30fece425f9a3866e04321773ca7a80056d55ca6`。原始压缩包 SHA-256 为 `eaa26651ca379a6f689cecd459eede68c3a67b0de64ce9e9bea62889c252fb2e`。

```powershell
python -X utf8 -m scripts.download_xlsum
```

脚本按固定版本下载、校验并处理数据。初始实验取 `text` 为正文，`title` 为目标，保留官方分区。过滤与跨分区去重后，训练、验证、测试分别为 **36762、4585、4662** 篇，训练 token 流长度为 **26312816**。词表只在训练分区学习。

处理结果写入 `data/processed_xlsum/`，其中 `manifest.json` 保存来源、参数和文件校验值。该命令用于首次准备数据，已有非空处理目录时程序会保护已有产物。需要另一份处理结果时传入新的 `--output` 目录。

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
4. 在固定验证和测试子集上评估模型与抽取基线。
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

## 导语补全与解码比较

完成初始实验后执行补充流水线。

```powershell
python -X utf8 -u -m scripts.run_refinement
```

该流程从同一原始压缩包恢复 `summary`，作为导语放在 `text` 正文之前。原文档 ID、排序、划分和分词器沿用初始实验，数据另存 `data/processed_xlsum_lead/`。

新增微调从原预训练的 `best.pt` 开始，使用相同的 1500 步微调配置。三组输入与权重组合分别比较贪心、重复惩罚 1.1、禁止重复 token 三元组。根据固定 200 条验证新闻选择解码后，再评估所选设置的 500 条测试新闻。

补充记录写入 `runs/refinement_20261104/`。

- `policy.json` 记录候选设置和选择规则。
- `selection.json` 记录测试前冻结的选择。
- `summary.json` 汇总实际训练和评测结果。
- `val_*/` 保存各候选的验证预测。
- `test_*/` 保存所选设置的测试预测。
- `sft/best.pt` 保存导语加正文条件下的最佳微调权重。

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

本机四个训练阶段的迭代时间累计约 **62 分钟**。环境准备、数据处理、验证、存盘和生成评测还会增加实际墙钟用时。

## 本地推理

预训练模型的中文续写。

```powershell
python -X utf8 -m minigpt.generate --checkpoint runs/local_4060/pretrain/best.pt --tokenizer data/processed_xlsum/tokenizer.json --text "人工智能技术的发展" --temperature 0.8 --max-new-tokens 64
```

最终标题模型接收真实新闻导语与正文。

```powershell
python -X utf8 -m scripts.title_decoding --checkpoint runs/refinement_20261104/sft/best.pt --tokenizer data/processed_xlsum/tokenizer.json --mode penalty_1p1 --lead "新闻原始导语" --text "新闻正文"
```

`--lead` 可省略。标题输出最多生成 64 个 token，输入按共享函数截取，为标题预留位置。

## 报告

报告的五个主章节依次为知识点梳理与学习、GPT 模型搭建、模型预训练、模型微调与任务效果、成果总结与体会，与作业要求一一对应。第一章从经典 Transformer 过渡至 GPT，输入和解码改进纳入微调章节。主文件为 `report/main.tex`，实验表格、曲线和样例来自保存的运行产物。

```powershell
python -X utf8 -m scripts.build_report_assets
python -X utf8 -m scripts.check_report_source
```

三张原理图由 `scripts/generate_concept_figures.py` 绘制，PNG 可直接用于报告，SVG 已将文字转为路径。现有图像随报告提供。需要重绘时，指定本机的静态中文字体文件。

```powershell
python -X utf8 -m scripts.generate_concept_figures --font "中文字体文件的完整路径.ttf" --output report/generated
```

将整个 `report/` 文件夹上传 Overleaf，设置 **XeLaTeX**，编译 `main.tex` 得到单份报告。源码检查涵盖环境配对、文件依赖和引用键，最终页面在 Overleaf 检查。

正文引用使用带方括号的上标，图表按正文顺序排版，图题说明写在相关段落中。报告总结部分提供仓库链接。

## 资料与许可

架构与训练流程学习参考 [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch)，单卡工程组织参考 [MiniMind](https://github.com/jingyaogong/minimind)。模型使用 [PyTorch](https://docs.pytorch.org/docs/stable/) 基础层、自动求导、AdamW 和 SDPA，字节级 BPE 使用 [Tokenizers](https://huggingface.co/docs/tokenizers/)。来源与具体使用方式见[资料记录](docs/SOURCES.md)。

项目源码采用 MIT 许可，见 [LICENSE](LICENSE)。XL-Sum 数据采用 CC-BY-NC-SA-4.0。数据、分词器、检查点、虚拟环境和运行日志通过 `.gitignore` 排除，复现时按上述步骤生成本地产物。
