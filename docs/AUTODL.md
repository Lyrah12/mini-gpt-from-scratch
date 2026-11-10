# AutoDL 从零操作指南

本指南配合本项目代码使用。现在不需要立即租机器。**先看完试跑流程，再结合实例页面实时报价决定预算。** 尚未为用户租用实例、充值或执行正式训练。

## 1. 准备上传文件

本地项目目录为 `D:\Users\Lenovo\Desktop\具身智能基础\作业\作业3`。本次已准备 `outputs/cloud-upload.zip`，只包含源码、配置、测试和说明，不包含数据、权重、虚拟环境、作业 PDF、AGENTS.md 或本地报告。

如之后修改过源码，生成一个新名称的包：

```powershell
.\work\venv\Scripts\python.exe -m scripts.make_cloud_bundle --output outputs/cloud-upload-v2.zip
```

本地 `report/` 文件夹始终保留为可编辑、未压缩的 Overleaf 项目；云端只需运行源码，完成后把生成的 `report/generated/` 下载回来即可。

## 2. 选择实例和环境

在 AutoDL 控制台进入算力市场，选择按量计费实例，初始候选条件如下：

- 1 张 RTX 4090，显存 24GB。没有该型号时，把可选型号、显存和页面报价提供给我，再调整。
- 建议至少 16GB CPU 内存；准备原始数据、处理文件和检查点，尽量保持数据盘有 30～40GB 可用空间。不要为本作业预先购买超大数据盘。
- 优先选 PyTorch 2.6.0、Python 3.12、CUDA 12.4 的基础镜像，或兼容的 PyTorch 镜像。镜像名称可能变化，按页面实际可用选项处理。
- 记下 GPU 每小时单价和额外存储费用。当前文档不预填租用价格。

AutoDL 的项目与数据建议放在 `/root/autodl-tmp` 数据盘。无卡模式可以用于上传和下载；全量数据处理也需要内存，不能假定小内存模式一定足够。[官方存储说明](https://www.autodl.com/docs/env/)、[快速入门](https://www.autodl.com/docs/quick_start/)。

## 3. 上传并进入项目

启动实例后点击 JupyterLab。左侧文件栏进入 `autodl-tmp`，上传 `cloud-upload.zip`。打开 Launcher 中的 Terminal，执行：

```bash
cd /root/autodl-tmp
unzip -n cloud-upload.zip -d mini-gpt
cd /root/autodl-tmp/mini-gpt
pwd
ls
```

应看到 `minigpt`、`configs`、`scripts`、`docs` 等。下文所有命令在这个项目根目录执行。`unzip -n` 不覆盖已有文件；更新代码时用新目录，避免新旧版本混在一起。

## 4. 检查 GPU 与安装环境

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果镜像已包含 `torch==2.6.0` 且 CUDA 可用，不必重新安装 torch。若不匹配，可以在确认镜像和驱动兼容后按官方命令安装：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

环境选择以 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/) 为依据。无需安装 torchvision、torchaudio、完整 CUDA toolkit 或修改系统驱动。安装失败或 CUDA 显示 False 时，将报错文字发给我排查。

## 5. 下载新闻语料

先尝试作者 Google Drive 下载链接：

```bash
python -m scripts.download_news
find data/raw -maxdepth 3 -type f
```

作者提供的压缩包约 3.6GB，解压约 9GB。脚本保留压缩包并记录其 SHA256。**当前仅核对了来源和下载链接，尚未在本项目中完成整个压缩包的下载验证。** 如果出现网络、访问权限或下载配额错误，不要反复空跑 GPU，改用作者的百度网盘路线：

- [作者原始说明](https://github.com/brightmart/nlp_chinese_corpus#2新闻语料json版news2016zh)
- [百度网盘](https://pan.baidu.com/s/1MLLM-CdM6BhJkj8D0u3atA)，提取码 `k265`

在本地下载后，可通过 JupyterLab 上传 ZIP；文件较大时使用平台提供的 SSH 信息和 `scp` 上传。下面的主机和端口需替换为实例页面中的值，在**本地 PowerShell**执行：

```powershell
scp -P <端口> "<本地压缩包的完整路径>" root@<SSH主机>:/root/autodl-tmp/mini-gpt/data/raw/
```

然后在云端项目目录运行：

```bash
python -m scripts.download_news
```

如果已经手动解压并有 JSON 文件，可以直接进入下一步，不必再次运行下载脚本。

## 6. 处理数据（只需一次）

依据 `find` 的实际结果填写输入路径。常见文件名如下；若文件解压到子目录，相应增加子目录：

```bash
python -m minigpt.prepare \
  --input data/raw/news2016zh_train.json data/raw/news2016zh_valid.json \
  --output data/processed \
  --max-records 300000 --vocab-size 16000 --block-size 512 --seed 42
```

脚本扫描全量提供文件，清洗、去重、抽样、划分，再训练 BPE 和编码数据。不要把扫描到 30 万条误认为已完成；程序扫描全部文件后才得到蓄水池样本。完成后查看：

```bash
python -c "import json; m=json.load(open('data/processed/manifest.json')); print(m['counts']); print(m['token_counts']); print(m['vocab_size'])"
df -h /root/autodl-tmp
```

保留这个数据目录，后续三组训练共用。输出目录非空时程序拒绝重建；调整抽样或分词器时必须使用新目录，并确保后续训练指向它。失败留下的半成品不能作为已处理数据使用，需选择新输出目录重试。

## 7. 运行 30 步试跑

```bash
bash scripts/run_cloud.sh pilot
python -m scripts.pilot_summary --run runs/cloud/pretrain
```

程序保存 30 步的 `last.pt`，不会改变正式 3000 步的学习率计划。查看 step_seconds、显存和是否出现 NaN。按页面实际单价计算预训练计算费用示例：

```bash
# 把下面的 0 替换为实例页面实际 GPU 小时价格；0 只是命令占位，不是报价。
python -m scripts.pilot_summary --run runs/cloud/pretrain --hourly-price 0
```

估计只覆盖剩余预训练计算，**不含验证、保存、SFT、下载、评测和存储**。将这一步输出发给我，我们再据此确定完整训练预算。如果需要暂停等待，可以先关机，检查点会保留；不要在等待期间让 GPU 一直空闲计费。

如遇显存不足，先把微批量从 8 减到 4、累积次数相应翻倍，保持有效批量。由于配置变化会改变训练签名，使用新的运行目录重新试跑：复制并修改配置，再通过 `python -m minigpt.train` 显式指定新配置与输出，不能直接恢复旧签名的运行。

## 8. 正式训练与观察日志

确定预算后，建议用 tmux 保持终端会话：

```bash
tmux new -s minigpt
cd /root/autodl-tmp/mini-gpt
bash scripts/run_cloud.sh full
```

如果没有 tmux，可在平台终端安装，或在 JupyterLab Terminal 中直接运行并保留该 Terminal 会话。[AutoDL Linux 操作说明](https://www.autodl.com/docs/linux/)。

`full` 依次运行：预训练（从 30 步恢复到 3000 步）、加载预训练最优权重的 SFT、随机初始化的 SFT 对照。三个阶段可以自动继续已有 `last.pt`。

tmux 中按 `Ctrl+B` 后松开，再按 `D` 可以离开会话；重新进入用 `tmux attach -t minigpt`。另开终端查看：

```bash
tail -n 8 runs/cloud/pretrain/metrics.jsonl
tail -n 8 runs/cloud/sft/metrics.jsonl
tail -f runs/cloud/console.log
```

`tail` 提示 SFT 文件尚不存在，可能只是仍在预训练，先看 console.log。训练日志每 10 步保存一次，每 100 步验证并保存 `last.pt` 与验证损失最低的 `best.pt`。

## 9. 暂停与断点续训

正常暂停：在训练终端按一次 `Ctrl+C`，等待当前优化器步完成并出现保存结束/进程退出，再关机。不要反复按 Ctrl+C。

重新开机后进入项目目录，再执行 `bash scripts/run_cloud.sh full`，脚本自动恢复。突发关机时恢复到上次成功保存的 checkpoint，可能损失最近若干步。

也可以单独恢复某阶段：

```bash
python -m minigpt.train --config configs/sft.json --data data/processed \
  --output runs/cloud/sft --resume runs/cloud/sft/last.pt --device cuda
```

不要把 `--init` 当作续训：`--init` 只加载预训练权重并开启新的 SFT；`--resume` 才恢复优化器、学习率进度与随机状态。恢复时配置和处理数据必须不变。

## 10. 验证、最终评测与报告素材

```bash
bash scripts/run_cloud.sh val-eval
```

先通过验证集检查生成结果和训练是否充分。若需要调整训练方案，使用新配置、新运行目录，并记录修改；不要根据测试集挑参数。

冻结方案后：

```bash
bash scripts/run_cloud.sh test-eval
bash scripts/run_cloud.sh export
```

测试评测为四组方法使用同一固定 500 条测试子集，贪心生成标题。导出程序检查数据指纹、训练完成状态、四组样本一致性及微调初始化来源，然后生成 `report/generated/results.tex`、`loss_curves.png` 和 `experiment_summary.json`。合成验证数据或未完成训练不能导出为正式结果。

评测输出目录不覆盖。如果重复执行同一评测会报错，先保留现有结果，确认是无需重跑还是需要新实验目录。模型很小，不保证超越首句抽取基线；结论按实际结果写。

## 11. 下载并完成报告

至少保留：

- `report/generated/`：拷回本地 `report/generated/`，替换等待实验的占位文件。
- `runs/cloud/`：三组 run/status/metrics、评测逐条预测、续写和检查点。
- `data/processed/manifest.json` 与 `tokenizer.json`：复现证据与推理必需文件。

文件夹下载可以在本地 PowerShell 使用 SCP：

```powershell
scp -r -P <端口> root@<SSH主机>:/root/autodl-tmp/mini-gpt/report/generated "D:\Users\Lenovo\Desktop\具身智能基础\作业\作业3\report\"
scp -r -P <端口> root@<SSH主机>:/root/autodl-tmp/mini-gpt/runs/cloud "D:\Users\Lenovo\Desktop\具身智能基础\作业\作业3\runs\"
```

先确认目标 `runs/` 存在。下载完成后核对云端生成文件已进入本地正确位置；不要多嵌套一层 `generated/generated`。检查点较大，可先只下载日志与最佳权重；如保留续训能力，需要同时保留 last.pt 和处理数据。

我会根据真实结果补充分析，你再填写姓名、学号以及实际个人体会。最终将本地完整、未压缩的 `report/` 文件夹上传 Overleaf，主文件 `main.tex`，编译器 XeLaTeX。报告主文件包含 `generated/` 的相对路径，不需要改成本地绝对路径。

## 12. 结束使用

确认结果已下载后，在 AutoDL 控制台关机，核对实例状态。不要只关闭浏览器。后续长期保留和释放规则、付费数据盘费用以 [AutoDL 官方说明](https://www.autodl.com/docs/instance_data/) 和 [计费说明](https://www.autodl.com/docs/price/) 为准；不要将未备份的唯一检查点留在实例中。
