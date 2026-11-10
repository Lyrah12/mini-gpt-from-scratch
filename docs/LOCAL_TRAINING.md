# RTX 4060 本地训练

2026-11-02 用户确认改为本地执行。项目目录为 `D:\Users\Lenovo\Desktop\具身智能基础\作业\作业3`。

## 数据与环境

正式数据改为 XL-Sum 官方简体中文 BBC 新闻，原 news2016zh 的 Google Drive 连接失败，因此未用于正式实验。使用 title 字段作为标题目标，不使用 summary 字段；保留官方 train/val/test 划分，再跨划分去重和过滤。

固定数据版本：`30fece425f9a3866e04321773ca7a80056d55ca6`。压缩包 SHA256：`eaa26651ca379a6f689cecd459eede68c3a67b0de64ce9e9bea62889c252fb2e`。许可为 CC-BY-NC-SA-4.0；源码许可不改变语料许可，数据不上传 GitHub。

清洗后：训练 36762 篇、验证 4585 篇、测试 4662 篇；训练 token 流含 26312816 个 token。16000 词表仅在训练部分学习。具体统计与哈希保存在 `data/processed_xlsum/manifest.json`。

本机为 RTX 4060 Laptop 8GB、约 16GB 内存。原 CPU 环境为 `work/venv`；独立 CUDA 环境为 `work/venv_cuda`。本地配置在 `configs/local_*.json`；有效预训练批量为 64 个 512-token 窗口，微调有效批量为 32 条。微批量及速度以 GPU 测试和实际 run.json 为准。

## 一条命令启动或恢复

在项目根目录的 PowerShell 执行：

```powershell
.\scripts\start_local.ps1
```

脚本后台运行，不弹出额外终端；已有活动进程时拒绝重复启动。也可以用前台方式：

```powershell
.\work\venv_cuda\Scripts\python.exe -X utf8 -u -m scripts.run_local
```

第一次硬件确认用 `scripts/start_local.ps1 -Pilot`，只训练预训练的前 30 步，保持完整学习率计划；之后正常启动会从 last.pt 继续。

流水线顺序：预训练 3000 步 → 加载预训练最佳检查点微调 1500 步 → 随机初始化微调对照 1500 步 → 固定验证/测试子集评测 → 前后续写收集 → 报告素材导出。只有真正完成的阶段才标记完成。

## 查看进度

```powershell
.\scripts\status_local.ps1
Get-Content runs/local_4060/logs/pretrain.log -Tail 5
```

阶段状态为 `runs/local_4060/pipeline.json`，详细指标在各阶段的 metrics.jsonl，检查点在 best.pt/last.pt，子进程错误在各阶段 logs 中。完成后状态为 `experiments_complete_report_assets_exported`；若失败会记录 failed 和错误信息，不自动跳过错误。

## 暂停与恢复

```powershell
.\scripts\pause_local.ps1
```

等待状态变为 paused 后再关机。训练程序会完成当前优化器步并保存完整检查点，流水线不会继续下一训练阶段。恢复时执行 start_local.ps1，它会清除本项目的暂停请求并自动恢复。不要用任务管理器强制结束作为正常暂停方式。

训练进程运行期间临时阻止系统因空闲自动睡眠，结束后释放该请求；不修改系统电源计划。请插电、保持散热，并保持电脑开机；合盖或手动关机仍可能中断训练。

## 结果与报告

实际结果仍须等待流水线完成。最后自动写入本地未压缩的 report/generated/，包括损失图、实际样例、ROUGE 表和原始汇总 JSON。不会在本地编译 PDF；还需要根据实际结果核对报告分析，并填写姓名、学号和个人体会，再上传 Overleaf 使用 XeLaTeX 编译。

## 从头复现数据

```powershell
.\work\venv_cuda\Scripts\python.exe -X utf8 -m scripts.download_xlsum
```

程序使用 HTTPS 分段下载和作者公开的 SHA256 校验，固定版本；已处理数据目录非空时拒绝覆盖。需要重新构建时传入新的 --output 目录，并让后续训练使用对应数据目录。纯 CPU 单元测试可用独立的新 --basetemp 目录，避免旧临时目录权限冲突。

数据来源：https://huggingface.co/datasets/csebuetnlp/xlsum 。项目与论文：https://github.com/csebuetnlp/xl-sum 。
