# 本地验证记录

日期：2026-11-01。验证在本作业目录的独立 Python 3.12 虚拟环境中进行，使用 **PyTorch 2.6.0+cpu**，没有使用本机 GPU，也没有执行正式中文新闻训练。机器有 NVIDIA 显卡，但 CPU 版 PyTorch 的 `cuda_available` 为 false 是该验证环境的预期状态。

## 已执行且通过的检查

1. `python -m pytest -q`：**11 项测试通过**。覆盖因果遮罩、手写/加速注意力前向与梯度、权重共享、小任务过拟合、BPE 字节往返、标题目标损失遮罩、字符级 ROUGE、数据篡改拒绝、合成结果导出拒绝、LaTeX 转义和曲线素材生成。Matplotlib 的依赖产生了弃用提示，不影响测试结果。
2. `python -m scripts.smoke --output work/smoke_verified`：完整 CPU 流程通过。扫描 802 条合成记录，识别两条重复，得到 800 条唯一记录；训练/验证/测试分别为 768/19/13 条，文档 ID 无交集。
3. 连续预训练 8 步，与训练 3 步后恢复再训练 5 步，最终模型权重、优化器状态和训练 RNG **逐项相同**。改变学习率后尝试恢复被拒绝。
4. 加载预训练权重微调 6 步，随机初始化微调 6 步，三种模型和抽取基线在相同 3 条合成测试记录上完成评测，样本选择哈希一致。另完成固定提示词的训练前后续写收集。
5. Python 源文件编译检查、`pip check` 和云端 Bash 脚本语法检查通过。
6. `python -m scripts.check_report_source`：LaTeX 环境配对、输入文件与引用键检查通过。该脚本不是 TeX 编译器，不能证明最终分页与字体渲染正确。

合成验证证据保存在 `work/smoke_verified/validation_summary.json`，控制台日志为 `work/smoke_verified_console.log`。本地环境记录见 `docs/local_validation_environment.json`，完整本地依赖见 `docs/local_validation_freeze.txt`。这些环境记录用于本地 CPU 复现，不要把 CPU 版 torch 的冻结文件用于云端 CUDA 环境。

## 尚未验证的部分

- 原作者约 3.6GB 完整新闻压缩包的下载及全量数据处理。
- AutoDL 实例、CUDA/BF16 路径的实际显存占用、吞吐与费用。
- 真实新闻预训练与微调的损失、生成质量和 ROUGE。
- Overleaf 的实际编译与视觉排版检查。按照用户要求未在本地编译 PDF。

因此当前状态是“可进入云端试跑的项目”，不是“正式实验已完成的作业”。素材导出程序会拒绝合成数据和未完成的正式训练。
