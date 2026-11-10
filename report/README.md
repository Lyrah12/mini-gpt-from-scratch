# 作业三报告源文件

将本文件夹的内容按原层级上传 Overleaf，主文件选择 `main.tex`，编译器选择 **XeLaTeX**。全文使用 UTF-8 和 Fandol 中文字体。

报告沿用作业一、作业二的公式说明与实验分析写法，五个主章节对应作业要求的五项内容。

1. 重要知识点梳理与学习
2. 小参数量 GPT 模型搭建
3. 模型预训练
4. 模型微调与任务效果
5. 成果总结与体会

第一章从经典 Transformer 开始，依次说明数据与分词、因果注意力、GPT 块、训练目标、优化与生成评价。后续章节集中说明实际实现、实验过程与结果，输入及解码改进纳入第四章。标题后直接进入正文，复现步骤放在源码仓库根目录的 README。

当前主文件读取 `metadata.tex`，以及 `generated/style_pretrain.tex`、`style_sft.tex`、`style_refinement.tex`、`style_examples.tex`。实验图像为 `pretrain_curve.png`、`sft_comparison.png` 和 `refinement_curve.png`。

新增的 `transformer_overview.png`、`causal_shift.png`、`gpt_preln.png` 为原创结构与机制示意图，分别说明编码器与解码器、因果遮罩与标签移位、本文 GPT 与 Pre-LN 块。主文件直接读取 PNG，保留的同名 SVG 为文字转路径版本，可用于编辑和缩放。

`fonts/CourseworkGlyph-Regular.ttf` 提供模型输出中“牠”的字形，由 Noto Serif SC 制作单字子集，许可和来源见 `fonts/OFL.txt`。更新 Overleaf 时一并上传 `fonts/` 文件夹。

姓名与学号使用既有占位，日期由编译当天生成。提交前在 Overleaf 核对身份信息、图表及分页，下载编译得到的单份 PDF。

正文引用采用带方括号的上标，必要资料列在文末参考文献中。图题与表题简短居中，图片说明放在相关正文中。

图表使用原位排版，按正文顺序出现。页面底部自然留白，避免浮动图表上下分散造成大段空白。小节标题设置最小剩余空间，减少标题与内容分页分离。最终分页在 Overleaf 编译后核对。

成果总结补充了手写模型、构造训练标签、梯度累积和断点恢复的学习体会，并列出项目 GitHub 仓库地址。
