# 资料与使用方式

资料核对日期：2026-11-01。以下列出本项目直接参考的知识和资源，不代表已经复现对应项目的性能或训练成本。

| 来源 | 网址 | 本项目使用方式 |
|---|---|---|
| news2016zh 原始发布 | https://github.com/brightmart/nlp_chinese_corpus | 正式数据候选，使用新闻正文与标题；按本文规则重建划分 |
| 作者 Google Drive 下载 | https://drive.google.com/file/d/1TMKu1FpTr6kcjWXWlQHX7YJsMfhhcVKp/view | 下载脚本的原始入口；页面与文件 ID 已核对，完整大文件尚未下载验证 |
| 作者百度网盘下载 | https://pan.baidu.com/s/1MLLM-CdM6BhJkj8D0u3atA | 云端无法使用 Google Drive 时的手动备用路线；提取码 k265 |
| 数据集引用记录 | https://doi.org/10.5281/zenodo.3402023 | 原项目建议的引用方式；Zenodo 的小型 ZIP 是代码发布快照，不是 9GB 新闻数据 |
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | 注意力、缩放与多头机制的理论来源 |
| LLMs from Scratch | https://github.com/rasbt/LLMs-from-scratch | 作业推荐的知识学习参考；本项目没有整体复制该仓库代码 |
| MiniMind | https://github.com/jingyaogong/minimind | 了解单卡小型中文模型训练路线；未加载其模型权重或复用其训练代码，不承诺相同成本和效果 |
| PyTorch 版本安装 | https://pytorch.org/get-started/previous-versions/ | 固定 2.6.0 的 CPU/CUDA 12.4 安装方式 |
| PyTorch SDPA | https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html | 使用因果注意力加速原语，并与手写矩阵版本做数值及梯度比较 |
| Hugging Face Tokenizers | https://huggingface.co/docs/tokenizers/ | 字节级 BPE 的训练、序列化和解码库；不使用现成分词器权重 |
| AutoDL 官方文档 | https://www.autodl.com/docs/ | 实例、JupyterLab、数据盘和上传下载的操作指南 |

## 开源边界

本项目新增源码采用 MIT 许可，见根目录 LICENSE。该许可只覆盖本项目新增源码，不把第三方语料、软件依赖或模型训练产物自动改为 MIT。原语料仓库标注 MIT，新闻文本仍来自不同媒体，本项目不重新打包发布原始新闻或处理后的语料，提供原作者链接、处理步骤和数据文件 SHA256 即可。

源码仓库中排除 AGENTS.md、作业 PDF、虚拟环境、数据、检查点及上传压缩包。报告可保留必要的短节选用于分析，避免把整份数据混入仓库。


## 2026-11-02 正式本地数据更新

正式实验采用 XL-Sum 官方简体中文新闻：https://huggingface.co/datasets/csebuetnlp/xlsum ，项目：https://github.com/csebuetnlp/xl-sum 。固定版本 30fece425f9a3866e04321773ca7a80056d55ca6，许可 CC-BY-NC-SA-4.0。取 text 为正文、title 为生成目标，保留官方划分。news2016zh 因原下载连接失败，仅保留为前期候选与备用获取脚本，不列作正式训练语料。

## 2026-11-04 补充实验的输入定义

在原固定版本 XL-Sum 上回查原始 summary 字段，将其作为导语放在原 text 正文前，生成目标仍为 title。没有从参考标题构造输入；原数据中的导语可能自然包含标题用语。保留原已清洗样本的 ID、排序、官方分区归属和原训练集 BPE 词表，数据许可仍为 CC-BY-NC-SA-4.0。初始 text-only 实验保留；新增输入增加了信息，不能把跨输入的指标差异全部视为模型算法收益。处理清单另存 data/processed_xlsum_lead/manifest.json，原始数据包校验方式不变。
