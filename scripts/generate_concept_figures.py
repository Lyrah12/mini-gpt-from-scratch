"""Draw the report's original architecture diagrams as PNG and outlined SVG."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

BLUE = '#216185'
PALE_BLUE = '#eaf3f8'
ORANGE = '#a95c27'
PALE_ORANGE = '#fbf1e7'
INK = '#20313b'
GREY = '#65737c'


def label(ax, x, y, text, font, size=10, color=INK, ha='center'):
    return ax.text(x, y, text, fontproperties=font, fontsize=size, color=color,
                   ha=ha, va='center', linespacing=1.3, zorder=5)


def box(ax, x, y, width, height, text, font, fill=PALE_BLUE, edge=BLUE, size=10):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
        boxstyle='round,pad=0.02,rounding_size=0.08', linewidth=1.1,
        facecolor=fill, edgecolor=edge, zorder=2))
    label(ax, x + width/2, y + height/2, text, font, size=size)


def arrow(ax, start, end, color=BLUE, connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
        linewidth=1.2, color=color, connectionstyle=connection, zorder=3))


def line_path(ax, points, color=ORANGE):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
            linewidth=1.15, zorder=3)
    arrow(ax, points[-2], points[-1], color=color)


def canvas(size, xlim, ylim):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=xlim, ylim=ylim)
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig, ax


def save(fig, root, name):
    fig.savefig(root / (name + '.png'), dpi=240, facecolor='white')
    fig.savefig(root / (name + '.svg'), facecolor='white')
    plt.close(fig)


def transformer(root, font):
    fig, ax = canvas((7.1, 4.8), (0, 10), (0, 7.1))
    label(ax, 2.4, 6.83, '编码器', font, size=13, color=BLUE)
    label(ax, 7.5, 6.83, '解码器', font, size=13, color=ORANGE)
    box(ax, 0.65, 5.95, 3.5, 0.53, '完整输入序列', font)
    box(ax, 5.75, 5.95, 3.5, 0.53, '已知目标前缀', font, fill=PALE_ORANGE, edge=ORANGE)
    box(ax, 0.65, 5.04, 3.5, 0.57, '词嵌入 + 位置表示', font)
    box(ax, 5.75, 5.04, 3.5, 0.57, '词嵌入 + 位置表示', font, fill=PALE_ORANGE, edge=ORANGE)
    for x in (2.4, 7.5):
        arrow(ax, (x, 5.92), (x, 5.64), color=BLUE if x == 2.4 else ORANGE)
        arrow(ax, (x, 5.01), (x, 4.17 if x == 2.4 else 4.29), color=BLUE if x == 2.4 else ORANGE)
    ax.add_patch(Rectangle((0.36, 1.7), 4.08, 3.02, fill=False, edgecolor=BLUE,
                           linewidth=1.0, linestyle='--'))
    ax.add_patch(Rectangle((5.46, 1.7), 4.08, 3.02, fill=False, edgecolor=ORANGE,
                           linewidth=1.0, linestyle='--'))
    label(ax, 0.56, 4.53, '编码层 × N', font, size=8.5, ha='left', color=GREY)
    label(ax, 5.66, 4.53, '解码层 × N', font, size=8.5, ha='left', color=GREY)
    box(ax, 0.8, 3.46, 3.2, 0.68, '自注意力\n读取完整输入', font, size=10)
    box(ax, 0.8, 2.24, 3.2, 0.68, '前馈网络', font)
    arrow(ax, (2.4, 3.43), (2.4, 2.95))
    box(ax, 5.9, 3.61, 3.2, 0.65, '因果自注意力\n读取已有目标', font, fill=PALE_ORANGE, edge=ORANGE)
    box(ax, 5.9, 2.67, 3.2, 0.65, '交叉注意力', font, fill=PALE_ORANGE, edge=ORANGE)
    box(ax, 5.9, 1.95, 3.2, 0.44, '前馈网络', font, fill=PALE_ORANGE, edge=ORANGE)
    arrow(ax, (7.5, 3.58), (7.5, 3.35), color=ORANGE)
    arrow(ax, (7.5, 2.64), (7.5, 2.42), color=ORANGE)
    arrow(ax, (2.4, 2.21), (2.4, 1.30))
    box(ax, 0.8, 0.63, 3.2, 0.6, '编码结果', font)
    line_path(ax, [(4.02, 0.94), (4.9, 0.94), (4.9, 3.0), (5.88, 3.0)], color=BLUE)
    label(ax, 4.98, 1.42, 'K、V', font, size=8.5, ha='left', color=BLUE)
    arrow(ax, (7.5, 1.92), (7.5, 1.36), color=ORANGE)
    box(ax, 5.75, 0.79, 3.5, 0.50, '线性投影 + Softmax', font, fill=PALE_ORANGE, edge=ORANGE)
    arrow(ax, (7.5, 0.76), (7.5, 0.40), color=ORANGE)
    label(ax, 7.5, 0.22, '下一个目标 token', font, size=10)
    label(ax, 2.4, 0.2, '各子层配合残差连接与归一化', font, size=8.5, color=GREY)
    save(fig, root, 'transformer_overview')


def gpt(root, font):
    fig, ax = canvas((7.1, 5.0), (0, 10), (0, 7.8))
    label(ax, 2.05, 7.48, '本文 GPT 的整体流程', font, size=12, color=BLUE)
    label(ax, 7.05, 7.48, '一个 Pre-LN 块', font, size=12, color=BLUE)
    left = [(6.54, 'token ID'), (5.52, '词嵌入 + 位置嵌入\n输入 Dropout'),
            (4.22, 'Pre-LN Transformer 块\n重复 8 次'), (3.13, '最终 LayerNorm'),
            (2.13, '词表投影\n与输入嵌入共享权重'), (1.11, '16000 维 logits')]
    heights = [0.52, 0.72, 0.78, 0.57, 0.72, 0.52]
    for (y, text), height in zip(left, heights):
        box(ax, 0.42, y, 3.25, height, text, font)
    for i in range(len(left)-1):
        arrow(ax, (2.05, left[i][0]-0.02), (2.05, left[i+1][0]+heights[i+1]+0.03))
    arrow(ax, (2.05, 1.08), (2.05, 0.69))
    label(ax, 2.05, 0.44, '选择新 token 并追加到序列', font, size=9.5)
    ax.plot([3.75, 4.35], [4.62, 4.62], linestyle='--', color=GREY, linewidth=1)
    arrow(ax, (4.35, 4.62), (4.85, 4.62), color=GREY)
    ax.add_patch(Rectangle((4.95, 0.12), 4.70, 7.09, fill=False, edgecolor=GREY, linestyle='--', linewidth=1))
    x = 6.75
    label(ax, x, 6.94, '输入 H', font, size=10)
    box(ax, 5.45, 5.95, 2.6, 0.52, 'LayerNorm', font)
    box(ax, 5.45, 5.03, 2.6, 0.65, '因果多头自注意力\n分支 Dropout', font)
    for start, end in [((x, 6.73), (x, 6.50)), ((x, 5.92), (x, 5.71)), ((x, 5.0), (x, 4.46))]:
        arrow(ax, start, end)
    for y in (4.22, 1.12):
        ax.add_patch(Circle((x, y), 0.22, facecolor='white', edgecolor=ORANGE, linewidth=1.3, zorder=4))
        label(ax, x, y, '+', font, size=14, color=ORANGE)
    line_path(ax, [(6.75, 6.72), (8.95, 6.72), (8.95, 4.22), (7.0, 4.22)])
    label(ax, 8.35, 7.00, '残差 H', font, size=8.5, color=ORANGE)
    box(ax, 5.45, 3.16, 2.6, 0.52, 'LayerNorm', font)
    box(ax, 5.45, 1.95, 2.6, 0.90, '前馈网络 + Dropout\n640 → 2560 → 640\nGELU 激活', font, size=9.5)
    arrow(ax, (x, 3.97), (x, 3.71))
    arrow(ax, (x, 3.13), (x, 2.88))
    arrow(ax, (x, 1.92), (x, 1.37))
    line_path(ax, [(7.02, 3.88), (8.62, 3.88), (8.62, 1.12), (7.0, 1.12)])
    ax.plot([x, 7.02], [3.88, 3.88], color=ORANGE, linewidth=1.15)
    label(ax, 9.04, 2.65, '残差 U', font, size=8.5, color=ORANGE)
    arrow(ax, (x, 0.87), (x, 0.60))
    label(ax, x, 0.42, '输出 H′', font, size=10)
    save(fig, root, 'gpt_preln')


def mask_shift(root, font):
    fig, ax = canvas((7.1, 3.3), (0, 10), (0, 5.0))
    label(ax, 2.1, 4.7, '因果遮罩的可见位置', font, size=12, color=BLUE)
    label(ax, 7.1, 4.7, '输入和下一 token 标签', font, size=12, color=BLUE)
    tokens = ['BOS', '科技', '展览', '。']
    cell, x0, y0 = 0.68, 1.18, 1.25
    for row in range(4):
        y = y0 + (3-row)*cell
        label(ax, 0.90, y+cell/2, tokens[row], font, size=9, ha='right')
        label(ax, x0+row*cell+cell/2, 4.21, tokens[row], font, size=9)
        for col in range(4):
            allowed = col <= row
            ax.add_patch(Rectangle((x0+col*cell, y), cell, cell,
                facecolor=PALE_BLUE if allowed else '#f3f4f5', edgecolor='white', linewidth=2))
            label(ax, x0+col*cell+cell/2, y+cell/2, '✓' if allowed else '×', font,
                  size=13, color=BLUE if allowed else '#a5acb1')
    label(ax, 2.45, 0.80, '蓝色可读取，灰色为未来位置', font, size=9, color=GREY)
    label(ax, 2.45, 0.39, '行表示当前位置，列表示输入位置', font, size=8.5, color=GREY)
    right_tokens = tokens
    targets = ['科技', '展览', '。', 'EOS']
    for i, (source, target) in enumerate(zip(right_tokens, targets)):
        x = 5.35+i*1.05
        box(ax, x, 3.05, 0.84, 0.67, source, font, size=9)
        box(ax, x, 1.66, 0.84, 0.67, target, font, fill=PALE_ORANGE, edge=ORANGE, size=9)
        arrow(ax, (x+0.42, 3.02), (x+0.42, 2.37), color=ORANGE)
    label(ax, 4.97, 3.38, '输入', font, size=9, ha='right')
    label(ax, 4.97, 2.00, '标签', font, size=9, ha='right')
    label(ax, 7.15, 1.06, '当前位置读取已有前缀，预测下一 token', font, size=9, color=GREY)
    label(ax, 7.15, 0.60, '示意序列用于说明移位与文档结束', font, size=8.5, color=GREY)
    save(fig, root, 'causal_shift')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, help='Path to a Chinese font file')
    parser.add_argument('--output', default='report/generated')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    font = FontProperties(fname=args.font)
    plt.rcParams['svg.fonttype'] = 'path'
    transformer(output, font)
    gpt(output, font)
    mask_shift(output, font)
    print('Drew three original concept figures in PNG and SVG formats.')


if __name__ == '__main__':
    main()
