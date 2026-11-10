"""Build the coursework report's result tables, examples and plots from saved runs."""
import argparse
from pathlib import Path
import sys


def table(caption, label, columns, headings, rows):
    lines = [r'\begin{table}[H]', r'\centering\small',
             r'\caption{' + caption + r'}\label{' + label + '}',
             r'\begin{tabularx}{\linewidth}{' + columns + '}',
             r'\toprule ' + headings + r' \\ \midrule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabularx}', r'\end{table}'])
    return '\n'.join(lines)


def plot_stages(logs, names, titles, destination, size):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, len(names), figsize=size, squeeze=False)
    for axis, name, title in zip(axes.flat, names, titles):
        training = [row for row in logs[name] if row['event'] == 'train']
        validation = [row for row in logs[name] if row['event'] == 'val']
        axis.plot([row['step'] for row in training], [row['loss'] for row in training],
                  color='#a9b5c3', linewidth=0.8, label='Train (logged)')
        axis.plot([row['step'] for row in validation], [row['loss'] for row in validation],
                  'o-', color='#216185', linewidth=1.5, markersize=3, label='Fixed validation')
        axis.set(title=title, xlabel='Optimizer step', ylabel='Cross entropy')
        axis.grid(alpha=0.18)
        axis.annotate(f"{validation[-1]['loss']:.4f}",
                      (validation[-1]['step'], validation[-1]['loss']),
                      xytext=(-4, 7), textcoords='offset points', ha='right', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.77, wspace=0.30)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def build(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    sys.path.insert(0, str(project))
    from minigpt.common import jsonl, read_json, sha256, write_json
    from scripts.export_report import tex as escape_tex, validate_inputs

    def tex(text):
        # Force the supplied glyph for a character rendered as a box by the report font.
        return escape_tex(text).replace('牠', r'\ReportAnimalPronoun{}')

    original = project / 'runs/local_4060'
    refinement = project / 'runs/refinement_20261104'
    _, runs, statuses, initial, continuations = validate_inputs(original, project / 'data/processed_xlsum')
    refined = read_json(refinement / 'summary.json')
    if not refined['training_status']['complete']:
        raise ValueError('Refinement training is incomplete')
    if refined['selection']['policy_sha256'] != sha256(refinement / 'policy.json'):
        raise ValueError('Refinement selection policy has changed')
    if refined['training_run']['config'] != runs['sft']['config']:
        raise ValueError('The two SFT budgets differ')
    if refined['training_run']['init_checkpoint_sha256'] != sha256(original / 'pretrain/best.pt'):
        raise ValueError('Refinement initialization has changed')
    selection_hashes = {entry['selection_hash'] for entry in initial.values()}
    for name, entry in refined['test'].items():
        folder = refinement / ('test_lead32_lead' if name == 'lead32_lead' else 'test_' + name)
        if sha256(folder / 'predictions.jsonl') != entry['predictions_sha256']:
            raise ValueError('Saved predictions have changed')
        selection_hashes.add(entry['identity']['selection_hash'])
        if name != 'lead32_lead' and entry['identity']['mode'] != refined['selection']['selection'][name]['mode']:
            raise ValueError('Test decoding differs from validation selection')
    if len(selection_hashes) != 1:
        raise ValueError('The comparisons use different test documents')

    generated = output / 'generated'
    generated.mkdir(parents=True, exist_ok=True)
    paths = {'pretrain': original / 'pretrain', 'sft': original / 'sft',
             'scratch': original / 'scratch', 'refined': refinement / 'sft'}
    logs = {name: list(jsonl(path / 'metrics.jsonl')) for name, path in paths.items()}
    best = {name: min((row for row in rows if row['event'] == 'val'), key=lambda row: row['loss'])
            for name, rows in logs.items()}
    first = next(row for row in logs['pretrain'] if row['event'] == 'val')
    test = refined['test']

    pretrain = [f"预训练固定验证交叉熵由{first['loss']:.4f}降至{best['pretrain']['loss']:.4f}，"
        f"困惑度由{first['ppl']:.2f}降至{best['pretrain']['ppl']:.2f}。最佳检查点位于第{best['pretrain']['step']}步，"
        f"累计处理98,304,000个输入token，训练迭代用时{statuses['pretrain']['train_seconds']/60:.1f}分钟。"
        r'图\ref{fig:pretrain}显示，损失在前期下降较快，后期下降幅度减小。']
    selected_steps = (0, 1000, 2000, 3000)
    validation_checkpoints = {row['step']: row for row in logs['pretrain'] if row['event'] == 'val'}
    checkpoint_rows = [[str(step), f"{validation_checkpoints[step]['loss']:.4f}",
                        f"{validation_checkpoints[step]['ppl']:.2f}"] for step in selected_steps]
    pretrain += [table('预训练验证结果', 'tab:pretrain_checkpoints', 'Y r r',
        '训练步数 & 平均交叉熵 & 困惑度', checkpoint_rows),
        r'\begin{figure}[H]',
        r'\centering\includegraphics[width=0.86\linewidth]{generated/pretrain_curve.png}',
        r'\caption{预训练损失曲线}\label{fig:pretrain}', r'\end{figure}',
        '固定使用“人工智能技术的发展”“今天上午，学校举办”“随着城市公共交通的发展”三个续写提示。'
        '随机种子为42，温度为0.8，top-k为40，最多生成64个新token。第一个提示的随机初始化输出取前80个字符，预训练后输出保留该上限内的完整片段。']
    before, after = continuations['samples']['before'][0], continuations['samples']['after'][0]
    pretrain += [r'\begin{quote}', r'\begin{minipage}{\linewidth}\small', r'\textbf{提示}\quad ' + tex(before['prompt']) + r'\par',
        r'\textbf{随机初始化}\quad ' + tex(before['continuation'][:80]) + r'\par',
        r'\textbf{预训练后}\quad ' + tex(after['continuation']) + r'\par', r'\end{minipage}', r'\end{quote}',
        '随机初始化输出主要是词语拼接，预训练后开始形成句子结构，但仍反复出现“人类”“进化”等词语。'
        '模型对句式和常见词组的学习与验证损失下降相符，连续重复仍反映出语义组织的不足。']
    school = continuations['samples']['after'][1]
    pretrain += [r'\begin{quote}', r'\begin{minipage}{\linewidth}\small',
        r'\textbf{提示}\quad ' + tex(school['prompt']) + r'\par',
        r'\textbf{预训练后}\quad ' + tex(school['continuation']) + r'\par', r'\end{minipage}', r'\end{quote}',
        '输出从学校活动转向曼德拉纪念仪式，并混合了不同时间表达。新闻语料中的常见实体与事件已经进入生成分布，'
        '但模型对提示主题和时间关系的约束仍较弱。本次验证损失由固定窗口估计，生成样例进一步展示了其内容组织能力的边界。']
    (generated / 'style_pretrain.tex').write_text('\n\n'.join(pretrain) + '\n', encoding='utf8')

    initial_rows = []
    for name, label in [('lead', '首句前32字符'), ('pretrain', '仅预训练'),
                        ('scratch', '随机初始化微调'), ('sft', '预训练后微调')]:
        initial_rows.append([label] + [f"{100*initial[name]['metrics'][metric]:.2f}" for metric in ('rouge1','rouge2','rougeL')])
    sft = [r'表\ref{tab:initial}给出同一500条测试新闻上的标题生成结果。预训练后微调的ROUGE-L为'
        f"{100*initial['sft']['metrics']['rougeL']:.2f}，随机初始化微调为{100*initial['scratch']['metrics']['rougeL']:.2f}，"
        f"前者高出{100*(initial['sft']['metrics']['rougeL']-initial['scratch']['metrics']['rougeL']):.2f}个分值点。"
        f"与同输入抽取基线相比，预训练后微调高出{100*(initial['sft']['metrics']['rougeL']-initial['lead']['metrics']['rougeL']):.2f}个分值点。",
        table('标题生成对照结果', 'tab:initial', 'Y r r r',
              '方法 & ROUGE-1 & ROUGE-2 & ROUGE-L', initial_rows),
        r'图\ref{fig:sft}比较两组微调损失，两个子图分别设置纵轴范围。',
        r'\begin{figure}[H]',
        r'\centering\includegraphics[width=\linewidth]{generated/sft_comparison.png}',
        r'\caption{标题微调损失曲线}\label{fig:sft}', r'\end{figure}',
        f"两组最佳验证交叉熵分别为{best['sft']['loss']:.4f}和{best['scratch']['loss']:.4f}，均出现在第1500步。"
        '仅预训练模型的标题得分较低，监督微调有助于学习标题格式和信息选择。'
        '预训练组还使用了额外预训练计算，随机初始化组沿用同一微调配置，上述差异反映当前设置下初始化与训练方案的共同影响。',
        f"在相同的标题监督范围下，两组固定验证困惑度分别为{best['sft']['ppl']:.2f}和{best['scratch']['ppl']:.2f}。"
        '预训练组更容易利用已有语言表示拟合标题目标。只有预训练时，模型尚未充分适配“读正文、生成标题”的任务格式，'
        '因此损失与生成评测共同支持预训练和任务监督分别发挥作用的解释。']
    (generated / 'style_sft.tex').write_text('\n\n'.join(sft) + '\n', encoding='utf8')

    groups = [('original_sft_text', '原微调权重', '正文'),
              ('original_sft_lead', '原微调权重', '导语加正文'),
              ('refined_sft_lead', '重新微调权重', '导语加正文')]
    modes = ('greedy', 'penalty_1p1', 'no_repeat_3')
    validation_rows = [[model + '，' + source] + [f"{100*refined['validation'][name+'/'+mode]['metrics']['rougeL']:.2f}" for mode in modes]
                       for name, model, source in groups]
    refinement_lines = [r'三组验证结果见表\ref{tab:validation}，均由重复惩罚1.1获得最高ROUGE-L，因此采用该设置进入测试。',
        table('解码方法比较', 'tab:validation', 'Y r r r',
              '模型与新闻输入 & 贪心 & 惩罚1.1 & 禁重复3-gram', validation_rows)]
    test_rows = [[model, source] + [f"{100*test[name]['metrics'][metric]:.2f}" for metric in ('rouge1','rouge2','rougeL')]
                 for name, model, source in groups]
    test_rows.append(['首句抽取', '导语加正文'] + [f"{100*test['lead32_lead']['metrics'][metric]:.2f}" for metric in ('rouge1','rouge2','rougeL')])
    refinement_lines.append(table('输入与微调改进结果', 'tab:refinement', 'Y l r r r',
        '模型 & 新闻输入 & ROUGE-1 & ROUGE-2 & ROUGE-L', test_rows))
    initial_l = 100*initial['sft']['metrics']['rougeL']
    decoded_l = 100*test['original_sft_text']['metrics']['rougeL']
    lead_l = 100*test['original_sft_lead']['metrics']['rougeL']
    final_l = 100*test['refined_sft_lead']['metrics']['rougeL']
    baseline_l = 100*test['lead32_lead']['metrics']['rougeL']
    refinement_lines += [r'表\ref{tab:refinement}中，原权重使用正文并调整解码后，ROUGE-L为'
        f'{decoded_l:.2f}，比初始贪心结果高{decoded_l-initial_l:.2f}个分值点。加入导语后达到{lead_l:.2f}，'
        f'再提高{lead_l-decoded_l:.2f}个分值点。在相同输入和解码下，重新微调达到{final_l:.2f}，'
        f'相对原权重提高{final_l-lead_l:.2f}个分值点，相对同输入抽取基线提高{final_l-baseline_l:.2f}个分值点。',
        '总分变化包含解码、输入信息和重新微调三项因素，各项增量依赖当前条件。'
        f"新增微调的最佳验证交叉熵为{best['refined']['loss']:.4f}，训练迭代用时{refined['training_status']['train_seconds']/60:.1f}分钟。",
        r'\begin{figure}[H]',
        r'\centering\includegraphics[width=0.82\linewidth]{generated/refinement_curve.png}',
        r'\caption{改进微调损失曲线}\label{fig:refinement}', r'\end{figure}']
    (generated / 'style_refinement.tex').write_text('\n\n'.join(refinement_lines) + '\n', encoding='utf8')

    original_predictions = {row['id']: row for row in jsonl(original / 'eval_test_sft/predictions.jsonl')}
    final_predictions = list(jsonl(refinement / 'test_refined_sft_lead/predictions.jsonl'))[:4]
    example_rows = [[tex(row['reference']), tex(original_predictions[row['id']]['prediction']), tex(row['prediction'])]
                    for row in final_predictions]
    examples = [r'表\ref{tab:examples}按文档ID顺序展示固定测试子集的前四条。初始方案输入正文，最终方案输入导语加正文。',
        table('标题生成样例', 'tab:examples', 'Y Y Y',
              '参考标题 & 初始方案 & 最终方案', example_rows),
        '样例中有些重复减少，上海自贸区新闻的输出也更贴近事件主题，但动物特工样例仍存在语义不清的问题。'
        '为进一步观察数字和事件关系，从按文档ID排序的前8条验证样例中取以下两例。']
    selected_mode = refined['selection']['selection']['refined_sft_lead']['mode']
    validation_predictions = list(jsonl(refinement / ('val_refined_sft_lead_' + selected_mode) / 'predictions.jsonl'))[:8]
    for prefix, explanation in [('01265c776a59', '导语明确给出12名嫌疑犯，输出仍写成7名，具体人数被错误生成。'),
        ('04984e8792c6', '导语描述完成六次陈江会后离台，输出却出现“六四后”，改变了事件含义。')]:
        row = next(row for row in validation_predictions if row['id'].startswith(prefix))
        lead = row['content'].split('。', 1)[0] + '。'
        examples += [r'\begin{quote}\small', r'\textbf{输入片段}\quad ' + tex(lead) + r'\par',
            r'\textbf{生成标题}\quad ' + tex(row['prediction']) + r'\par', r'\end{quote}', explanation]
    final = test['refined_sft_lead']
    examples.append(f"初始贪心输出中，四字符片段至少重复三次的提示有17条，最终方案降至{final['raw_repetition_flags']}条。"
        f"最终输出中还有{final['numeric_flags']}条数字片段未直接出现在完整输入中。"
        '这些规则用于定位待复核样本，事实判断还需检查上下文与表达转换。上述案例表明，输入补全和重复约束之后，数字与事件关系仍可能出错。')
    (generated / 'style_examples.tex').write_text('\n\n'.join(examples) + '\n', encoding='utf8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    plot_stages(logs, ['pretrain'], ['Pretraining'], generated / 'pretrain_curve.png', (6.0, 2.8))
    plot_stages(logs, ['sft', 'scratch'], ['SFT from pretraining', 'SFT from random initialization'],
                generated / 'sft_comparison.png', (6.8, 2.9))
    plot_stages(logs, ['refined'], ['SFT with lead + text'], generated / 'refinement_curve.png', (5.5, 2.7))
    write_json(generated / 'report_revision.json', {
        'revision_date': '2026-11-06', 'style': 'user-supplied coursework one and two',
        'experiment_data_reused': True, 'refinement_summary_sha256': sha256(refinement / 'summary.json'),
        'initial_sft_rougeL_percent': initial_l, 'refined_sft_rougeL_percent': final_l,
        'same_input_training_delta_points': final_l-lead_l,
        'pdf_compiled_locally': False, 'overleaf_page_layout_checked': False,
        'active_assets': ['style_pretrain.tex', 'style_sft.tex', 'style_refinement.tex', 'style_examples.tex',
                          'pretrain_curve.png', 'sft_comparison.png', 'refinement_curve.png']})
    print('Built report results and figures from the completed experiments.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default='.')
    parser.add_argument('--output', default='report')
    args = parser.parse_args()
    build(args.project, args.output)


if __name__ == '__main__':
    main()
