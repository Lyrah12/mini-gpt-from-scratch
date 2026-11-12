"""Append measured refinement results to the report; never compile a PDF."""
import argparse
from pathlib import Path
import shutil

from minigpt.common import jsonl, read_json, sha256, write_json
from scripts.export_report import tex


def export(root, report='report'):
    root, report = Path(root), Path(report)
    summary = read_json(root / 'summary.json')
    selection = summary['selection']['selection']
    if not summary['training_status']['complete']:
        raise ValueError('Refinement training incomplete')
    if summary['training_run']['init_checkpoint_sha256'] != sha256('runs/local_4060/pretrain/best.pt'):
        raise ValueError('Unexpected pretraining initialization')
    for name, result in summary['test'].items():
        folder = root / ('test_lead32_lead' if name == 'lead32_lead' else 'test_' + name)
        if sha256(folder / 'predictions.jsonl') != result['predictions_sha256']:
            raise ValueError('Test predictions changed: ' + name)
        if name != 'lead32_lead' and result['identity']['mode'] != selection[name]['mode']:
            raise ValueError('Test did not use frozen validation selection')
    metadata_hash = sha256(report / 'metadata.tex')
    backup = root / 'report_before_refinement'
    backup.mkdir(exist_ok=True)
    for filename in ('main.tex', 'README.md'):
        if not (backup / filename).exists():
            shutil.copyfile(report / filename, backup / filename)
    val = summary['validation']
    tests = summary['test']
    labels = {'original_sft_text': '原微调／仅正文', 'original_sft_lead': '原微调／导语加正文',
              'refined_sft_lead': '重新微调／导语加正文'}
    mode_labels = {'greedy': '贪心', 'penalty_1p1': '重复惩罚1.1', 'no_repeat_3': '禁重复3-gram'}
    lines = [r'\subsection{检查后补充实验：输入信息与生成重复}',
        '初始实验完成后，逐条核对编码、损失遮罩与评测结果，未发现本轮检查范围内的致命实现错误。'
        '回查原始 XL-Sum 发现，summary 字段中有时包含 text 字段未保留的新闻导语；例如拘捕人数只在导语中出现。'
        '因此补充实验把原始 summary 作为导语放在原正文前，以 title 为目标，未把参考标题人为拼入输入。'
        '这改变了输入信息与任务难度，不能把分数变化全部归因于模型或解码算法。',
        '补充数据严格沿用初始实验保留的全部新闻、官方划分、文档 ID 和顺序，仍为训练36762、验证4585、测试4662篇；'
        '分词器、约4996万参数架构和512-token上下文均保持不变。正文预算仍为419 token，优先保留导语，长文截断仍然存在。',
        '新增微调从与初始主方法相同的自训练预训练检查点开始，执行相同的1500步、有效批量32、最大学习率'
        r'$5\times10^{-5}$及随机种子42，没有增加预训练，也没有引入外部模型权重。'
        f"新增训练迭代累计{summary['training_status']['train_seconds']/60:.1f}分钟，不含准备、验证、存盘和生成评测。",
        '为区分输入变化与任务适配，分别比较原微调权重的仅正文输入、同一权重的导语加正文输入，以及重新微调后的导语加正文输入。'
        '每组在固定200条验证新闻上比较贪心、仅对已生成token施加1.1重复惩罚，以及禁止重复已生成token三元组三种设置；'
        '统一贪心选取调整后logit最大值，最多生成64个token，原文token不参与重复惩罚。',
        r'\begin{center}\small\begin{tabular}{lrrr}\toprule',
        r'验证集输入与模型 & 贪心 & 惩罚1.1 & 禁重复3-gram \\ \midrule']
    for name, label in labels.items():
        values = [100*val[name+'/'+mode]['metrics']['rougeL'] for mode in ('greedy', 'penalty_1p1', 'no_repeat_3')]
        lines.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
        '上表为字符级ROUGE-L乘以100。预先规定每组选择验证ROUGE-L最高的设置；相同时先选择重复提示较少的设置，再按声明顺序决胜。'
        '选项与选择结果在本轮测试前写入policy.json与selection.json。初始测试子集此前已经查看，因此本次不声称测试集从未被接触，'
        '也不根据本轮测试结果反过来改变配置。',
        r'\begin{center}\small\begin{tabular}{lrrr}\toprule',
        r'固定500条测试新闻 & ROUGE-1 & ROUGE-2 & ROUGE-L \\ \midrule']
    table = [('初始首句抽取／正文', summary['original_test_lead32']['metrics']),
             ('初始微调／正文／贪心', summary['original_test_sft_greedy']['metrics'])]
    table += [(labels[name] + '／' + mode_labels[selection[name]['mode']], tests[name]['metrics']) for name in labels]
    table += [('首句抽取／导语加正文', tests['lead32_lead']['metrics'])]
    for label, values in table:
        lines.append(label + ' & ' + ' & '.join(f'{100*values[k]:.2f}' for k in ('rouge1', 'rouge2', 'rougeL')) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    improved = tests['refined_sft_lead']['metrics']['rougeL']
    input_only = tests['original_sft_lead']['metrics']['rougeL']
    lead = tests['lead32_lead']['metrics']['rougeL']
    lines.append(f'在各自验证集选定的生成设置下，重新微调的ROUGE-L相对同输入的原权重为{100*(improved-input_only):+.2f}个点，'
                 f'相对同输入抽取基线为{100*(improved-lead):+.2f}个点。输入比较需结合验证集贪心列观察，不能忽略解码设置不同的影响。')
    final = tests['refined_sft_lead']
    lines.append(f"重新微调的500条输出中，空输出{final['empty_n']}条，原文四字符片段至少重复三次的提示"
                 f"{final['raw_repetition_flags']}条，数字未在其完整输入中直接出现的提示{final['numeric_flags']}条。"
                 '后两项不是事实错误率；加入导语也改变了数字筛查的输入范围，不能直接据此宣称事实性改善。')
    validation_rows = list(jsonl(root / ('val_refined_sft_lead_' + selection['refined_sft_lead']['mode']) / 'predictions.jsonl'))[:8]
    lines.append('另外，对固定验证子集按文档ID排序的前8条作人工复核，可见有些输出更贴近输入，但数字和事件关系仍可能错误。'
        '例如以下案例位于这8条之内；它们用于解释失败机制，不构成人工事实准确率，也未用于改变已冻结的生成设置。')
    for prefix, diagnosis in (
        ('01265c776a59', '原始导语明确写了扣押12名嫌疑犯，输出仍写7名，说明补全信息并不能保证数字复制正确。'),
        ('03b663645558', '输出比原来关于央行宣布措施的说法更接近当前导语，抓住了经济状况恶化。'),
        ('04984e8792c6', '导语描述完成六次陈江会后离台，输出却出现“六四后”，属于输入未支持的事件或时间表述。')):
        row = next(r for r in validation_rows if r['id'].startswith(prefix))
        lines += [r'\paragraph{验证案例 ' + prefix + '}',
            r'\textbf{参考标题：}' + tex(row['reference']) + r'\par',
            r'\textbf{生成标题：}' + tex(row['prediction']) + r'\par', tex(diagnosis)]
    lines.append('以下展示相同测试子集按文档ID排序的前四条，未根据输出质量挑选。新旧方法接收的信息不同，必须结合原始逐条输入检查事实。')
    old_rows = {r['id']: r for r in jsonl('runs/local_4060/eval_test_sft/predictions.jsonl')}
    for i, row in enumerate(list(jsonl(root / 'test_refined_sft_lead/predictions.jsonl'))[:4], 1):
        lines += [r'\paragraph{补充样例 ' + str(i) + '}',
            r'\textbf{参考标题：}' + tex(row['reference']) + r'\par',
            r'\textbf{初始输出：}' + tex(old_rows[row['id']]['prediction']) + r'\par',
            r'\textbf{补充实验输出：}' + tex(row['prediction']) + r'\par']
    lines += ['该实验只覆盖一个随机种子、一组固定训练预算和少量解码候选。输入完整性和去重复约束都不能保证人物关系、数字或时间关系正确。'
        '若用户仅提供正文而没有导语，不能凭空补造导语，也不能将本次导语加正文评测的指标直接承诺给该输入条件。']
    generated = report / 'generated'
    (generated / 'refinement.tex').write_text('\n\n'.join(lines) + '\n', encoding='utf8')
    write_json(generated / 'refinement_summary.json', summary)
    abstract = (f'补充实验将原始summary作为导语补入输入，在相同预训练起点和1500步微调预算下，'
        f'按验证集选择的重新微调模型在同一500条测试新闻上取得字符级ROUGE-L {100*improved:.2f}，'
        f'同输入首句抽取为{100*lead:.2f}；该比较改变了输入信息，不能把相对初始实验的变化全部归因于模型算法。')
    (generated / 'refinement_abstract.tex').write_text(abstract + '\n', encoding='utf8')
    main = (report / 'main.tex').read_text(encoding='utf8')
    marker = r'\input{generated/refinement.tex}'
    if marker not in main:
        main = main.replace(r'\input{generated/analysis.tex}', r'\input{generated/analysis.tex}' + '\n' + marker)
    abstract_marker = r'\input{generated/refinement_abstract.tex}'
    if abstract_marker not in main:
        main = main.replace(r'\input{generated/abstract_results.tex}', r'\input{generated/abstract_results.tex}' + abstract_marker)
    main = main.replace('标题的定量评测统一用贪心解码；', '初始标题定量评测采用贪心解码，补充实验另比较两种去重复设置；')
    main = main.replace('再进行云端正式训练', '再进行本地正式训练')
    (report / 'main.tex').write_text(main, encoding='utf8')
    readme = (report / 'README.md').read_text(encoding='utf8')
    if 'refinement.tex' not in readme:
        readme += '\n2026-11-04补充实验已写入 `generated/refinement.tex`，保留初始实验，并明确导语输入变化与验证集选参。无需额外上传权重或数据；继续使用XeLaTeX，未在本地编译PDF。\n'
    (report / 'README.md').write_text(readme, encoding='utf8')
    report_status = read_json(generated / 'report_status.json')
    report_status['refinement'] = {'complete': True, 'input': 'original summary + text',
        'selected_mode': selection['refined_sft_lead']['mode'], 'test_examples': final['n'],
        'rougeL_percent': 100*improved, 'lead32_same_input_rougeL_percent': 100*lead,
        'policy_sha256': summary['selection']['policy_sha256'], 'pdf_compiled_locally': False}
    write_json(generated / 'report_status.json', report_status)
    status = (report / 'FINAL_STATUS.md').read_text(encoding='utf8')
    if 'refinement.tex' not in status:
        status += '\n2026-11-04补充实验已完成，见generated/refinement.tex和refinement_summary.json；初始结果仍保留。输入变化与验证选参已明确说明。\n'
        (report / 'FINAL_STATUS.md').write_text(status, encoding='utf8')
    assert sha256(report / 'metadata.tex') == metadata_hash
    doc = ['# 有限优化实验（2026-11-04）', '',
           '本轮新增一组同预算微调，原实验和权重保留。输入从原text字段改为原summary导语加text，目标仍为title；这是输入条件变化，不能把全部提升称作算法收益。', '',
           '| 模型／输入 | 验证集选择 | 验证ROUGE-L | 测试ROUGE-L | 测试重复提示 |',
           '|---|---|---:|---:|---:|']
    for name, label in labels.items():
        chosen, test = selection[name], tests[name]
        doc.append(f"| {label} | {mode_labels[chosen['mode']]} | {100*chosen['validation']['metrics']['rougeL']:.2f} | {100*test['metrics']['rougeL']:.2f} | {test['raw_repetition_flags']}/500 |")
    doc += ['', f"同输入的导语加正文首句抽取测试ROUGE-L：{100*lead:.2f}。重复提示只是启发式，不代表事实正确率。",
        '', f"新增模型数字筛查提示为{final['numeric_flags']}/500；该指标不是事实错误率。原初始贪心结果为9/500，但输入信息和生成设置均已改变，不能据此给出可靠性增减的定量结论。",
        '', '训练：沿用原预训练best.pt、原训练集BPE、种子42及原1500步微调配置。原始ID、排序和划分不变；验证集200条选解码，测试500条只用于所选设置的最终记录。测试子集此前已用于初始实验，并非从未接触。',
        '', f"实际新增训练迭代：{summary['training_status']['train_seconds']/60:.1f}分钟。原始数据许可CC-BY-NC-SA-4.0，源版本及文件校验值见数据manifest。没有使用外部预训练模型。",
        '', '## 人工复核的剩余问题', '',
        '固定验证子集按ID排序的前8条中，仍有数字和事件关系错误：巴黎袭击新闻的原始导语明确写了12名嫌疑犯，模型仍输出7名；陈云林会后离台的报道被生成为“六四后两岸和平大趋势”。另有经济新闻的输出更贴近导语。上述案例已补入报告，不代表人工事实准确率。ROUGE和重复提示改善，不能说明模型已经可靠；本轮不继续扩大模型或追加无明确依据的训练。',
        '', '## 复现', '', '在项目根目录运行：', '', '```powershell',
        '.\\work\\venv_cuda\\Scripts\\python.exe -B -X utf8 -m scripts.run_refinement', '```',
        '', '暂停：在runs/refinement_20261104/创建pause.request，训练会完成当前步并保存；继续前移除该文件，再运行同一命令。不要删除pipeline.lock来并行启动第二个相同实验。',
        '', '## 使用新增权重', '', '根据上表选择模式，提供真实新闻正文及可选原始导语；没有导语时不要编造。', '', '```powershell',
        '.\\work\\venv_cuda\\Scripts\\python.exe -B -X utf8 -m scripts.title_decoding --checkpoint runs/refinement_20261104/sft/best.pt --mode ' + selection['refined_sft_lead']['mode'] + ' --lead "新闻原始导语" --text "新闻正文"', '```',
        '', '详细策略、各候选验证预测、冻结选择与所选测试预测位于runs/refinement_20261104/。这些产物均被现有.gitignore排除。报告添加补充实验，不覆盖原始表格；姓名与学号仍占位，未在本地编译PDF。']
    (root / 'refinement_notes.md').write_text('\n'.join(doc) + '\n', encoding='utf8')
    print('Refinement results exported; no PDF compiled.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', default='runs/refinement_20261104')
    ap.add_argument('--report', default='report')
    a = ap.parse_args()
    export(a.runs, a.report)


if __name__ == '__main__':
    main()
