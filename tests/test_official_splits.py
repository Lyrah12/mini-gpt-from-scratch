"""检查官方分区归属与训练集词表学习范围。"""
# 导入本模块使用的标准库工具。
import json
from pathlib import Path
# 复用项目内的数据、模型或实验工具。
from minigpt import prepare as module
from minigpt.common import jsonl


# 检查官方分区被保留，词表学习只读取训练文本。
def test_official_splits_preserved_and_tokenizer_sees_only_train(tmp_path, monkeypatch):
    raw=tmp_path/"raw.jsonl"
    records=[]
    for i in range(120):
        split="train" if i<90 else "val" if i<105 else "test"
        marker={"train":"训练甲","val":"验证乙","test":"测试丙"}[split]
        records.append({"title":"学校举办新闻活动", "content":f"{marker}第{i}号学校消息。"+"记者介绍，学校组织学生参加科学与文化活动，现场设置互动体验区，帮助学生将理论知识用于实践。"*4,"_split":split})
    records.append({**records[0],"_split":"test"})
    raw.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    captured=[]; original=module.train_tokenizer
    # 记录分词器实际接收到的文本，供后续断言检查。
    def observe(texts,*args,**kwargs):
        texts=list(texts);captured.extend(texts)
        return original(texts,*args,**kwargs)
    # 替换训练接口以观察文本来源，再执行保留官方分区的数据准备。
    monkeypatch.setattr(module,"train_tokenizer",observe)
    out=tmp_path/"prepared"
    manifest=module.prepare([str(raw)],out,max_records=500,vocab_size=512,block_size=256,preserve_splits=True)
    assert manifest["counts"]=={"train":90,"val":15,"test":15}
    assert all("验证乙" not in text and "测试丙" not in text for text in captured)
    for split in ("train","val","test"):
        rows=list(jsonl(out/f"{split}.jsonl"))
        assert all(row["_split"]==split for row in rows)
    assert manifest["statistics"]["duplicate_body_or_lead"]==1
