"""以内存映射方式读取训练数据，构造下一 token 预测批次。"""
# 导入本模块使用的标准库工具。
from pathlib import Path

# 使用 NumPy 内存映射与 PyTorch 张量构造训练批次。
import numpy as np
import torch


# 分别管理连续 token 流和标题微调数组的批次采样。
class Batches:
    # 以内存映射方式打开数据，并检查长度与输入标签形状。
    def __init__(self, directory, stage, block_size, batch_size):
        self.root = Path(directory)
        self.stage, self.block_size, self.batch_size = stage, block_size, batch_size
        # 预训练按连续 token 流读取，微调按样本行读取输入与标签。
        if stage == "pretrain":
            self.arrays = {s: np.memmap(self.root / f"{s}.bin", mode="r", dtype="<u4") for s in ("train", "val")}
            if any(len(a) < block_size + 1 for a in self.arrays.values()):
                raise ValueError("Token stream too short for block_size")
        # 微调按行保存输入和标签，两份数组形状与窗口长度应一致。
        elif stage == "sft":
            self.arrays = {s: (np.load(self.root / f"sft_{s}_ids.npy", mmap_mode="r"),
                               np.load(self.root / f"sft_{s}_labels.npy", mmap_mode="r")) for s in ("train", "val")}
            for ids, labels in self.arrays.values():
                if ids.shape != labels.shape or ids.shape[1] != block_size + 1 or not len(ids):
                    raise ValueError("SFT shape mismatch; prepare data with this block_size")
        else:
            raise ValueError(f"Unknown stage {stage}")

    # 抽取指定分区的样本，将输入与下一 token 标签对齐。
    def sample(self, split, generator):
        """返回对齐的输入与下一 token 标签，抽样使用调用方独立的 RNG。"""
        # 每个窗口多取一个 token，分别截取为输入和后继标签。
        if self.stage == "pretrain":
            a = self.arrays[split]
            starts = torch.randint(len(a) - self.block_size, (self.batch_size,), generator=generator).tolist()
            full = torch.from_numpy(np.stack([a[i:i + self.block_size + 1] for i in starts]).astype(np.int64))
            return full[:, :-1].contiguous(), full[:, 1:].contiguous()
        ids, labels = self.arrays[split]
        indices = torch.randint(len(ids), (self.batch_size,), generator=generator).numpy()
        # 行索引已产生独立数组，astype 仅在需要时转换为 PyTorch 的整数类型。
        # labels 左移一位后，提示末尾位置恰好监督第一个标题 token。
        x = torch.from_numpy(ids[indices, :-1].astype(np.int64, copy=False))
        y = torch.from_numpy(labels[indices, 1:].astype(np.int64, copy=False))
        if not (y != -100).any():
            raise ValueError("No supervised target tokens in batch")
        return x, y
