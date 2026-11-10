"""Memory-mapped training batches and deterministic validation sampling."""
from pathlib import Path

import numpy as np
import torch


class Batches:
    def __init__(self, directory, stage, block_size, batch_size):
        self.root = Path(directory)
        self.stage, self.block_size, self.batch_size = stage, block_size, batch_size
        if stage == "pretrain":
            self.arrays = {s: np.memmap(self.root / f"{s}.bin", mode="r", dtype="<u4") for s in ("train", "val")}
            if any(len(a) < block_size + 1 for a in self.arrays.values()):
                raise ValueError("Token stream too short for block_size")
        elif stage == "sft":
            self.arrays = {s: (np.load(self.root / f"sft_{s}_ids.npy", mmap_mode="r"),
                               np.load(self.root / f"sft_{s}_labels.npy", mmap_mode="r")) for s in ("train", "val")}
            for ids, labels in self.arrays.values():
                if ids.shape != labels.shape or ids.shape[1] != block_size + 1 or not len(ids):
                    raise ValueError("SFT shape mismatch; prepare data with this block_size")
        else:
            raise ValueError(f"Unknown stage {stage}")

    def sample(self, split, generator):
        """返回对齐的输入与下一 token 标签，抽样使用调用方独立的 RNG。"""
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
