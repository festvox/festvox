"""CGN v2 dataset and data loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .tokenizer import PAD


class CGNv2Dataset(Dataset):
    """Load pre-built JSONL sequences for CGN v2 training."""

    def __init__(self, jsonl_path: str | Path, max_seq_len: int = 4096):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if len(data["input_ids"]) <= max_seq_len:
                    self.samples.append(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
        }


def collate_left_pad(batch: list[dict], pad_id: int = PAD) -> dict:
    """Left-pad batch for causal LM training.

    Left-padding ensures the final token is always at the rightmost position,
    which is correct for causal attention masks.
    """
    max_len = max(len(s["input_ids"]) for s in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for s in batch:
        seq_len = len(s["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), s["input_ids"]])
        )
        labels.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), s["labels"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(seq_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def build_dataloaders(
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    batch_size: int = 8,
    max_seq_len: int = 4096,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders with optional DDP."""
    train_ds = CGNv2Dataset(train_jsonl, max_seq_len)
    val_ds = CGNv2Dataset(val_jsonl, max_seq_len)

    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_left_pad,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_left_pad,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
