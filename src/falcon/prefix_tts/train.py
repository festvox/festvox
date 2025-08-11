#!/usr/bin/env python3
"""
Training script for the prefix‑LM TTS model.

This is a minimal training loop intended to demonstrate how to load the
prepared dataset, construct combined text/audio token sequences and train a
Transformer language model.  It uses PyTorch and does not depend on
external libraries such as HuggingFace.  For large‑scale training you
should consider integrating with an accelerator framework (e.g. PyTorch
Lightning or HuggingFace Accelerate) and adding features such as mixed
precision, gradient accumulation, checkpointing and logging.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TTSPrefixLM, TTSPrefixLMConfig


class PrefixLMSequenceDataset(Dataset):
    """Dataset yielding combined text/audio token sequences for prefix‑LM training."""

    def __init__(self, jsonl_path: Path, pad_token_id: int = 0) -> None:
        self.examples: List[dict] = []
        with jsonl_path.open() as f:
            for line in f:
                data = json.loads(line)
                # Our prepared data already has input_ids and labels
                self.examples.append(data)
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[idx]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of sequences to the same length."""
    input_ids_batch, labels_batch = zip(*batch)
    
    max_len = max(x.size(0) for x in input_ids_batch)
    
    # Pad input_ids and labels
    padded_input_ids = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)  # -100 is ignored by CrossEntropyLoss
    
    for i, (input_ids, labels) in enumerate(zip(input_ids_batch, labels_batch)):
        seq_len = input_ids.size(0)
        padded_input_ids[i, :seq_len] = input_ids
        padded_labels[i, :seq_len] = labels
    
    return padded_input_ids, padded_labels


def train_epoch(model: TTSPrefixLM, dataloader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.CrossEntropyLoss, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids)  # Model predicts next tokens for all positions
        
        # Compute loss using the pre-computed labels
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        # Count non-ignored tokens for averaging
        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens
        
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prefix‑LM TTS model")
    parser.add_argument("--train_jsonl", type=Path, required=True, help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", type=Path, help="Path to validation JSONL file (optional)")
    parser.add_argument("--vocab_json", type=Path, required=True, help="Path to vocabulary JSON file")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=Path, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Load vocabulary info
    with open(args.vocab_json) as f:
        vocab_info = json.load(f)
    vocab_size = vocab_info["total_vocab_size"]
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training on device: {args.device}")

    # Create datasets
    train_dataset = PrefixLMSequenceDataset(args.train_jsonl)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    val_dataloader = None
    if args.val_jsonl and args.val_jsonl.exists():
        val_dataset = PrefixLMSequenceDataset(args.val_jsonl)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Build model
    config = TTSPrefixLMConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )
    model = TTSPrefixLM(config).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # -100 is ignored (padding)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device=torch.device(args.device))
        print(f"Epoch {epoch}: train loss {train_loss:.4f}")
        
        # Validation if available
        if val_dataloader:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for input_ids, labels in val_dataloader:
                    input_ids = input_ids.to(args.device)
                    labels = labels.to(args.device)
                    logits = model(input_ids)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    valid_tokens = (labels != -100).sum().item()
                    val_loss += loss.item() * valid_tokens
                    val_tokens += valid_tokens
            val_loss = val_loss / val_tokens if val_tokens > 0 else 0.0
            print(f"Epoch {epoch}: val loss {val_loss:.4f}")
    
    # Save checkpoint
    args.save_dir.mkdir(exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'vocab_info': vocab_info,
        'epoch': args.epochs,
    }
    ckpt_path = args.save_dir / "prefix_lm_tts.pth"
    torch.save(checkpoint, ckpt_path)
    print(f"Saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()