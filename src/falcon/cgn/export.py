"""CGN v2 model export — ONNX and weight conversion utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import get_config
from .model import CGNv2
from .tokenizer import CGNv2Tokenizer


def export_weights(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str,
    model_size: str = "small",
):
    """Export model weights + tokenizer as a standalone package.

    Creates a directory with:
        model.pt       — model state_dict only (no optimizer)
        tokenizer.json — tokenizer config
        config.json    — model config as JSON
    """
    import json
    from dataclasses import asdict

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = CGNv2Tokenizer.load(tokenizer_path)
    config = get_config(model_size, vocab_size=tokenizer.vocab_size)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)

    # Save model weights only
    torch.save(state_dict, out / "model.pt")

    # Save config
    config_dict = asdict(config)
    (out / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Copy tokenizer
    import shutil
    shutil.copy2(tokenizer_path, out / "tokenizer.json")

    n_params = sum(p.numel() for p in state_dict.values())
    size_mb = sum(p.nbytes for p in state_dict.values()) / 1e6
    print(f"Exported to {out}/")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Model size: {size_mb:.1f} MB")
    print(f"  Files: model.pt, config.json, tokenizer.json")


def export_onnx(
    checkpoint_path: str,
    tokenizer_path: str,
    output_path: str,
    model_size: str = "small",
    max_seq_len: int = 512,
):
    """Export model to ONNX format.

    Note: Only exports the forward pass (single-step logit computation).
    KV-cache and generation loop must be handled externally.
    """
    tokenizer = CGNv2Tokenizer.load(tokenizer_path)
    config = get_config(model_size, vocab_size=tokenizer.vocab_size)

    model = CGNv2(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randint(0, config.vocab_size, (1, max_seq_len))

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )
    print(f"ONNX model saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CGN v2 model export")
    sub = parser.add_subparsers(dest="command", required=True)

    # Export weights
    weights = sub.add_parser("weights", help="Export standalone weight package")
    weights.add_argument("--checkpoint", type=str, required=True)
    weights.add_argument("--tokenizer_json", type=str, required=True)
    weights.add_argument("--output_dir", type=str, required=True)
    weights.add_argument("--model_size", type=str, default="small")

    # Export ONNX
    onnx = sub.add_parser("onnx", help="Export to ONNX")
    onnx.add_argument("--checkpoint", type=str, required=True)
    onnx.add_argument("--tokenizer_json", type=str, required=True)
    onnx.add_argument("--output", type=str, required=True)
    onnx.add_argument("--model_size", type=str, default="small")

    args = parser.parse_args()

    if args.command == "weights":
        export_weights(args.checkpoint, args.tokenizer_json, args.output_dir, args.model_size)
    elif args.command == "onnx":
        export_onnx(args.checkpoint, args.tokenizer_json, args.output, args.model_size)


if __name__ == "__main__":
    main()
