"""CGN v2 interactive inference — load checkpoint, generate audio from text input.

Usage:
    python -m festvox3.models.cgn_v2.inference \
        --checkpoint_dir recipes/ljspeech_cgn_v2/work/checkpoints \
        --tokenizer_json recipes/ljspeech_cgn_v2/work/prepared/sequences/tokenizer.json

Interactive commands:
    <text>              Generate audio from text
    :temp <float>       Set temperature (default 0.8)
    :topk <int>         Set top-k (default 50)
    :topp <float>       Set top-p (default 0.9)
    :cfg <float>        Set CFG scale (default 1.5)
    :max <int>          Set max tokens (default 2048)
    :outdir <path>      Change output directory
    :settings           Show current settings
    :quit / :q          Exit
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch
import torchaudio

from .config import get_config
from .model import CGNv2
from .tokenizer import CGNv2Tokenizer, EOS
from .generate import synthesize, tokens_to_audio, extract_plan_tokens


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint in a directory.

    Priority:
      1. best.pt (if saved by training with val-loss tracking)
      2. latest.pt
      3. Highest-numbered checkpoint_*.pt
    """
    best = checkpoint_dir / "best.pt"
    if best.exists():
        return best

    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest

    # Fall back to highest step checkpoint
    ckpts = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if ckpts:
        return ckpts[-1]

    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")


def load_model(
    checkpoint_path: Path,
    tokenizer: CGNv2Tokenizer,
    model_size: str,
    device: str,
) -> tuple[CGNv2, dict]:
    """Load model from checkpoint. Returns (model, metadata)."""
    config = get_config(
        model_size,
        vocab_size=tokenizer.vocab_size,
        cfg_drop_prob=0.0,
    )
    model = CGNv2(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "epoch": ckpt.get("epoch", "?"),
        "global_step": ckpt.get("global_step", "?"),
        "n_params": model.count_parameters(),
        "config": f"{config.d_model}d {config.n_layers}L {config.n_heads}H",
    }
    return model, metadata


class InferenceSession:
    """Interactive TTS session with adjustable parameters."""

    def __init__(
        self,
        model: CGNv2,
        tokenizer: CGNv2Tokenizer,
        codec,
        device: str,
        output_dir: Path,
        flite_bin: str = "flite",
        codec_type: str = "snac",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.codec = codec
        self.device = device
        self.output_dir = output_dir
        self.flite_bin = flite_bin
        self.codec_type = codec_type

        # Generation parameters
        self.temperature = 0.8
        self.top_k = 50
        self.top_p = 0.9
        self.cfg_scale = 1.5
        self.max_new_tokens = 2048

        self._gen_count = 0

    def settings_str(self) -> str:
        return (
            f"  temperature: {self.temperature}\n"
            f"  top_k:       {self.top_k}\n"
            f"  top_p:       {self.top_p}\n"
            f"  cfg_scale:   {self.cfg_scale}\n"
            f"  max_tokens:  {self.max_new_tokens}\n"
            f"  output_dir:  {self.output_dir}"
        )

    def handle_command(self, cmd: str) -> bool:
        """Handle a : command. Returns True if should continue, False to quit."""
        parts = cmd.strip().split(None, 1)
        name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if name in (":quit", ":q"):
            return False
        elif name == ":temp" and arg:
            self.temperature = float(arg)
            print(f"  temperature = {self.temperature}")
        elif name == ":topk" and arg:
            self.top_k = int(arg)
            print(f"  top_k = {self.top_k}")
        elif name == ":topp" and arg:
            self.top_p = float(arg)
            print(f"  top_p = {self.top_p}")
        elif name == ":cfg" and arg:
            self.cfg_scale = float(arg)
            print(f"  cfg_scale = {self.cfg_scale}")
        elif name == ":max" and arg:
            self.max_new_tokens = int(arg)
            print(f"  max_tokens = {self.max_new_tokens}")
        elif name == ":outdir" and arg:
            self.output_dir = Path(arg)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"  output_dir = {self.output_dir}")
        elif name == ":settings":
            print(self.settings_str())
        else:
            print(f"  Unknown command: {name}")
        return True

    def generate(self, text: str) -> Path | None:
        """Generate audio from text. Returns output path or None on failure."""
        self._gen_count += 1
        # Build filename from sanitized text
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", text)[:40].strip("_").lower()
        out_path = self.output_dir / f"{self._gen_count:04d}_{safe}.wav"

        t0 = time.time()
        try:
            _, token_ids = synthesize(
                self.model, self.tokenizer, text,
                device=self.device,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                cfg_scale=self.cfg_scale,
                flite_bin=self.flite_bin,
            )
        except Exception as e:
            print(f"  Generation failed: {e}")
            return None

        gen_time = time.time() - t0

        # Extract plan
        plan = extract_plan_tokens(token_ids, self.tokenizer)
        n_audio = sum(1 for t in token_ids if self.tokenizer.is_audio_token(t))

        print(f"  Plan ({len(plan)} tokens): {' '.join(plan[:20])}"
              + ("..." if len(plan) > 20 else ""))
        print(f"  Generated {len(token_ids)} tokens ({n_audio} audio) in {gen_time:.1f}s")

        # Decode to audio
        t1 = time.time()
        try:
            waveform = tokens_to_audio(
                token_ids, self.tokenizer, self.codec,
                codec_type=self.codec_type,
            )
        except Exception as e:
            print(f"  Audio decode failed: {e}")
            return None

        decode_time = time.time() - t1
        duration = waveform.shape[-1] / 24000

        import soundfile as sf
        sf.write(str(out_path), waveform.squeeze().cpu().numpy(), 24000)
        print(f"  Audio: {duration:.2f}s, decode: {decode_time:.1f}s")
        print(f"  Saved: {out_path}")

        return out_path


def main():
    parser = argparse.ArgumentParser(
        description="CGN v2 interactive TTS inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint_dir", type=str,
                        help="Directory containing checkpoints (uses best/latest)")
    parser.add_argument("--checkpoint", type=str,
                        help="Specific checkpoint file (overrides --checkpoint_dir)")
    parser.add_argument("--tokenizer_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated audio")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "mini", "small", "base"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--flite_bin", type=str, default="flite")
    # Allow single-shot mode via --text
    parser.add_argument("--text", type=str, default=None,
                        help="Generate from this text and exit (non-interactive)")
    # Default generation params
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--codec", type=str, default="snac",
                        choices=["snac", "qwen12hz"],
                        help="Audio codec for decoding")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.checkpoint_dir:
        ckpt_path = find_best_checkpoint(Path(args.checkpoint_dir))
    else:
        parser.error("Provide --checkpoint or --checkpoint_dir")

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading tokenizer: {args.tokenizer_json}")
    tokenizer = CGNv2Tokenizer.load(args.tokenizer_json)

    print(f"Loading model: {ckpt_path}")
    model, meta = load_model(ckpt_path, tokenizer, args.model_size, args.device)
    print(f"  {meta['config']}, {meta['n_params']/1e6:.1f}M params, "
          f"step {meta['global_step']}, epoch {meta['epoch']}")

    if args.codec == "qwen12hz":
        from .qwen_codec import QwenCodec
        print(f"Loading Qwen 12Hz codec...")
        codec = QwenCodec(device=args.device)
    else:
        from .snac_codec import SNACCodec
        print(f"Loading SNAC codec...")
        codec = SNACCodec(device=args.device)

    session = InferenceSession(
        model, tokenizer, codec, args.device, output_dir, args.flite_bin,
        codec_type=args.codec,
    )
    session.temperature = args.temperature
    session.top_k = args.top_k
    session.top_p = args.top_p
    session.cfg_scale = args.cfg_scale
    session.max_new_tokens = args.max_new_tokens

    # Single-shot mode
    if args.text:
        session.generate(args.text)
        return

    # Interactive mode
    print(f"\nReady. Type text to synthesize, :settings to view params, :q to quit.\n")
    print(session.settings_str())
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        if text.startswith(":"):
            if not session.handle_command(text):
                break
            continue

        session.generate(text)
        print()


if __name__ == "__main__":
    main()
