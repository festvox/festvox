"""CGN v2 inference — text-to-speech with prosodic chain-of-thought."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import torch

from .config import get_config
from .model import CGNv2
from .tokenizer import (
    CGNv2Tokenizer, BOS, EOS, PLAN_START, AUDIO_START, UNK,
)


def _flite_g2p(text: str, flite_bin: str = "flite") -> list[str]:
    """Run Flite G2P to get phonemes."""
    result = subprocess.run(
        [flite_bin, "-t", text, "-o", "/dev/null", "-ps"],
        capture_output=True, text=True, timeout=30,
    )
    phones = result.stdout.strip().split()
    # Strip stress digits for clean phoneme tokens
    clean = []
    for p in phones:
        if p[-1:].isdigit():
            clean.append(p[:-1])
        else:
            clean.append(p)
    return clean


def synthesize(
    model: CGNv2,
    tokenizer: CGNv2Tokenizer,
    text: str,
    device: str = "cuda",
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 2048,
    cfg_scale: float = 1.0,
    flite_bin: str = "flite",
) -> tuple[torch.Tensor, list[int]]:
    """Full TTS pipeline: text -> phonemes -> model -> generated tokens.

    Returns:
        (generated_token_ids, raw_token_ids_list)
    """
    # Step 1: Text to phonemes
    phonemes = _flite_g2p(text, flite_bin)
    if not phonemes:
        raise ValueError(f"Flite G2P returned no phonemes for: {text!r}")

    # Step 2: Encode linguistic prefix
    ling_ids = tokenizer.encode_phonemes(phonemes)

    # Step 3: Build prompt = [BOS] + ling_ids + [PLAN_START]
    # The model will generate: prosodic_plan + [AUDIO_START] + audio_tokens + [EOS]
    prompt = [BOS] + ling_ids + [PLAN_START]
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

    # Step 4: Build unconditional prompt for CFG (linguistic tokens replaced with UNK)
    uncond_input_ids = None
    if cfg_scale > 1.0:
        uncond_prompt = [BOS] + [UNK] * len(ling_ids) + [PLAN_START]
        uncond_input_ids = torch.tensor([uncond_prompt], dtype=torch.long, device=device)

    # Step 5: Generate
    stop_tokens = {EOS}
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        stop_tokens=stop_tokens,
        cfg_scale=cfg_scale,
        uncond_input_ids=uncond_input_ids,
    )

    return generated, generated[0].tolist()


def tokens_to_audio(
    token_ids: list[int],
    tokenizer: CGNv2Tokenizer,
    codec,
    codec_type: str = "snac",
) -> torch.Tensor:
    """Extract audio tokens from generated sequence and decode to waveform.

    Supports both SNAC (3-level interleaved) and Qwen 12Hz (single codebook).

    Returns:
        [1, 1, T] waveform tensor at 24kHz (SNAC)
        or [1, T] waveform tensor at 24kHz (Qwen, converted from numpy)
    """
    # Find AUDIO_START marker
    try:
        audio_start_idx = token_ids.index(AUDIO_START)
    except ValueError:
        raise ValueError("Generated sequence does not contain AUDIO_START token")

    # Extract audio tokens (everything after AUDIO_START, before EOS)
    audio_token_ids = []
    for tid in token_ids[audio_start_idx + 1:]:
        if tid == EOS:
            break
        if tokenizer.is_audio_token(tid):
            audio_token_ids.append(tid)

    if not audio_token_ids:
        raise ValueError("No audio tokens found in generated sequence")

    # Decode to (level, code) pairs
    flat_codes = [tokenizer.decode_snac(tid) for tid in audio_token_ids]

    if codec_type == "qwen12hz":
        # Qwen 12Hz: repack interleaved codes into (n_frames, n_codebooks) array
        import numpy as np
        n_levels = tokenizer.snac_n_levels
        if n_levels == 1:
            semantic_codes = [code for _level, code in flat_codes]
            waveform_np = codec.decode_semantic(semantic_codes)
        else:
            # Multi-codebook: group by frame
            n_frames = len(flat_codes) // n_levels
            codes = np.zeros((n_frames, 16), dtype=np.int16)
            for i in range(n_frames):
                for lvl in range(n_levels):
                    idx = i * n_levels + lvl
                    if idx < len(flat_codes):
                        _, code = flat_codes[idx]
                        codes[i, lvl] = code
            waveform_np = codec.decode(codes)
        return torch.from_numpy(waveform_np).float().unsqueeze(0)
    else:
        # SNAC: repack into valid frames (1 coarse + 2 mid + 4 fine = 7 per frame)
        flat_codes = _repack_snac_frames(flat_codes)
        if not flat_codes:
            raise ValueError("Could not form any valid SNAC frames from generated tokens")
        return codec.decode(flat_codes)


def _repack_snac_frames(flat_codes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Repack (level, code) pairs into valid SNAC frames.

    SNAC expects interleaved pattern: [c, m, m, f, f, f, f] per frame.
    If model output follows this pattern, pass through directly.
    Otherwise, separate by level and re-interleave.
    """
    # Check if already correctly interleaved
    if _is_valid_interleaving(flat_codes):
        return flat_codes

    # Separate by level
    coarse, mid, fine = [], [], []
    for level, code in flat_codes:
        if level == 0:
            coarse.append(code)
        elif level == 1:
            mid.append(code)
        elif level == 2:
            fine.append(code)

    # Determine how many complete frames we can form
    n_frames = min(
        len(coarse),
        len(mid) // 2,
        len(fine) // 4,
    )

    if n_frames == 0:
        return []

    # Re-interleave into valid pattern
    repacked = []
    for i in range(n_frames):
        repacked.append((0, coarse[i]))
        repacked.append((1, mid[2 * i]))
        repacked.append((1, mid[2 * i + 1]))
        repacked.append((2, fine[4 * i]))
        repacked.append((2, fine[4 * i + 1]))
        repacked.append((2, fine[4 * i + 2]))
        repacked.append((2, fine[4 * i + 3]))

    return repacked


def _is_valid_interleaving(flat_codes: list[tuple[int, int]]) -> bool:
    """Check if codes follow the expected [0,1,1,2,2,2,2] pattern."""
    if len(flat_codes) % 7 != 0:
        return False
    pattern = [0, 1, 1, 2, 2, 2, 2]
    for i, (level, _) in enumerate(flat_codes):
        if level != pattern[i % 7]:
            return False
    return True


def extract_plan_tokens(
    token_ids: list[int],
    tokenizer: CGNv2Tokenizer,
) -> list[str]:
    """Extract the prosodic plan from generated tokens (for inspection)."""
    try:
        plan_start = token_ids.index(PLAN_START)
    except ValueError:
        return []

    try:
        audio_start = token_ids.index(AUDIO_START)
    except ValueError:
        audio_start = len(token_ids)

    plan_ids = token_ids[plan_start + 1 : audio_start]
    plan_tokens = []
    for tid in plan_ids:
        if tokenizer.is_prosody_token(tid):
            name = tokenizer._id2pros.get(tid, f"<unk:{tid}>")
            plan_tokens.append(name)
    return plan_tokens


def main():
    parser = argparse.ArgumentParser(description="CGN v2 text-to-speech")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--tokenizer_json", type=str, required=True, help="Tokenizer JSON path")
    parser.add_argument("--output", type=str, default="output.wav", help="Output wav path")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (1.0 = no guidance)")
    parser.add_argument("--flite_bin", type=str, default="flite")
    parser.add_argument("--show_plan", action="store_true",
                        help="Print the generated prosodic plan")
    parser.add_argument("--codec", type=str, default="snac",
                        choices=["snac", "qwen12hz"],
                        help="Audio codec for decoding (snac or qwen12hz)")
    args = parser.parse_args()

    device = args.device

    # Load tokenizer
    tokenizer = CGNv2Tokenizer.load(args.tokenizer_json)

    # Load model
    config = get_config(
        args.model_size,
        vocab_size=tokenizer.vocab_size,
        cfg_drop_prob=0.0,  # No CFG dropout at inference
    )
    model = CGNv2(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded model: {config.d_model}d, {config.n_layers}L, "
          f"{model.count_parameters()/1e6:.1f}M params")

    # Synthesize
    print(f"Synthesizing: {args.text!r}")
    _, token_ids = synthesize(
        model, tokenizer, args.text,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        flite_bin=args.flite_bin,
    )

    # Show prosodic plan if requested
    if args.show_plan:
        plan = extract_plan_tokens(token_ids, tokenizer)
        print(f"Prosodic plan ({len(plan)} tokens):")
        print("  " + " ".join(plan))

    # Count audio tokens
    n_audio = sum(1 for t in token_ids if tokenizer.is_audio_token(t))
    print(f"Generated {len(token_ids)} tokens ({n_audio} audio)")

    # Decode to audio
    if args.codec == "qwen12hz":
        from .qwen_codec import QwenCodec
        codec = QwenCodec(device=device)
    else:
        from .snac_codec import SNACCodec
        codec = SNACCodec(device=device)
    waveform = tokens_to_audio(token_ids, tokenizer, codec, codec_type=args.codec)

    # Save
    audio_np = waveform.squeeze().cpu().numpy()
    duration = len(audio_np) / 24000
    import soundfile as sf
    sf.write(args.output, audio_np, 24000)
    print(f"Saved: {args.output} ({duration:.2f}s)")


if __name__ == "__main__":
    main()
