"""
codec_extract.py
=================

Extract discrete audio tokens from waveforms using a neural audio codec.

This script reads the JSON Lines file produced by `prepare_lj.py`, loads each
audio file, passes it through a neural codec to obtain a sequence of
discrete codes, and writes out a new JSON Lines file containing both the
phoneme ID sequence and the corresponding audio token sequence.

At present this implementation is a skeleton.  To make it functional you
should choose a suitable neural codec (e.g. [Encodec](https://github.com/facebookresearch/encodec)
or [SoundStream](https://arxiv.org/abs/2007.15408)) and implement the
`encode_audio` function accordingly.  The codec should produce a small
number of codebooks (each with a discrete vocabulary) per time step.  The
resulting codes can be flattened into a single sequence of integer tokens.

Usage:
    python codec_extract.py \
        --input_jsonl data/ljspeech_prepared/ljspeech_prepared.jsonl \
        --output_jsonl data/ljspeech_tokens/ljspeech_tokens.jsonl
"""

import argparse
import os, sys 
import json
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from xcodec2 import XCodec2Model
import soundfile as sf  # type: ignore

print(torch)
sys.exit()    

# Global model instance (loaded once)
_xcodec_model = None

def get_xcodec_model():
    """Load XCodec2 model (singleton pattern)."""
    global _xcodec_model
    if _xcodec_model is None:
        if XCodec2Model is None:
            raise ImportError("XCodec2 not installed. Please install xcodec2 library.")
        _xcodec_model = XCodec2Model.from_pretrained("xcodec2-24khz")
    return _xcodec_model


def encode_audio(waveform: np.ndarray, sample_rate: int) -> List[int]:
    """Encode a waveform into a sequence of discrete audio tokens using XCodec2.

    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio

    Returns:
        List of integer tokens representing the compressed audio signal
    """
    if torch is None:
        raise ImportError("PyTorch not installed")
    
    model = get_xcodec_model()
    
    # Convert to torch tensor and ensure correct shape and dtype
    if waveform.ndim == 1:
        # Add batch and channel dimensions: (samples,) -> (1, 1, samples)
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        # Add batch dimension: (channels, samples) -> (1, channels, samples)
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    else:
        # Assume it's already (batch, channels, samples)
        waveform_tensor = torch.from_numpy(waveform).float()
    
    # Ensure we have the right sample rate for XCodec2 (24kHz model)
    # Note: You may need to resample if your audio is not 24kHz
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    waveform_tensor = waveform_tensor.to(device)
    
    # Encode audio to discrete tokens
    with torch.no_grad():
        try:
            # XCodec2 encode method returns quantized codes
            encoded = model.encode(waveform_tensor)
            
            # Handle different return formats
            if isinstance(encoded, tuple):
                # Usually (codes, indices) or (codes, commitment_loss, entropy)
                codes = encoded[0]  # Get the codes tensor
            elif isinstance(encoded, dict):
                # Some implementations return a dictionary
                codes = encoded.get('codes', encoded.get('indices', encoded))
            else:
                codes = encoded
            
            # Ensure codes is a tensor
            if not isinstance(codes, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(codes)}")
            
            # XCodec2 typically returns shape (batch, n_codebooks, time)
            # Flatten across codebooks and time: (B, n_codebooks, T) -> (B * n_codebooks * T,)
            if codes.ndim == 3:
                # Standard codebook format
                audio_tokens = codes.flatten().cpu().numpy().astype(int).tolist()
            elif codes.ndim == 2:
                # Already flattened or single codebook
                audio_tokens = codes.flatten().cpu().numpy().astype(int).tolist()
            else:
                # Handle other dimensions
                audio_tokens = codes.flatten().cpu().numpy().astype(int).tolist()
                
        except Exception as e:
            raise RuntimeError(f"Failed to encode audio with XCodec2: {str(e)}")
    
    return audio_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio codec tokens for prefix‑LM TTS")
    parser.add_argument("--input_jsonl", type=Path, required=True, help="Path to JSONL file with phoneme sequences")
    parser.add_argument("--output_jsonl", type=Path, required=True, help="Path to write JSONL file with audio tokens added")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Expected sample rate of audio files")
    parser.add_argument("--progress", action="store_true", help="Show progress bar")
    args = parser.parse_args()

    if sf is None:
        raise ImportError("soundfile library not installed; please install to read WAV files")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Count total lines for progress tracking
    total_lines = 0
    with args.input_jsonl.open() as f:
        for _ in f:
            total_lines += 1

    print(f"Processing {total_lines} audio files...")
    
    processed = 0
    errors = 0
    
    with args.input_jsonl.open() as in_f, args.output_jsonl.open("w") as out_f:
        for line_no, line in enumerate(in_f, 1):
            try:
                example = json.loads(line.strip())
                audio_path = example["audio_path"]
                
                if not Path(audio_path).exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    errors += 1
                    continue
                
                # Load audio
                waveform, sr = sf.read(audio_path)
                if sr != args.sample_rate:
                    print(f"Warning: Sample rate mismatch for {audio_path}: expected {args.sample_rate}, got {sr}")
                    # You might want to resample here instead of raising an error
                    # For now, we'll continue with the actual sample rate
                
                # Extract audio tokens
                audio_tokens = encode_audio(waveform, sr)
                example["audio_tokens"] = audio_tokens
                
                # Write enhanced example
                out_f.write(json.dumps(example) + "\n")
                processed += 1
                
                # Progress reporting
                if args.progress and (processed % 100 == 0 or processed == total_lines):
                    print(f"Processed {processed}/{total_lines} files ({processed/total_lines*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing line {line_no}: {str(e)}")
                errors += 1
                continue
    
    print(f"Codec extraction completed!")
    print(f"  Successfully processed: {processed} files")
    print(f"  Errors: {errors} files")
    print(f"  Output written to: {args.output_jsonl}")
    
    if errors > 0:
        print(f"Warning: {errors} files had errors and were skipped")


if __name__ == "__main__":
    main()