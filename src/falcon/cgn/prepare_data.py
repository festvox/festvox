"""CGN v2 data preparation pipeline.

8-stage pipeline:
  1. Extract Flite phonemes + stress
  2. Extract codec tokens — SNAC (84 tok/s) or Qwen 12Hz (12.5 Hz, GPU)
  3. Whisper forced alignment (word boundaries)
  4. Extract prosodic features (F0, energy, pauses)
  5. Build prosodic plan tokens (uses Whisper boundaries if available)
  6. Split data + create augmented WAVs + run feature extraction on augmented
  7. Build tokenizer + JSONL sequences (train/val/test with text field)

Each stage is idempotent (skip if output already exists).
Codec selection: --codec snac (default) or --codec qwen12hz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import CGNv2Tokenizer


# ── Stage 1: Flite phonemes ────────────────────────────────────────────────

def _run_flite_g2p(text: str, flite_bin: str = "flite") -> dict:
    """Run Flite G2P, return phonemes and stress info."""
    try:
        result = subprocess.run(
            [flite_bin, "-t", text, "-o", "/dev/null", "-ps"],
            capture_output=True, text=True, timeout=30,
        )
        phones_str = result.stdout.strip()
        if not phones_str:
            return {"phonemes": [], "stress": []}

        phones = phones_str.split()

        # Extract stress from vowel phonemes (Flite convention: 0/1/2 suffix)
        clean_phones = []
        stress = []
        for p in phones:
            if p == "pau":
                clean_phones.append(p)
                stress.append(0)
            elif p[-1].isdigit():
                clean_phones.append(p[:-1])
                stress.append(int(p[-1]))
            else:
                clean_phones.append(p)
                stress.append(0)

        return {"phonemes": clean_phones, "stress": stress}
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: Flite G2P failed for '{text[:50]}...': {e}")
        return {"phonemes": [], "stress": []}


def stage1_extract_phonemes(
    metadata_csv: Path,
    output_dir: Path,
    flite_bin: str = "flite",
    dataset_format: str = "ljspeech",
):
    """Extract phonemes and stress from text using Flite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = _read_metadata(metadata_csv, dataset_format)

    done, skipped = 0, 0
    for utt_id, text in tqdm(utterances, desc="Stage 1: Phonemes"):
        out_file = output_dir / f"{utt_id}.json"
        if out_file.exists():
            skipped += 1
            continue

        result = _run_flite_g2p(text, flite_bin)
        if result["phonemes"]:
            out_file.write_text(json.dumps(result))
            done += 1

    print(f"Stage 1 done: {done} new, {skipped} skipped")


# ── Stage 2: SNAC codec tokens ────────────────────────────────────────────

def stage2_extract_snac(
    wav_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
    device: str = "cuda",
):
    """Extract SNAC codec tokens from audio files."""
    from .snac_codec import SNACCodec

    output_dir.mkdir(parents=True, exist_ok=True)
    codec = SNACCodec(device=device)
    utterances = _read_metadata(metadata_csv, dataset_format)

    done, skipped, failed = 0, 0, 0
    for utt_id, _text in tqdm(utterances, desc="Stage 2: SNAC tokens"):
        out_file = output_dir / f"{utt_id}.npy"
        if out_file.exists():
            skipped += 1
            continue

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            failed += 1
            continue

        try:
            flat_codes = codec.encode_file(str(wav_path))
            codec.save_codes(flat_codes, out_file)
            done += 1
        except Exception as e:
            print(f"  Failed {utt_id}: {e}")
            failed += 1

    print(f"Stage 2 done: {done} new, {skipped} skipped, {failed} failed")


# ── Stage 2 (Qwen 12Hz variant): Qwen codec tokens ──────────────────────

def stage2_extract_qwen(
    wav_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
    device: str = "cuda",
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Extract Qwen 12Hz codec tokens from audio files.

    Saves (n_frames, 16) int16 arrays. All 16 codebooks are stored
    for future MTP integration; only codebook 0 is used in phase 4a.
    """
    from .qwen_codec import QwenCodec

    output_dir.mkdir(parents=True, exist_ok=True)
    codec = QwenCodec(device=device)
    utterances = _read_metadata(metadata_csv, dataset_format)

    # Shard: each worker takes every num_shards-th item
    if num_shards > 1:
        utterances = utterances[shard_id::num_shards]

    shard_desc = f" [shard {shard_id}]" if num_shards > 1 else ""
    done, skipped, failed = 0, 0, 0
    for utt_id, _text in tqdm(utterances, desc=f"Stage 2{shard_desc}: Qwen 12Hz tokens"):
        out_file = output_dir / f"{utt_id}.npy"
        if out_file.exists():
            skipped += 1
            continue

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            failed += 1
            continue

        try:
            codes = codec.encode_file(str(wav_path))  # (n_frames, 16)
            QwenCodec.save_codes(codes, out_file)
            done += 1
        except Exception as e:
            print(f"  Failed {utt_id}: {e}")
            failed += 1

    print(f"Stage 2{shard_desc} done: {done} new, {skipped} skipped, {failed} failed")


def _codec_dir(output_dir: Path, codec_type: str) -> Path:
    """Return the codec token directory for the given codec type."""
    if codec_type == "qwen12hz":
        return output_dir / "qwen_tokens"
    return output_dir / "snac_tokens"


# ── Stage 3: Whisper forced alignment ──────────────────────────────────────

def stage3_align_whisper(
    wav_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
    whisper_model: str = "base.en",
    device: str = "cuda",
    batch_size: int = 16,
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Extract word-level boundaries using faster-whisper with word timestamps.

    Saves per-utterance JSON files at output_dir/{utt_id}.json:
        {"words": [{"word": str, "start": float, "end": float}, ...]}

    For multi-GPU parallel runs, use shard_id/num_shards to split work:
        GPU 0: --shard_id 0 --num_shards 2
        GPU 1: --shard_id 1 --num_shards 2
    """
    from faster_whisper import WhisperModel

    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = _read_metadata(metadata_csv, dataset_format)

    # Filter to only utterances that need processing
    to_process = []
    skipped = 0
    for utt_id, text in utterances:
        out_file = output_dir / f"{utt_id}.json"
        if out_file.exists():
            skipped += 1
        else:
            wav_path = wav_dir / f"{utt_id}.wav"
            if wav_path.exists():
                to_process.append((utt_id, text, wav_path))

    # Shard: each worker takes every num_shards-th item
    if num_shards > 1:
        to_process = to_process[shard_id::num_shards]

    if not to_process:
        print(f"Stage 3 [shard {shard_id}/{num_shards}]: All done, skipping")
        return

    print(f"Stage 3 [shard {shard_id}/{num_shards}]: {len(to_process)} to align, {skipped} already done")

    # Load Whisper model — faster-whisper wants "cuda"/"cpu", not "cuda:0"
    whisper_device = "cuda" if "cuda" in device else "cpu"
    device_index = 0
    if ":" in device:
        device_index = int(device.split(":")[1])
    compute_type = "float32" if whisper_device == "cuda" else "int8"
    model = WhisperModel(
        whisper_model, device=whisper_device,
        device_index=device_index, compute_type=compute_type,
    )

    done, failed = 0, 0
    shard_desc = f" [shard {shard_id}]" if num_shards > 1 else ""
    for utt_id, text, wav_path in tqdm(to_process, desc=f"Stage 3{shard_desc}: Whisper alignment"):
        out_file = output_dir / f"{utt_id}.json"

        try:
            segments, _info = model.transcribe(
                str(wav_path),
                language="en",
                word_timestamps=True,
                beam_size=1,
                vad_filter=True,
            )

            words = []
            for segment in segments:
                if segment.words:
                    for w in segment.words:
                        words.append({
                            "word": w.word.strip(),
                            "start": round(w.start, 4),
                            "end": round(w.end, 4),
                        })

            if words:
                out_file.write_text(json.dumps({"words": words}))
                done += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Failed {utt_id}: {e}")
            failed += 1

    print(f"Stage 3 done: {done} new, {skipped} skipped, {failed} failed")


# ── Stage 4: Prosodic features ────────────────────────────────────────────

def stage4_extract_prosody(
    wav_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
):
    """Extract F0, energy, and detect pauses."""
    from .prosody import extract_f0, extract_energy, detect_pauses

    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = _read_metadata(metadata_csv, dataset_format)

    done, skipped, failed = 0, 0, 0
    for utt_id, text in tqdm(utterances, desc="Stage 4: Prosody features"):
        out_file = output_dir / f"{utt_id}.json"
        if out_file.exists():
            skipped += 1
            continue

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            failed += 1
            continue

        try:
            f0, times = extract_f0(str(wav_path))
            energy = extract_energy(str(wav_path))
            pauses = detect_pauses(energy)

            out_file.write_text(json.dumps({
                "f0_mean": float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
                "f0_std": float(np.std(f0[f0 > 0])) if np.sum(f0 > 0) > 1 else 0.0,
                "energy_mean": float(np.mean(energy)),
                "energy_std": float(np.std(energy)),
                "duration_sec": float(len(f0) * 0.01),  # 10ms frames
                "n_pauses": len(pauses),
                "text": text,
            }))
            done += 1
        except Exception as e:
            print(f"  Failed {utt_id}: {e}")
            failed += 1

    print(f"Stage 4 done: {done} new, {skipped} skipped, {failed} failed")


# ── Stage 5: Build prosodic plan tokens ────────────────────────────────────

def stage5_build_plans(
    wav_dir: Path,
    prosody_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
    word_boundaries_dir: Path | None = None,
):
    """Build prosodic plan token sequences from raw features.

    If word_boundaries_dir is provided and contains a JSON for the utterance,
    those Whisper-derived boundaries are used instead of uniform word detection.
    """
    from .prosody import extract_prosodic_plan

    output_dir.mkdir(parents=True, exist_ok=True)
    utterances = _read_metadata(metadata_csv, dataset_format)

    done, skipped, failed = 0, 0, 0
    aligned_used, fallback_used = 0, 0

    for utt_id, text in tqdm(utterances, desc="Stage 5: Prosodic plans"):
        out_file = output_dir / f"{utt_id}.json"
        if out_file.exists():
            skipped += 1
            continue

        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            failed += 1
            continue

        # Load word boundaries if available (from Whisper alignment)
        word_boundaries = None
        if word_boundaries_dir is not None:
            wb_file = word_boundaries_dir / f"{utt_id}.json"
            if wb_file.exists():
                try:
                    wb_data = json.loads(wb_file.read_text())
                    word_boundaries = wb_data["words"]
                    aligned_used += 1
                except (json.JSONDecodeError, KeyError):
                    fallback_used += 1
            else:
                fallback_used += 1

        try:
            plan_tokens = extract_prosodic_plan(
                str(wav_path), text, word_boundaries=word_boundaries,
            )
            out_file.write_text(json.dumps({"plan": plan_tokens}))
            done += 1
        except Exception as e:
            print(f"  Failed {utt_id}: {e}")
            failed += 1

    print(f"Stage 5 done: {done} new, {skipped} skipped, {failed} failed")
    if word_boundaries_dir is not None:
        print(f"  Whisper boundaries used: {aligned_used}, fallback (uniform): {fallback_used}")


# ── Stage 6: Split + Augment ──────────────────────────────────────────────

def _scale_word_boundaries(
    word_boundaries: list[dict], speed_factor: float
) -> list[dict]:
    """Scale word boundary timings for speed-perturbed audio.

    When audio is sped up by `speed_factor`, the duration changes by 1/speed_factor.
    E.g., 1.1x speed → times shrink by 1/1.1 ≈ 0.909.
    """
    scale = 1.0 / speed_factor
    return [
        {
            "word": wb["word"],
            "start": wb["start"] * scale,
            "end": wb["end"] * scale,
        }
        for wb in word_boundaries
    ]


def stage6_split_augment(
    metadata_csv: Path,
    wav_dir: Path,
    output_dir: Path,
    dataset_format: str = "ljspeech",
    val_split: float = 0.05,
    test_split: float = 0.05,
    speed_factors: list[float] | None = None,
    seed: int = 42,
    device: str = "cuda",
    codec_type: str = "snac",
):
    """Split data into train/val/test and create augmented training data.

    Creates speed-perturbed WAVs for training data only, then runs
    codec/prosody/plan extraction on augmented files. Also generates
    scaled word boundaries for augmented WAVs from the original Whisper output.
    """
    if speed_factors is None:
        speed_factors = [0.9, 1.1]

    import torchaudio

    # Read metadata for text mapping
    utterances = _read_metadata(metadata_csv, dataset_format)
    utt_text = {uid: text for uid, text in utterances}

    # Find utterances with all features from stages 1-5
    phoneme_dir = output_dir / "phonemes"
    codec_tok_dir = _codec_dir(output_dir, codec_type)
    plan_dir = output_dir / "prosody_plans"
    prosody_dir = output_dir / "prosody_raw"
    wb_dir = output_dir / "word_boundaries"

    phone_ids = {f.stem for f in phoneme_dir.glob("*.json")}
    snac_ids = {f.stem for f in codec_tok_dir.glob("*.npy")}
    plan_ids = {f.stem for f in plan_dir.glob("*.json")}
    # Only consider original (non-augmented) IDs
    orig_ids = phone_ids & snac_ids & plan_ids
    orig_ids = {uid for uid in orig_ids if "_sp" not in uid}
    common = sorted(orig_ids & set(utt_text.keys()))
    print(f"Found {len(common)} complete original utterances")

    # Deterministic split
    rng = random.Random(seed)
    shuffled = list(common)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_split))
    n_val = max(1, int(len(shuffled) * val_split))

    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test:n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Create augmented WAVs
    aug_wav_dir = output_dir / "augmented_wavs"
    aug_wav_dir.mkdir(parents=True, exist_ok=True)

    aug_mapping = {}  # aug_id → orig_id
    done, skipped = 0, 0

    for utt_id in tqdm(train_ids, desc="Creating augmented WAVs"):
        for sp in speed_factors:
            sp_str = str(sp).replace(".", "")
            aug_id = f"{utt_id}_sp{sp_str}"
            aug_wav_path = aug_wav_dir / f"{aug_id}.wav"

            aug_mapping[aug_id] = utt_id

            if aug_wav_path.exists():
                skipped += 1
                continue

            orig_wav = wav_dir / f"{utt_id}.wav"
            if not orig_wav.exists():
                del aug_mapping[aug_id]
                continue

            try:
                waveform, sr = torchaudio.load(str(orig_wav))
                perturbed = torchaudio.functional.resample(
                    waveform,
                    orig_freq=int(sr * sp),
                    new_freq=sr,
                )
                torchaudio.save(str(aug_wav_path), perturbed, sr)
                done += 1
            except Exception as e:
                print(f"  Failed augmenting {utt_id} sp={sp}: {e}")
                del aug_mapping[aug_id]

    print(f"Augmented WAVs: {done} new, {skipped} skipped, {len(aug_mapping)} total")

    # Copy phoneme files for augmented IDs (same text = same phonemes)
    copied = 0
    for aug_id, orig_id in tqdm(aug_mapping.items(), desc="Copying phonemes"):
        dst = phoneme_dir / f"{aug_id}.json"
        if not dst.exists():
            src = phoneme_dir / f"{orig_id}.json"
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
    print(f"Copied {copied} phoneme files for augmented data")

    # Generate scaled word boundaries for augmented WAVs
    if wb_dir.exists():
        wb_scaled = 0
        for aug_id, orig_id in tqdm(aug_mapping.items(), desc="Scaling word boundaries"):
            dst = wb_dir / f"{aug_id}.json"
            if dst.exists():
                continue

            src = wb_dir / f"{orig_id}.json"
            if not src.exists():
                continue

            try:
                wb_data = json.loads(src.read_text())
                # Extract speed factor from aug_id: e.g. "LJ001-0001_sp09" → 0.9
                sp_str = aug_id.rsplit("_sp", 1)[1]
                sp_factor = float(sp_str[0] + "." + sp_str[1:])

                scaled_words = _scale_word_boundaries(wb_data["words"], sp_factor)
                dst.write_text(json.dumps({"words": scaled_words}))
                wb_scaled += 1
            except Exception as e:
                print(f"  Failed scaling word boundaries for {aug_id}: {e}")

        print(f"Scaled {wb_scaled} word boundary files for augmented data")

    # Write augmented metadata CSV for stages 2-5
    aug_meta_path = output_dir / "augmented_metadata.csv"
    with open(aug_meta_path, "w") as f:
        for aug_id, orig_id in sorted(aug_mapping.items()):
            text = utt_text.get(orig_id, "")
            f.write(f"{aug_id}|{text}|{text}\n")

    # Run codec extraction on augmented WAVs
    if codec_type == "qwen12hz":
        print("\n--- Running Qwen 12Hz on augmented data ---")
        stage2_extract_qwen(aug_wav_dir, codec_tok_dir, aug_meta_path, "ljspeech", device)
    else:
        print("\n--- Running SNAC on augmented data ---")
        stage2_extract_snac(aug_wav_dir, codec_tok_dir, aug_meta_path, "ljspeech", device)

    # Run prosody extraction on augmented WAVs
    print("\n--- Running prosody on augmented data ---")
    stage4_extract_prosody(aug_wav_dir, prosody_dir, aug_meta_path, "ljspeech")

    # Run plan building on augmented WAVs (with word boundaries if available)
    print("\n--- Building plans for augmented data ---")
    stage5_build_plans(
        aug_wav_dir, prosody_dir, plan_dir, aug_meta_path, "ljspeech",
        word_boundaries_dir=wb_dir if wb_dir.exists() else None,
    )

    # Save split info
    split_info = {
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "augmented": aug_mapping,
        "speed_factors": speed_factors,
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split_info))
    print(f"Split info saved: {split_path}")


# ── Stage 7: Build tokenizer + JSONL ──────────────────────────────────────

def stage7_build_sequences(
    phoneme_dir: Path,
    codec_dir: Path,
    plan_dir: Path,
    output_dir: Path,
    metadata_csv: Path,
    dataset_format: str = "ljspeech",
    max_seq_len: int = 4096,
    split_json: Path | None = None,
    codec_type: str = "snac",
    n_codebooks: int = 4,
):
    """Build tokenizer + train/val/test JSONL sequences using split info.

    Each JSONL line includes: input_ids, labels, text, utt_id
    codec_type controls tokenizer params and code loading:
      - "snac": 3 levels * 4096 codebook, flat (level, code) pairs
      - "qwen12hz": n_codebooks levels * 2048 codebook, interleaved
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read text mapping from original metadata
    utterances = _read_metadata(metadata_csv, dataset_format)
    utt_text = {uid: text for uid, text in utterances}

    # Collect unique phonemes from ALL available phoneme files
    # (only from original IDs to keep tokenizer consistent)
    all_phones = set()
    phone_files = sorted(phoneme_dir.glob("*.json"))
    for f in phone_files:
        if "_sp" not in f.stem:  # Only originals for tokenizer building
            data = json.loads(f.read_text())
            all_phones.update(data["phonemes"])

    phoneme_list = sorted(all_phones)
    print(f"Found {len(phoneme_list)} unique phonemes")

    # Build tokenizer — codec-specific params
    if codec_type == "qwen12hz":
        tokenizer = CGNv2Tokenizer(
            phoneme_list, snac_codebook_size=2048, snac_n_levels=n_codebooks,
        )
    else:
        tokenizer = CGNv2Tokenizer(phoneme_list)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved: vocab_size={tokenizer.vocab_size} (codec={codec_type})")

    # Load split info
    if split_json and split_json.exists():
        split_info = json.loads(split_json.read_text())
        train_ids = split_info["train_ids"]
        val_ids = split_info["val_ids"]
        test_ids = split_info.get("test_ids", [])
        aug_mapping = split_info.get("augmented", {})

        # Add augmented text mapping
        for aug_id, orig_id in aug_mapping.items():
            utt_text[aug_id] = utt_text.get(orig_id, "")

        # Find augmented IDs that have all features
        aug_ids_with_features = []
        for aug_id in sorted(aug_mapping.keys()):
            if ((phoneme_dir / f"{aug_id}.json").exists()
                    and (codec_dir / f"{aug_id}.npy").exists()
                    and (plan_dir / f"{aug_id}.json").exists()):
                aug_ids_with_features.append(aug_id)

        all_train_ids = train_ids + aug_ids_with_features
        print(f"Train: {len(train_ids)} orig + {len(aug_ids_with_features)} aug "
              f"= {len(all_train_ids)} total")

        splits_to_build = [
            ("train", all_train_ids, output_dir / "data.train.jsonl"),
            ("val", val_ids, output_dir / "data.val.jsonl"),
            ("test", test_ids, output_dir / "data.test.jsonl"),
        ]
    else:
        # Fallback: old behavior (all data, 95/5 split, no test set)
        phone_ids = {f.stem for f in phoneme_dir.glob("*.json")}
        codec_ids = {f.stem for f in codec_dir.glob("*.npy")}
        plan_ids_set = {f.stem for f in plan_dir.glob("*.json")}
        common = sorted(phone_ids & codec_ids & plan_ids_set)

        n_val = max(1, int(len(common) * 0.05))
        val_set = set(common[:n_val])

        splits_to_build = [
            ("train", [u for u in common if u not in val_set],
             output_dir / "data.train.jsonl"),
            ("val", [u for u in common if u in val_set],
             output_dir / "data.val.jsonl"),
        ]

    stats = {}
    for split_name, utt_ids, out_path in splits_to_build:
        stats[split_name] = 0
        stats[f"{split_name}_too_long"] = 0

        with open(out_path, "w") as fout:
            for utt_id in tqdm(utt_ids, desc=f"Building {split_name}"):
                phone_file = phoneme_dir / f"{utt_id}.json"
                codec_file = codec_dir / f"{utt_id}.npy"
                plan_file = plan_dir / f"{utt_id}.json"

                if not (phone_file.exists() and codec_file.exists()
                        and plan_file.exists()):
                    continue

                # Load phonemes
                phone_data = json.loads(phone_file.read_text())
                ling_ids = tokenizer.encode_phonemes(phone_data["phonemes"])

                # Load prosodic plan
                plan_data = json.loads(plan_file.read_text())
                pros_ids = tokenizer.encode_prosody_seq(plan_data["plan"])

                # Load codec codes
                if codec_type == "qwen12hz":
                    flat_codes = QwenCodec_load_codes(codec_file, n_codebooks=n_codebooks)
                else:
                    flat_codes = SNACCodec_load_codes(codec_file)
                audio_ids = tokenizer.encode_snac_flat(flat_codes)

                # Build sequence
                seq = tokenizer.build_training_sequence(
                    ling_ids, pros_ids, audio_ids)

                if len(seq["input_ids"]) > max_seq_len:
                    stats[f"{split_name}_too_long"] += 1
                    continue

                # Get text for this utterance
                text = utt_text.get(utt_id, "")
                if not text and "_sp" in utt_id:
                    orig_id = utt_id.rsplit("_sp", 1)[0]
                    text = utt_text.get(orig_id, "")

                fout.write(json.dumps({
                    "input_ids": seq["input_ids"],
                    "labels": seq["labels"],
                    "text": text,
                    "utt_id": utt_id,
                }) + "\n")
                stats[split_name] += 1

    summary = ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
    print(f"Stage 7 done: {summary}")


def SNACCodec_load_codes(path: Path) -> list[tuple[int, int]]:
    """Load flat SNAC codes from numpy file."""
    arr = np.load(str(path))
    return [(int(row[0]), int(row[1])) for row in arr]


def QwenCodec_load_codes(path: Path, n_codebooks: int = 1) -> list[tuple[int, int]]:
    """Load Qwen 12Hz codes and return as flat interleaved (level, code) pairs.

    Loads (n_frames, 16) array, interleaves first n_codebooks per frame.
    For n_codebooks=4, each frame produces: [(0,c0), (1,c1), (2,c2), (3,c3)]
    Returns list of (level, code) tuples for use with tokenizer.encode_snac_flat().
    """
    arr = np.load(str(path))
    if n_codebooks == 1:
        return [(0, int(c)) for c in arr[:, 0]]
    flat = []
    for frame in arr:
        for lvl in range(n_codebooks):
            flat.append((lvl, int(frame[lvl])))
    return flat


# ── Metadata reader ───────────────────────────────────────────────────────

def _read_metadata(
    metadata_csv: Path, dataset_format: str = "ljspeech"
) -> list[tuple[str, str]]:
    """Read metadata CSV → list of (utt_id, text) tuples."""
    utterances = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        if dataset_format == "ljspeech":
            reader = csv.reader(f, delimiter="|")
            for row in reader:
                if len(row) >= 2:
                    utt_id = row[0].strip()
                    text = row[-1].strip()
                    utterances.append((utt_id, text))
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    utterances.append((row[0].strip(), row[1].strip()))
    return utterances


def _split_only(
    metadata_csv: Path,
    output_dir: Path,
    dataset_format: str = "ljspeech",
    val_split: float = 0.05,
    test_split: float = 0.05,
    seed: int = 42,
    codec_type: str = "snac",
):
    """Create train/val/test split without augmentation."""
    utterances = _read_metadata(metadata_csv, dataset_format)
    utt_text = {uid: text for uid, text in utterances}

    phoneme_dir = output_dir / "phonemes"
    codec_tok_dir = _codec_dir(output_dir, codec_type)
    plan_dir = output_dir / "prosody_plans"

    phone_ids = {f.stem for f in phoneme_dir.glob("*.json")}
    snac_ids = {f.stem for f in codec_tok_dir.glob("*.npy")}
    plan_ids = {f.stem for f in plan_dir.glob("*.json")}
    orig_ids = phone_ids & snac_ids & plan_ids
    orig_ids = {uid for uid in orig_ids if "_sp" not in uid}
    common = sorted(orig_ids & set(utt_text.keys()))
    print(f"Found {len(common)} complete utterances")

    rng = random.Random(seed)
    shuffled = list(common)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_split))
    n_val = max(1, int(len(shuffled) * val_split))

    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test:n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val:])

    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    split_info = {
        "seed": seed,
        "val_split": val_split,
        "test_split": test_split,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "augmented": {},
        "speed_factors": [],
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split_info))
    print(f"Split info saved: {split_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CGN v2 data preparation")
    parser.add_argument("--stage", type=int, default=-1,
                        help="Run specific stage (1-7). -1 = all.")
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, default="ljspeech")
    parser.add_argument("--flite_bin", type=str, default="flite")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--test_split", type=float, default=0.05)
    parser.add_argument("--augment_speeds", type=str, default="0.9,1.1",
                        help="Comma-separated speed factors for augmentation")
    parser.add_argument("--no_augment", action="store_true",
                        help="Skip augmentation (stages 1-5 + 7 only)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper_model", type=str, default="base.en",
                        help="Whisper model size for alignment (e.g. tiny.en, base.en, small.en)")
    parser.add_argument("--align_batch_size", type=int, default=16,
                        help="Batch size for Whisper alignment")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for parallel alignment (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel alignment")
    parser.add_argument("--codec", type=str, default="snac",
                        choices=["snac", "qwen12hz"],
                        help="Audio codec: snac (84 tok/s) or qwen12hz (12.5 Hz)")
    parser.add_argument("--n_codebooks", type=int, default=4,
                        help="Number of Qwen codebooks to interleave (1-16, default 4)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    wav_dir = Path(args.wav_dir)
    metadata = Path(args.metadata_csv)
    codec_type = args.codec

    phoneme_dir = out / "phonemes"
    codec_dir = _codec_dir(out, codec_type)
    wb_dir = out / "word_boundaries"
    prosody_dir = out / "prosody_raw"
    plan_dir = out / "prosody_plans"
    seq_dir = out / "sequences"
    split_json = out / "split.json"

    speed_factors = [float(x) for x in args.augment_speeds.split(",")]

    run_all = args.stage == -1

    if run_all or args.stage == 1:
        print("\n=== Stage 1: Extract phonemes ===")
        stage1_extract_phonemes(metadata, phoneme_dir, args.flite_bin,
                                args.dataset_format)

    if run_all or args.stage == 2:
        if codec_type == "qwen12hz":
            print(f"\n=== Stage 2: Extract Qwen 12Hz tokens [shard {args.shard_id}/{args.num_shards}] ===")
            stage2_extract_qwen(wav_dir, codec_dir, metadata,
                                args.dataset_format, args.device,
                                shard_id=args.shard_id,
                                num_shards=args.num_shards)
        else:
            print("\n=== Stage 2: Extract SNAC tokens ===")
            stage2_extract_snac(wav_dir, codec_dir, metadata,
                                args.dataset_format, args.device)

    if run_all or args.stage == 3:
        print("\n=== Stage 3: Whisper forced alignment ===")
        stage3_align_whisper(
            wav_dir, wb_dir, metadata,
            dataset_format=args.dataset_format,
            whisper_model=args.whisper_model,
            device=args.device,
            batch_size=args.align_batch_size,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

    if run_all or args.stage == 4:
        print("\n=== Stage 4: Extract prosodic features ===")
        stage4_extract_prosody(wav_dir, prosody_dir, metadata,
                               args.dataset_format)

    if run_all or args.stage == 5:
        print("\n=== Stage 5: Build prosodic plans ===")
        stage5_build_plans(
            wav_dir, prosody_dir, plan_dir, metadata,
            dataset_format=args.dataset_format,
            word_boundaries_dir=wb_dir if wb_dir.exists() else None,
        )

    if run_all or args.stage == 6:
        if args.no_augment:
            print("\n=== Stage 6: Split only (no augmentation) ===")
            _split_only(metadata, out, args.dataset_format,
                        args.val_split, args.test_split, args.seed,
                        codec_type=codec_type)
        else:
            print("\n=== Stage 6: Split + Augment ===")
            stage6_split_augment(
                metadata, wav_dir, out,
                dataset_format=args.dataset_format,
                val_split=args.val_split,
                test_split=args.test_split,
                speed_factors=speed_factors,
                seed=args.seed,
                device=args.device,
                codec_type=codec_type,
            )

    if run_all or args.stage == 7:
        n_cb = args.n_codebooks if codec_type == "qwen12hz" else 3
        print(f"\n=== Stage 7: Build tokenizer + JSONL (codec={codec_type}, n_codebooks={n_cb}) ===")
        stage7_build_sequences(
            phoneme_dir, codec_dir, plan_dir, seq_dir,
            metadata_csv=metadata,
            dataset_format=args.dataset_format,
            max_seq_len=args.max_seq_len,
            split_json=split_json if split_json.exists() else None,
            codec_type=codec_type,
            n_codebooks=n_cb,
        )


if __name__ == "__main__":
    main()
