"""Prosodic feature extraction and plan token generation.

Extracts F0, energy, word boundaries, and phrase structure from audio,
then quantizes into discrete prosodic plan tokens.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np


# ── F0 and energy extraction ───────────────────────────────────────────────

def extract_f0(
    wav_path: str,
    sr: int = 24000,
    frame_period_ms: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 and timebase using pyworld DIO + StoneMask.

    Returns:
        f0: [N] array of F0 values in Hz (0 = unvoiced)
        times: [N] array of time positions in seconds
    """
    import pyworld as pw
    import soundfile as sf

    audio, file_sr = sf.read(wav_path, dtype="float64")
    if file_sr != sr:
        import torchaudio
        import torch
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio = torchaudio.functional.resample(waveform, file_sr, sr).squeeze(0).numpy().astype("float64")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    f0, times = pw.dio(audio, sr, frame_period=frame_period_ms)
    f0 = pw.stonemask(audio, f0, times, sr)
    return f0, times


def extract_energy(
    wav_path: str,
    sr: int = 24000,
    hop_length: int = 240,  # 10ms at 24kHz
) -> np.ndarray:
    """Extract frame-level RMS energy.

    Returns:
        energy: [N] array of RMS energy values
    """
    import soundfile as sf

    audio, file_sr = sf.read(wav_path, dtype="float64")
    if file_sr != sr:
        import torchaudio
        import torch
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio = torchaudio.functional.resample(waveform, file_sr, sr).squeeze(0).numpy().astype("float64")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # RMS energy per frame
    n_frames = len(audio) // hop_length
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + hop_length
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return energy


# ── Word boundary detection ────────────────────────────────────────────────

def detect_word_boundaries_simple(
    wav_path: str,
    text: str,
    sr: int = 24000,
) -> List[dict]:
    """Detect word boundaries using silence-based segmentation.

    Falls back to uniform distribution if detection fails.

    Returns:
        List of dicts: [{"word": str, "start": float, "end": float}, ...]
    """
    import soundfile as sf

    audio, file_sr = sf.read(wav_path, dtype="float64")
    if file_sr != sr:
        import torchaudio
        import torch
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio = torchaudio.functional.resample(waveform, file_sr, sr).squeeze(0).numpy().astype("float64")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    words = text.strip().split()
    if not words:
        return []

    # Simple uniform distribution as baseline
    # In production, replace with Whisper word timestamps or MFA
    word_dur = duration / len(words)
    result = []
    for i, word in enumerate(words):
        result.append({
            "word": word,
            "start": i * word_dur,
            "end": (i + 1) * word_dur,
        })
    return result


def detect_pauses(
    energy: np.ndarray,
    sr: int = 24000,
    hop_length: int = 240,
    silence_threshold: float = 0.01,
    min_pause_frames: int = 5,  # 50ms
) -> List[dict]:
    """Detect pauses from energy contour.

    Returns:
        List of dicts: [{"start": float, "end": float, "duration": float}, ...]
    """
    is_silent = energy < silence_threshold
    pauses = []
    in_pause = False
    pause_start = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_pause:
            in_pause = True
            pause_start = i
        elif not silent and in_pause:
            in_pause = False
            pause_len = i - pause_start
            if pause_len >= min_pause_frames:
                t_start = pause_start * hop_length / sr
                t_end = i * hop_length / sr
                pauses.append({
                    "start": t_start,
                    "end": t_end,
                    "duration": t_end - t_start,
                })

    # Handle trailing pause
    if in_pause:
        pause_len = len(is_silent) - pause_start
        if pause_len >= min_pause_frames:
            t_start = pause_start * hop_length / sr
            t_end = len(is_silent) * hop_length / sr
            pauses.append({
                "start": t_start,
                "end": t_end,
                "duration": t_end - t_start,
            })

    return pauses


# ── Prosodic plan quantization ─────────────────────────────────────────────

def quantize_duration(duration_sec: float, speaking_rate: float = 1.0) -> str:
    """Quantize a word duration into one of 8 bins.

    speaking_rate: average words per second in the utterance (for normalization).
    """
    # Normalize duration relative to average
    if speaking_rate > 0:
        avg_dur = 1.0 / speaking_rate
        ratio = duration_sec / avg_dur
    else:
        ratio = 1.0

    if ratio < 0.3:
        return "dur:very_short"
    elif ratio < 0.6:
        return "dur:short"
    elif ratio < 0.8:
        return "dur:med_short"
    elif ratio < 1.2:
        return "dur:medium"
    elif ratio < 1.5:
        return "dur:med_long"
    elif ratio < 2.0:
        return "dur:long"
    elif ratio < 3.0:
        return "dur:very_long"
    else:
        return "dur:extra_long"


def classify_pitch_contour(
    f0_segment: np.ndarray,
    is_question: bool = False,
) -> str:
    """Classify the F0 contour shape of a phrase segment.

    Args:
        f0_segment: F0 values for this phrase (0 = unvoiced, filtered out)
        is_question: whether the text ends with '?'
    """
    voiced = f0_segment[f0_segment > 0]
    if len(voiced) < 3:
        return "pitch:flat"

    if is_question:
        return "pitch:question"

    # Compute trend
    mean_f0 = np.mean(voiced)
    first_half = np.mean(voiced[: len(voiced) // 2])
    second_half = np.mean(voiced[len(voiced) // 2 :])

    rel_change = (second_half - first_half) / (mean_f0 + 1e-10)

    # Check for rise-fall or fall-rise
    if len(voiced) >= 6:
        third = len(voiced) // 3
        t1 = np.mean(voiced[:third])
        t2 = np.mean(voiced[third : 2 * third])
        t3 = np.mean(voiced[2 * third :])

        if t2 > t1 * 1.05 and t2 > t3 * 1.05:
            return "pitch:rise_fall"
        if t2 < t1 * 0.95 and t2 < t3 * 0.95:
            return "pitch:fall_rise"

    if rel_change > 0.1:
        return "pitch:rising"
    elif rel_change < -0.1:
        return "pitch:falling"

    if mean_f0 > 200:  # relatively high (rough heuristic)
        return "pitch:high"
    elif mean_f0 < 120:
        return "pitch:low"

    return "pitch:flat"


def classify_emphasis(
    word_energy: float,
    word_f0_mean: float,
    global_energy_mean: float,
    global_energy_std: float,
    global_f0_mean: float,
    global_f0_std: float,
) -> str:
    """Classify emphasis level for a word based on F0 and energy deviation."""
    energy_z = (word_energy - global_energy_mean) / (global_energy_std + 1e-10)
    f0_z = (word_f0_mean - global_f0_mean) / (global_f0_std + 1e-10)

    # Combined score
    emphasis_score = 0.6 * energy_z + 0.4 * f0_z

    if emphasis_score > 1.5:
        return "emph:contrastive"
    elif emphasis_score > 0.5:
        return "emph:stressed"
    elif emphasis_score > -0.5:
        return "emph:normal"
    else:
        return "emph:unstressed"


def quantize_pause(duration_sec: float) -> str:
    """Quantize a pause duration into one of 4 types."""
    if duration_sec < 0.05:
        return "pause:none"
    elif duration_sec < 0.2:
        return "pause:short"
    elif duration_sec < 0.5:
        return "pause:medium"
    else:
        return "pause:long"


# ── Full prosodic plan extraction ──────────────────────────────────────────

def extract_prosodic_plan(
    wav_path: str,
    text: str,
    sr: int = 24000,
    word_boundaries: list[dict] | None = None,
) -> List[str]:
    """Extract a full prosodic plan token sequence for an utterance.

    Args:
        word_boundaries: If provided, use these instead of uniform detection.
            List of dicts: [{"word": str, "start": float, "end": float}, ...]

    Returns:
        List of prosodic plan token strings in sequence order.
    """
    # Extract raw features
    f0, times = extract_f0(wav_path, sr=sr)
    energy = extract_energy(wav_path, sr=sr)
    if word_boundaries is not None:
        words = word_boundaries
    else:
        words = detect_word_boundaries_simple(wav_path, text, sr=sr)
    pauses = detect_pauses(energy, sr=sr)

    if not words:
        return []

    # Global stats for normalization
    voiced_f0 = f0[f0 > 0]
    global_f0_mean = np.mean(voiced_f0) if len(voiced_f0) > 0 else 150.0
    global_f0_std = np.std(voiced_f0) if len(voiced_f0) > 1 else 30.0
    global_energy_mean = np.mean(energy) if len(energy) > 0 else 0.05
    global_energy_std = np.std(energy) if len(energy) > 1 else 0.02

    # Speaking rate for duration normalization
    total_dur = words[-1]["end"] - words[0]["start"]
    speaking_rate = len(words) / total_dur if total_dur > 0 else 3.0

    hop_sec = 240.0 / sr  # 10ms frames

    plan_tokens: List[str] = []

    # Simple phrase detection: split on pauses > 200ms
    phrase_boundaries = set()
    for p in pauses:
        if p["duration"] > 0.2:
            # Find the word just after this pause
            for j, w in enumerate(words):
                if w["start"] >= p["end"] - 0.05:
                    phrase_boundaries.add(j)
                    break

    # Add phrase start at beginning
    phrase_boundaries.add(0)

    # Detect if text ends with question mark
    is_question = text.strip().endswith("?")

    for i, word_info in enumerate(words):
        # Phrase boundary
        if i in phrase_boundaries:
            if i > 0:
                # Close previous phrase with pitch contour
                phrase_start_time = words[phrase_start_idx]["start"]
                phrase_end_time = words[i - 1]["end"]
                f0_start = int(phrase_start_time / hop_sec)
                f0_end = int(phrase_end_time / hop_sec)
                f0_end = min(f0_end, len(f0))
                phrase_f0 = f0[f0_start:f0_end]
                is_q = is_question and (i == len(words))
                contour = classify_pitch_contour(phrase_f0, is_q)
                plan_tokens.append(contour)
                plan_tokens.append("struct:phrase_end")

            plan_tokens.append("struct:phrase_start")
            phrase_start_idx = i

        # Check for pause before this word
        pause_before = "pause:none"
        if i > 0:
            gap = word_info["start"] - words[i - 1]["end"]
            pause_before = quantize_pause(gap)
        if pause_before != "pause:none":
            plan_tokens.append(pause_before)

        # Word start
        plan_tokens.append("struct:word_start")

        # Duration
        word_dur = word_info["end"] - word_info["start"]
        plan_tokens.append(quantize_duration(word_dur, speaking_rate))

        # Emphasis (from energy and F0 in word region)
        w_start_frame = int(word_info["start"] / hop_sec)
        w_end_frame = int(word_info["end"] / hop_sec)
        w_end_frame = min(w_end_frame, len(energy))

        if w_start_frame < w_end_frame:
            word_energy = np.mean(energy[w_start_frame:w_end_frame])
            word_f0_seg = f0[min(w_start_frame, len(f0) - 1):min(w_end_frame, len(f0))]
            word_f0_voiced = word_f0_seg[word_f0_seg > 0]
            word_f0_mean = np.mean(word_f0_voiced) if len(word_f0_voiced) > 0 else global_f0_mean
        else:
            word_energy = global_energy_mean
            word_f0_mean = global_f0_mean

        plan_tokens.append(classify_emphasis(
            word_energy, word_f0_mean,
            global_energy_mean, global_energy_std,
            global_f0_mean, global_f0_std,
        ))

    # Close final phrase
    if words:
        phrase_start_time = words[phrase_start_idx]["start"]
        phrase_end_time = words[-1]["end"]
        f0_start = int(phrase_start_time / hop_sec)
        f0_end = int(phrase_end_time / hop_sec)
        f0_end = min(f0_end, len(f0))
        phrase_f0 = f0[f0_start:f0_end]
        contour = classify_pitch_contour(phrase_f0, is_question)
        plan_tokens.append(contour)
        plan_tokens.append("struct:phrase_end")

    return plan_tokens
