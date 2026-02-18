#!/bin/bash
# CGN v2 LJSpeech recipe — full pipeline
# Usage: bash recipes/ljspeech_cgn_v2/run.sh [stage]
#
# Stages:
#   0  Download LJSpeech data
#   1  Extract Flite phonemes
#   2  Extract SNAC codec tokens (GPU)
#   3  Whisper forced alignment (word boundaries)
#   4  Extract prosodic features (F0, energy)
#   5  Build prosodic plan tokens
#   6  Split + augment data
#   7  Build tokenizer + JSONL sequences
#   8  Train model
#   9  Generate test samples
#
# Set STAGE to run from a specific stage: bash run.sh 3

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$RECIPE_DIR/../.." && pwd)"
WORK="$RECIPE_DIR/work"
DATA="$WORK/LJSpeech-1.1"
PREPARED="$WORK/prepared"
CKPT_DIR="$WORK/checkpoints"

STAGE=${1:-0}
DEVICE=${DEVICE:-cuda:0}
FLITE=${FLITE:-flite}
MODEL_SIZE=${MODEL_SIZE:-small}
WANDB_PROJECT=${WANDB_PROJECT:-festvox3.0}
WHISPER_MODEL=${WHISPER_MODEL:-base.en}
CODEC=${CODEC:-snac}  # "snac" or "qwen12hz"
N_CODEBOOKS=${N_CODEBOOKS:-4}  # Qwen codebooks to interleave (1-16)

cd "$PROJECT_DIR"

# ── Stage 0: Download data ─────────────────────────────────────────────────
if [ "$STAGE" -le 0 ]; then
    echo "=== Stage 0: Download LJSpeech ==="
    mkdir -p "$WORK"
    if [ ! -d "$DATA" ]; then
        cd "$WORK"
        wget -q "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        tar xjf LJSpeech-1.1.tar.bz2
        rm -f LJSpeech-1.1.tar.bz2
        cd "$PROJECT_DIR"
    else
        echo "  LJSpeech already downloaded."
    fi
fi

# ── Stage 1: Phonemes ──────────────────────────────────────────────────────
if [ "$STAGE" -le 1 ]; then
    echo "=== Stage 1: Extract phonemes ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 1 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED" \
        --flite_bin "$FLITE"
fi

# ── Stage 2: Codec tokens ─────────────────────────────────────────────────
if [ "$STAGE" -le 2 ]; then
    echo "=== Stage 2: Extract codec tokens ($CODEC) ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 2 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED" \
        --device "$DEVICE" \
        --codec "$CODEC"
fi

# ── Stage 3: Whisper forced alignment ────────────────────────────────────────
if [ "$STAGE" -le 3 ]; then
    echo "=== Stage 3: Whisper forced alignment ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 3 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED" \
        --whisper_model "$WHISPER_MODEL" \
        --device "$DEVICE"
fi

# ── Stage 4: Prosodic features ────────────────────────────────────────────
if [ "$STAGE" -le 4 ]; then
    echo "=== Stage 4: Extract prosodic features ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 4 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED"
fi

# ── Stage 5: Prosodic plans ───────────────────────────────────────────────
if [ "$STAGE" -le 5 ]; then
    echo "=== Stage 5: Build prosodic plans ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 5 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED"
fi

# ── Stage 6: Split + Augment ────────────────────────────────────────────
if [ "$STAGE" -le 6 ]; then
    echo "=== Stage 6: Split + Augment ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 6 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED" \
        --device "$DEVICE" \
        --codec "$CODEC"
fi

# ── Stage 7: Tokenizer + JSONL ────────────────────────────────────────────
if [ "$STAGE" -le 7 ]; then
    echo "=== Stage 7: Build tokenizer + JSONL ==="
    python3 -u -m festvox3.models.cgn_v2.prepare_data \
        --stage 7 \
        --metadata_csv "$DATA/metadata.csv" \
        --wav_dir "$DATA/wavs" \
        --output_dir "$PREPARED" \
        --codec "$CODEC" \
        --n_codebooks "$N_CODEBOOKS"
fi

# ── Stage 8: Train ────────────────────────────────────────────────────────
if [ "$STAGE" -le 8 ]; then
    echo "=== Stage 8: Train CGN v2 ==="
    mkdir -p "$CKPT_DIR"
    python3 -u -m festvox3.models.cgn_v2.train \
        --train_jsonl "$PREPARED/sequences/data.train.jsonl" \
        --val_jsonl "$PREPARED/sequences/data.val.jsonl" \
        --tokenizer_json "$PREPARED/sequences/tokenizer.json" \
        --output_dir "$CKPT_DIR" \
        --model_size "$MODEL_SIZE" \
        --fp16 \
        --wandb_project "$WANDB_PROJECT" \
        --device "$DEVICE" \
        --codec "$CODEC"
fi

# ── Stage 9: Generate test samples ────────────────────────────────────────
if [ "$STAGE" -le 9 ]; then
    echo "=== Stage 9: Generate test samples ==="
    SAMPLE_DIR="$WORK/samples"
    mkdir -p "$SAMPLE_DIR"

    LATEST="$CKPT_DIR/latest.pt"
    if [ ! -f "$LATEST" ]; then
        echo "  No checkpoint found at $LATEST"
        exit 1
    fi

    sentences=(
        "The quick brown fox jumps over the lazy dog."
        "Hello, my name is LJ. How can I help you today?"
        "In a hole in the ground there lived a hobbit."
        "To be or not to be, that is the question."
    )

    for i in "${!sentences[@]}"; do
        echo "  Generating sample $i..."
        python3 -u -m festvox3.models.cgn_v2.generate \
            --text "${sentences[$i]}" \
            --checkpoint "$LATEST" \
            --tokenizer_json "$PREPARED/sequences/tokenizer.json" \
            --output "$SAMPLE_DIR/sample_${i}.wav" \
            --model_size "$MODEL_SIZE" \
            --device "$DEVICE" \
            --cfg_scale 1.5 \
            --show_plan \
            --codec "$CODEC"
    done

    echo "  Samples saved to $SAMPLE_DIR/"
fi

echo "Done!"
