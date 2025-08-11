#!/bin/bash

# Main pipeline script for Prefix-LM TTS project
# This script runs the complete pipeline from data preparation to model training

set -e  # Exit on any error

# Project configuration
PROJECT_DIR="/home2/srallaba/projects/project_lightweighttts"
DATA_DIR="$PROJECT_DIR/data"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
SCRIPTS_DIR='/home2/srallaba/projects/project_lightweighttts/repos/festvox/src/falcon/prefix_tts'

# Data paths
LJSPEECH_DIR="$DATA_DIR/LJSpeech-1.1"
PHONEMES_DIR="$DATA_DIR/phonemes_ljspeech"
MAX_SAMPLES=100000  # optional limit for testing

# Check required environment variables
if [ -z "${FLITEDIR:-}" ]; then
    echo "Error: FLITEDIR environment variable not set"
    echo "Please set FLITEDIR to point to your Flite installation"
    exit 1
fi

echo "=============================================="
echo "Prefix-LM TTS Pipeline"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "LJSpeech data: $LJSPEECH_DIR"
echo "Flite directory: $FLITEDIR"
if [ -n "$MAX_SAMPLES" ]; then
    echo "Sample limit: $MAX_SAMPLES utterances"
fi
echo ""

# Function to run a step with timing and error handling
run_step() {
    local step_name="$1"
    local step_command="$2"
    
    echo "----------------------------------------"
    echo "Step: $step_name"
    echo "----------------------------------------"
    
    start_time=$(date +%s)
    
    if eval "$step_command"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "✓ $step_name completed successfully in ${duration}s"
    else
        echo "✗ $step_name failed"
        exit 1
    fi
    echo ""
}

# Step 1: Extract phonemes from LJSpeech dataset
echo "Starting pipeline..."
echo ""

# Check if phonemes already exist
if [ -f "$PHONEMES_DIR/ljspeech_phonemes.jsonl" ] && [ -f "$PHONEMES_DIR/phoneme_vocab.txt" ]; then
    phoneme_count=$(wc -l < "$PHONEMES_DIR/ljspeech_phonemes.jsonl")
    vocab_size=$(wc -l < "$PHONEMES_DIR/phoneme_vocab.txt")
    echo "Phonemes already extracted - skipping extraction step"
    echo "  - Found phonemes for $phoneme_count utterances"
    echo "  - Vocabulary size: $vocab_size unique phonemes"
    echo ""
else
    run_step "Extract Phonemes" \
        "$SCRIPTS_DIR/extract_phonemes_simple.sh '$LJSPEECH_DIR' '$PHONEMES_DIR' '$MAX_SAMPLES'"
fi

# Check phoneme extraction results
if [ -f "$PHONEMES_DIR/ljspeech_phonemes.jsonl" ]; then
    phoneme_count=$(wc -l < "$PHONEMES_DIR/ljspeech_phonemes.jsonl")
    vocab_size=$(wc -l < "$PHONEMES_DIR/phoneme_vocab.txt")
    echo "Phoneme extraction results:"
    echo "  - Extracted phonemes for $phoneme_count utterances"
    echo "  - Vocabulary size: $vocab_size unique phonemes"
    echo ""
else
    echo "Error: Phoneme extraction failed - no JSONL output file found"
    exit 1
fi

# Step 2: Extract audio tokens using codec directly from phoneme JSONL
CODEC_DIR="$DATA_DIR/ljspeech_tokens"

# Check if codec tokens already exist
if [ -d "$CODEC_DIR/vq_codes" ] && [ $(find "$CODEC_DIR/vq_codes" -name "*.npy" | wc -l) -gt 0 ]; then
    codec_count=$(find "$CODEC_DIR/vq_codes" -name "*.npy" | wc -l)
    echo "Audio codec tokens already extracted - skipping codec extraction step"
    echo "  - Found tokens for $codec_count audio files"
    echo ""
else
    echo 'fname,dur' > flist.txt
    cat /home2/srallaba/projects/project_lightweighttts/data/ljspeech_prepared/ljspeech_prepared.jsonl | cut -d',' -f 1 | tr "\"" " " | cut -d':' -f 2 | cut -d'/' -f 9 | sed 's/$/\t1.0/' >> flist.txt
    run_step "Extract audio tokens" \
    "CUDA_VISIBLE_DEVICES=1 python3 /home2/srallaba/projects/project_lightweighttts/repos/X-Codec-2.0/inference_save_code.py \
          --input-dir /home2/srallaba/projects/project_lightweighttts/data/LJSpeech-1.1/wavs \
          --flist_file flist.txt \
          --ckpt /home2/srallaba/projects/project_lightweighttts/repos/X-Codec-2.0/ckpt/epoch=4-step=1400000.ckpt \
          --output-dir $CODEC_DIR"
fi

# Check codec extraction results
if [ -d "$CODEC_DIR/vq_codes" ]; then
    codec_count=$(find "$CODEC_DIR/vq_codes" -name "*.npy" | wc -l)
    echo "Codec extraction results:"
    echo "  - Extracted tokens for $codec_count audio files"
    echo ""
else
    echo "Error: Codec extraction failed - no VQ codes directory found"
    exit 1
fi

# Step 3: Prepare training dataset
TRAINING_DIR="$DATA_DIR/training_data"

# Check if training dataset already exists
if [ -f "$TRAINING_DIR/train.jsonl" ] && [ -f "$TRAINING_DIR/val.jsonl" ] && [ -f "$TRAINING_DIR/vocab.json" ]; then
    train_count=$(wc -l < "$TRAINING_DIR/train.jsonl")
    val_count=$(wc -l < "$TRAINING_DIR/val.jsonl")
    echo "Training dataset already prepared - skipping dataset preparation step"
    echo "  - Training examples: $train_count"
    echo "  - Validation examples: $val_count"
    echo ""
else
    run_step "Prepare Training Dataset" \
        "python3 $SCRIPTS_DIR/prepare_training_data.py \
            --phonemes_jsonl '$PHONEMES_DIR/ljspeech_phonemes.jsonl' \
            --codec_dir '$CODEC_DIR/vq_codes' \
            --phoneme_vocab '$PHONEMES_DIR/phoneme_vocab.txt' \
            --output_dir '$TRAINING_DIR' \
            --val_split 0.1"
fi

# Check training data preparation results
if [ -f "$TRAINING_DIR/train.jsonl" ] && [ -f "$TRAINING_DIR/val.jsonl" ]; then
    train_count=$(wc -l < "$TRAINING_DIR/train.jsonl")
    val_count=$(wc -l < "$TRAINING_DIR/val.jsonl")
    echo "Training data preparation results:"
    echo "  - Training examples: $train_count"
    echo "  - Validation examples: $val_count"
    echo "  - Vocabulary file: $TRAINING_DIR/vocab.json"
    echo "  - Dataset stats: $TRAINING_DIR/dataset_stats.json"
    echo ""
else
    echo "Error: Training data preparation failed"
    exit 1
fi

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Generated files:"
echo "  - Phonemes (JSONL): $PHONEMES_DIR/ljspeech_phonemes.jsonl"
echo "  - Phoneme vocabulary: $PHONEMES_DIR/phoneme_vocab.txt"
echo "  - Audio codec tokens: $CODEC_DIR/vq_codes/*.npy"
echo "  - Training data: $TRAINING_DIR/train.jsonl"
echo "  - Validation data: $TRAINING_DIR/val.jsonl"
echo "  - Model vocabulary: $TRAINING_DIR/vocab.json"
echo "  - Dataset statistics: $TRAINING_DIR/dataset_stats.json"
echo ""
echo "Next steps to implement:"
echo "  - Model architecture definition (Prefix-LM Transformer)"
echo "  - Training script with PyTorch/HuggingFace"
echo "  - Model evaluation and inference"
echo "  - Model export and deployment"
echo "=============================================="
