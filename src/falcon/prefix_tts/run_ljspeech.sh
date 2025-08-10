#!/bin/bash

# Main pipeline script for Prefix-LM TTS project
# This script runs the complete pipeline from data preparation to model training

set -e  # Exit on any error

# Project configuration
PROJECT_DIR="/lambda/nfs/falcon-revamp/projects/project_prefixtts"
DATA_DIR="$PROJECT_DIR/data"
SCRIPTS_DIR="$PROJECT_DIR/scripts"

# Data paths
LJSPEECH_DIR="$DATA_DIR/ljspeech/LJSpeech-1.1"
PHONEMES_DIR="$DATA_DIR/phonemes"

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

run_step "Extract Phonemes" \
    "$SCRIPTS_DIR/extract_phonemes_simple.sh '$LJSPEECH_DIR' '$PHONEMES_DIR'"

# Check phoneme extraction results
if [ -f "$PHONEMES_DIR/ljspeech_phonemes.csv" ]; then
    phoneme_count=$(tail -n +2 "$PHONEMES_DIR/ljspeech_phonemes.csv" | wc -l)
    vocab_size=$(wc -l < "$PHONEMES_DIR/phoneme_vocab.txt")
    echo "Phoneme extraction results:"
    echo "  - Extracted phonemes for $phoneme_count utterances"
    echo "  - Vocabulary size: $vocab_size unique phonemes"
    echo ""
else
    echo "Error: Phoneme extraction failed - no output file found"
    exit 1
fi

echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Next steps to implement:"
echo "  - Audio tokenization (neural codec)"
echo "  - Dataset preparation for training"
echo "  - Model training (Prefix-LM Transformer)"
echo "  - Model evaluation and export"
echo "=============================================="
