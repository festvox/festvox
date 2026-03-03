#!/bin/bash

# Simple phoneme extraction script for LJSpeech dataset using Flite

# Default paths
LJSPEECH_DIR="${1:-/lambda/nfs/falcon-revamp/projects/project_prefixtts/data/ljspeech/LJSpeech-1.1}"
OUTPUT_DIR="${2:-/lambda/nfs/falcon-revamp/projects/project_prefixtts/data/phonemes}"
MAX_SAMPLES="${3:-}"  # optional limit for testing

# Check if FLITEDIR is set
if [ -z "${FLITEDIR:-}" ]; then
    echo "Error: FLITEDIR environment variable not set"
    exit 1
fi

FLITE_BIN="$FLITEDIR/bin/flite"
if [ ! -x "$FLITE_BIN" ]; then
    echo "Error: Flite binary not found at $FLITE_BIN"
    exit 1
fi

METADATA_FILE="$LJSPEECH_DIR/metadata.csv"
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: Metadata file not found: $METADATA_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting phoneme extraction..."
echo "LJSpeech directory: $LJSPEECH_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Using Flite from: $FLITE_BIN"

# Output files
PHONEME_FILE="$OUTPUT_DIR/ljspeech_phonemes.csv"
ERROR_FILE="$OUTPUT_DIR/extraction_errors.csv"

# Initialize output files
echo "audio_id|text|phonemes" > "$PHONEME_FILE"
echo "audio_id|error" > "$ERROR_FILE"

# Counters
count=0
success=0
errors=0

echo "Starting processing metadata file: $METADATA_FILE"

# Process each line
while IFS='|' read -r audio_id original_text normalized_text; do
    ((count++))
    
    # Use normalized text (3rd column)
    text="$normalized_text"
    
    # Progress
    if ((count % 100 == 1)) || ((count <= 10)); then
        echo "Processing $count: $audio_id"
    fi
    
    # Limit samples if specified
    if [ -n "$MAX_SAMPLES" ] && ((count > MAX_SAMPLES)); then
        echo "Reached limit of $MAX_SAMPLES samples"
        break
    fi
    
    # Extract phonemes
    phonemes=$("$FLITE_BIN" -t "$text" -o /dev/null -ps 2>/dev/null) || phonemes=""
    if [ -n "$phonemes" ]; then
        # Escape pipe characters for CSV
        escaped_text=$(echo "$text" | sed 's/|/\\|/g')
        escaped_phonemes=$(echo "$phonemes" | tr -s ' ' | sed 's/^ *//;s/ *$//' | sed 's/|/\\|/g')
        
        echo "$audio_id|$escaped_text|$escaped_phonemes" >> "$PHONEME_FILE"
        ((success++))
    else
        echo "$audio_id|No phonemes extracted or flite failed" >> "$ERROR_FILE"
        ((errors++))
    fi
    
done < "$METADATA_FILE"

echo "Extraction completed!"
echo "Total processed: $count"
echo "Successful: $success"
echo "Errors: $errors"

# Generate vocabulary if we have results
if [ "$success" -gt 0 ]; then
    VOCAB_FILE="$OUTPUT_DIR/phoneme_vocab.txt"
    tail -n +2 "$PHONEME_FILE" | cut -d'|' -f3 | tr ' ' '\n' | grep -v '^$' | sort -u > "$VOCAB_FILE"
    vocab_size=$(wc -l < "$VOCAB_FILE")
    echo "Phoneme vocabulary saved to $VOCAB_FILE ($vocab_size unique phonemes)"
fi

echo "Output files:"
echo "  Phonemes: $PHONEME_FILE"
if [ -f "$VOCAB_FILE" ]; then
    echo "  Vocabulary: $VOCAB_FILE"
fi
if [ "$errors" -gt 0 ]; then
    echo "  Errors: $ERROR_FILE"
fi
