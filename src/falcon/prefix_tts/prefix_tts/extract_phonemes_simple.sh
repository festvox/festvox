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
PHONEME_FILE="$OUTPUT_DIR/ljspeech_phonemes.jsonl"
ERROR_FILE="$OUTPUT_DIR/extraction_errors.csv"
VOCAB_FILE="$OUTPUT_DIR/phoneme_vocab.txt"

# Initialize output files
> "$PHONEME_FILE"  # Empty the JSONL file
echo "audio_id|error" > "$ERROR_FILE"

# Initialize vocabulary tracking
> "$VOCAB_FILE.tmp"  # Temporary file for collecting phonemes

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
        # Clean up phonemes (remove extra spaces)
        cleaned_phonemes=$(echo "$phonemes" | tr -s ' ' | sed 's/^ *//;s/ *$//')
        
        # Escape quotes and backslashes for JSON
        escaped_text=$(echo "$text" | sed 's/\\/\\\\/g; s/"/\\"/g')
        escaped_phonemes=$(echo "$cleaned_phonemes" | sed 's/\\/\\\\/g; s/"/\\"/g')
        
        # Create audio path
        audio_path="/home2/srallaba/data/tts/ljspeech/LJSpeech-1.1/wavs/${audio_id}.wav"
        
        # Create JSONL entry with audio_path
        echo "{\"audio_id\": \"$audio_id\", \"text\": \"$escaped_text\", \"phonemes\": \"$escaped_phonemes\", \"audio_path\": \"$audio_path\"}" >> "$PHONEME_FILE"
        
        # Collect phonemes for vocabulary
        echo "$cleaned_phonemes" | tr ' ' '\n' | grep -v '^$' >> "$VOCAB_FILE.tmp"
        
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
    sort -u "$VOCAB_FILE.tmp" > "$VOCAB_FILE"
    rm -f "$VOCAB_FILE.tmp"
    vocab_size=$(wc -l < "$VOCAB_FILE")
    echo "Phoneme vocabulary saved to $VOCAB_FILE ($vocab_size unique phonemes)"
fi

echo "Output files:"
echo "  Phonemes (JSONL): $PHONEME_FILE"
if [ -f "$VOCAB_FILE" ]; then
    echo "  Vocabulary: $VOCAB_FILE"
fi
if [ "$errors" -gt 0 ]; then
    echo "  Errors: $ERROR_FILE"
fi
