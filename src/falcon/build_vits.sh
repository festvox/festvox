#!/bin/bash

# Build VITS2 training data and features for any dataset using Festvox tools
# Usage: ./build_vits.sh <dataset_name> <wav_dir> <exp_dir>

set -e




DATASET="$1"
EXP_DIR="$2"
WAV_DIR="$EXP_DIR/wav"
MCEP_DIR="$EXP_DIR/mcep"


mkdir -p "$EXP_DIR"
mkdir -p "$MCEP_DIR"
mkdir -p "$WAV_DIR"




# Step 0: Download or link/copy dataset WAVs if needed
if [ ! -d "$WAV_DIR" ] || [ -z "$(ls -A "$WAV_DIR" 2>/dev/null)" ]; then
    echo "WAV directory $WAV_DIR is empty."
    read -p "Do you want to (d)ownload or (l)ink/copy from local? [d/l]: " choice
    if [ "$choice" = "d" ]; then
        echo "Downloading $DATASET WAVs to $WAV_DIR ..."
        if [ "$DATASET" = "ljspeech" ]; then
            python "$FESTVOXDIR/src/falcon/utils/ljspeech/download_ljspeech.py" "$WAV_DIR"
        else
            echo "Dataset $DATASET not supported for automatic download. Please add a download script."
            exit 1
        fi
    elif [ "$choice" = "l" ]; then
        read -p "Enter full path to local WAV source directory: " SRC_WAV_DIR
        echo "Linking WAVs from $SRC_WAV_DIR to $WAV_DIR ..."
        for f in "$SRC_WAV_DIR"/*.wav; do
            [ -e "$f" ] && ln -s "$f" "$WAV_DIR/"
        done
        echo "Symlinks created in $WAV_DIR."
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi
fi

# Step 1: Extract mel-spectrogram features using torchaudio (if not already present)
if ! ls "$MCEP_DIR"/*.mel.pt 1> /dev/null 2>&1; then
    echo "Extracting mel-spectrograms from WAVs using torchaudio..."
    python "$FESTVOXDIR/src/falcon/utils/ljspeech/extract_mel_ljspeech.py" "$WAV_DIR" "$MCEP_DIR"
else
    echo "Mel-spectrogram files already present in $MCEP_DIR, skipping extraction."
fi


# Step 2: (Optional) Convert MCEP to mel-spectrogram if needed
# TODO: Add conversion step if required by VITS2
# Example:
# python "$FESTVOXDIR/src/falcon/utils/ljspeech/mcep_to_mel.py" "$MCEP_DIR" "$EXP_DIR/mel"


# Step 3: Prepare training manifest or file list

ls "$MCEP_DIR"/*.mcep > "$EXP_DIR/mcep_files.txt"
ls "$WAV_DIR"/*.wav > "$EXP_DIR/wav_files.txt"

echo "Extracted $num_mcep MCEP files to $MCEP_DIR"
echo "Manifest files created in $EXP_DIR"
echo "Ready for VITS2 training."

# Step 4: Print summary
num_mcep=$(ls "$MCEP_DIR"/*.mcep | wc -l)
echo "Extracted $num_mcep MCEP files to $MCEP_DIR"
echo "Manifest files created in $EXP_DIR"

# Step 5: Launch VITS2 training (edit config path as needed)
CONFIG_PATH="$EXP_DIR/config.yaml"
echo "Launching VITS2 training..."
python "$FESTVOXDIR/src/falcon/train_vits2.py" --config "$CONFIG_PATH" --output_dir "$EXP_DIR" --epochs 10 --batch_size 4
