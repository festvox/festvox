#!/bin/bash
# MCEP-VITS LJSpeech recipe — full pipeline
# Usage: bash run.sh [stage]
#
# Stages:
#   0  Download LJSpeech + install dependencies
#   1  Build filelists (train/val/test splits)
#   2  Preprocess text (flite G2P -> ARPAbet phones)
#   3  Build Monotonic Alignment Search (Cython)
#   4  Train
#   5  Generate test samples
#
# Set STAGE to run from a specific stage: bash run.sh 3

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "$0")" && pwd)"
FESTVOX_DIR="$(cd "$RECIPE_DIR/../../.." && pwd)"
MCEP_VITS_DIR="$FESTVOX_DIR/src/falcon/mcep_vits"
WORK="$RECIPE_DIR/work"
DATA="$WORK/LJSpeech-1.1"
MODEL_DIR="$WORK/model"

STAGE=${1:-0}
CONFIG=${CONFIG:-"$MCEP_VITS_DIR/configs/ljs_mcep_vits.json"}
BATCH_SIZE=${BATCH_SIZE:-48}
N_GPUS=${N_GPUS:-1}

# ── Stage 0: Download data + install deps ─────────────────────────────────
if [ "$STAGE" -le 0 ]; then
    echo "=== Stage 0: Download LJSpeech + install dependencies ==="
    mkdir -p "$WORK"

    # Install Python dependencies
    pip install torch torchaudio pysptk scipy librosa unidecode soundfile Cython

    # Check flite is installed
    if ! command -v flite &> /dev/null; then
        echo "ERROR: flite not found. Install it:"
        echo "  Ubuntu/Debian: sudo apt-get install flite"
        echo "  From source:   git clone https://github.com/festvox/flite.git && cd flite && ./configure && make && sudo make install"
        exit 1
    fi

    # Download LJSpeech
    if [ ! -d "$DATA" ]; then
        echo "  Downloading LJSpeech-1.1..."
        cd "$WORK"
        wget -q --show-progress "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        tar xjf LJSpeech-1.1.tar.bz2
        rm -f LJSpeech-1.1.tar.bz2
        cd "$RECIPE_DIR"
    else
        echo "  LJSpeech already downloaded at $DATA"
    fi
fi

# ── Stage 1: Build filelists ──────────────────────────────────────────────
if [ "$STAGE" -le 1 ]; then
    echo "=== Stage 1: Build filelists ==="

    WAVDIR="$DATA/wavs"
    METADATA="$DATA/metadata.csv"
    FILELIST_DIR="$MCEP_VITS_DIR/filelists"

    # Generate filelists from metadata.csv: path|text
    # LJSpeech metadata format: ID|text|normalized_text
    python3 -c "
import os, random

metadata = '$METADATA'
wavdir = '$WAVDIR'
outdir = '$FILELIST_DIR'

entries = []
with open(metadata, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        utt_id = parts[0]
        # Use normalized text (3rd column) if available, else raw (2nd)
        text = parts[2] if len(parts) > 2 and parts[2].strip() else parts[1]
        wav_path = os.path.join(wavdir, utt_id + '.wav')
        if os.path.exists(wav_path):
            entries.append(f'{wav_path}|{text}')

random.seed(1234)
random.shuffle(entries)

n = len(entries)
n_val = 100
n_test = 500
train = entries[n_val + n_test:]
val = entries[:n_val]
test = entries[n_val:n_val + n_test]

for split, data in [('train', train), ('val', val), ('test', test)]:
    path = os.path.join(outdir, f'ljs_audio_text_{split}_filelist.txt')
    with open(path, 'w') as f:
        f.writelines([e + '\n' for e in data])
    print(f'  {split}: {len(data)} utterances -> {path}')
"
fi

# ── Stage 2: Preprocess text (flite G2P) ─────────────────────────────────
if [ "$STAGE" -le 2 ]; then
    echo "=== Stage 2: Preprocess text (flite G2P -> ARPAbet) ==="
    cd "$MCEP_VITS_DIR"
    python3 preprocess.py \
        --filelists filelists/ljs_audio_text_train_filelist.txt \
                    filelists/ljs_audio_text_val_filelist.txt \
                    filelists/ljs_audio_text_test_filelist.txt \
        --text_cleaners flite_cleaners
    echo "  Created .cleaned filelists with phoneme sequences"
    cd "$RECIPE_DIR"
fi

# ── Stage 3: Build Monotonic Alignment Search ─────────────────────────────
if [ "$STAGE" -le 3 ]; then
    echo "=== Stage 3: Build Monotonic Alignment Search (Cython) ==="
    cd "$MCEP_VITS_DIR/monotonic_align"
    python3 setup.py build_ext --inplace
    cd "$RECIPE_DIR"
    echo "  Built monotonic_align Cython extension"
fi

# ── Stage 4: Train ────────────────────────────────────────────────────────
if [ "$STAGE" -le 4 ]; then
    echo "=== Stage 4: Train MCEP-VITS ==="
    mkdir -p "$MODEL_DIR"
    cd "$MCEP_VITS_DIR"

    if [ "$N_GPUS" -gt 1 ]; then
        echo "  Training with $N_GPUS GPUs (DDP)..."
        python3 -m torch.distributed.run --nproc_per_node="$N_GPUS" \
            train_latest.py -c "$CONFIG" -m "$MODEL_DIR"
    else
        echo "  Training with 1 GPU..."
        python3 train_latest.py -c "$CONFIG" -m "$MODEL_DIR"
    fi
    cd "$RECIPE_DIR"
fi

# ── Stage 5: Generate test samples ────────────────────────────────────────
if [ "$STAGE" -le 5 ]; then
    echo "=== Stage 5: Generate test samples ==="
    SAMPLE_DIR="$WORK/samples"
    mkdir -p "$SAMPLE_DIR"

    # Find latest generator checkpoint
    LATEST_G=$(ls -t "$MODEL_DIR"/G_*.pth 2>/dev/null | head -1)
    if [ -z "$LATEST_G" ]; then
        echo "  ERROR: No generator checkpoint found in $MODEL_DIR"
        exit 1
    fi
    echo "  Using checkpoint: $LATEST_G"

    cd "$MCEP_VITS_DIR"
    python3 -c "
import torch
import json
import soundfile as sf
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols
import commons

config = '$CONFIG'
checkpoint = '$LATEST_G'
sample_dir = '$SAMPLE_DIR'

with open(config, 'r') as f:
    hps = json.load(f)

# Build model
net_g = SynthesizerTrn(
    len(symbols),
    hps['data']['filter_length'] // 2 + 1,
    hps['train']['segment_size'] // hps['data']['hop_length'],
    **hps['model']
).cuda().eval()

# Load checkpoint
ckpt = torch.load(checkpoint, map_location='cpu')
net_g.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

sentences = [
    'The quick brown fox jumps over the lazy dog.',
    'Hello, my name is LJ. How can I help you today?',
    'In a hole in the ground there lived a hobbit.',
    'To be or not to be, that is the question.',
]

for i, sent in enumerate(sentences):
    seq = text_to_sequence(sent, hps['data']['text_cleaners'])
    x = torch.LongTensor(seq).unsqueeze(0).cuda()
    x_len = torch.LongTensor([len(seq)]).cuda()

    with torch.no_grad():
        audio = net_g.infer(x, x_len, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.0)[0]
        audio = audio[0, 0].cpu().numpy()

    out_path = f'{sample_dir}/sample_{i}.wav'
    sf.write(out_path, audio, hps['data']['sampling_rate'])
    print(f'  [{i}] {sent}')
    print(f'       -> {out_path} ({len(audio)/hps[\"data\"][\"sampling_rate\"]:.1f}s)')
"
    cd "$RECIPE_DIR"
    echo "  Samples saved to $SAMPLE_DIR/"
fi

echo "Done!"
