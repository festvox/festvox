# FALCON: Prefix-LM TTS Pipeline

FALCON is a neural extension to the Festvox voice building suite, now featuring a modern **Prefix Language Model approach to Text-to-Speech synthesis**.

## Overview

This implementation uses a prefix language modeling approach where:
1. **Text phonemes** are encoded as input prefix tokens
2. **Audio codec tokens** are generated autoregressively as continuation
3. The model learns to predict discrete audio tokens given phonemic context

## Architecture

```
Text: "Hello world" 
  ↓ (phoneme extraction)
Phonemes: [h, ə, l, oʊ, w, ɜ˞, l, d]
  ↓ (prefix-LM training)
Sequence: <START> h ə l oʊ w ɜ˞ l d <AUDIO_START> 245 891 342 ... <AUDIO_END> <END>
  ↓ (neural codec decoding)
Audio: waveform.wav
```

## Pipeline Components

### 1. Phoneme Extraction (`extract_phonemes_simple.sh`)
- Converts text to phonemes using Flite's letter-to-sound rules
- Outputs JSONL format: `{"text": "...", "phonemes": [...], "audio_path": "..."}`
- Creates phoneme vocabulary for model training

### 2. Audio Codec Extraction (`X-Codec-2.0`)
- Uses neural audio codec (X-Codec-2.0) to tokenize audio into discrete codes
- Produces vector quantized (VQ) representations: `audio → tokens [245, 891, 342, ...]`
- Saves compressed audio representations as `.npy` files

### 3. Training Data Preparation (`prepare_training_data.py`)
- Combines phonemes and audio tokens into prefix-LM sequences
- Format: `<START> phonemes... <AUDIO_START> audio_tokens... <AUDIO_END> <END>`
- Creates train/validation splits with vocabulary mappings

## Quick Start

### Prerequisites
```bash
# Set Flite directory
export FLITEDIR=/path/to/flite

# Ensure X-Codec-2.0 checkpoint is available
ls /path/to/X-Codec-2.0/ckpt/epoch=4-step=1400000.ckpt
```

### Run Complete Pipeline
```bash
cd src/falcon/prefix_tts
./run_ljspeech.sh
```

This will:
1. Extract phonemes from LJSpeech dataset
2. Generate audio codec tokens using GPU
3. Prepare training dataset in prefix-LM format

### Pipeline Output
```
data/
├── phonemes_ljspeech/
│   ├── ljspeech_phonemes.jsonl      # Phoneme sequences
│   └── phoneme_vocab.txt            # Phoneme vocabulary
├── ljspeech_tokens/
│   └── vq_codes/                    # Audio codec tokens (.npy)
└── training_data/
    ├── train.jsonl                  # Training sequences
    ├── val.jsonl                    # Validation sequences
    ├── vocab.json                   # Complete vocabulary
    └── dataset_stats.json           # Dataset statistics
```

## Key Features

### Smart Resume Capability
- Pipeline automatically skips completed steps
- Resume from any point if process is interrupted
- Efficient re-runs for development and testing

### GPU Optimization
- Configurable GPU usage (`CUDA_VISIBLE_DEVICES=1`)
- Efficient batch processing for codec extraction
- Memory-optimized training data preparation

### Flexible Configuration
- Adjustable sample limits for testing (`MAX_SAMPLES=100`)
- Configurable train/validation splits
- Modular components for easy experimentation

## Technical Details

### Sequence Format
```python
# Training sequence structure:
sequence = [
    2,              # <START>
    phoneme_ids,    # [15, 23, 8, 45, ...]
    4,              # <AUDIO_START>
    audio_tokens,   # [1245, 891, 342, ...]
    5,              # <AUDIO_END>  
    3               # <END>
]
```

### Vocabulary Structure
- **Special tokens**: `<START>`, `<END>`, `<AUDIO_START>`, `<AUDIO_END>`, `<PAD>`, `<UNK>`
- **Phoneme tokens**: Mapped from phoneme vocabulary (e.g., 'ae' → 6)
- **Audio tokens**: Offset by phoneme vocab size (e.g., codec 245 → vocab_offset + 245)

### Model Requirements
- **Input**: Prefix sequences (phonemes + special tokens)
- **Output**: Audio token predictions
- **Architecture**: Transformer-based language model with causal attention
- **Training objective**: Next-token prediction with teacher forcing

## Next Steps

1. **Model Architecture**: Implement Transformer-based Prefix-LM
2. **Training Script**: PyTorch/HuggingFace training loop
3. **Inference Engine**: Text → phonemes → audio tokens → waveform
4. **Evaluation**: Objective metrics and subjective listening tests

## Development Status

**Current Phase**: Data pipeline completion
- ✅ Phoneme extraction with Flite
- ✅ Audio codec integration (X-Codec-2.0)  
- ✅ Training data preparation
- 🔄 Model architecture design
- ⏳ Training implementation
- ⏳ Inference and evaluation

## Repository Structure

```
src/falcon/prefix_tts/
├── run_ljspeech.sh              # Main pipeline script
├── extract_phonemes_simple.sh   # Phoneme extraction
├── prepare_training_data.py     # Dataset preparation
└── codec_extract.py             # Audio tokenization (alternative)
```

---

*This represents a modern approach to neural TTS using language modeling techniques, building on Festvox's traditional strengths while incorporating state-of-the-art generative AI methods.*

