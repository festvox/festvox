# MCEP-VITS: Mel-Cepstral Decoder for VITS

Based on [VITS](https://github.com/jaywalnut310/vits) by Jaehyeon Kim et al. (MIT License).
The text encoder, posterior encoder, flow, discriminator, monotonic alignment search, and
training framework are derived from the original VITS implementation. The decoder is replaced
with a novel MCEP-based architecture (see below).

MCEP-VITS replaces the standard convolutional upsampling decoder in VITS with a compact
mel-cepstral (MCEP) magnitude predictor + minimum phase + learned phase residual, followed
by a single fullband iSTFT. Instead of upsampling the latent representation through
transposed convolutions and ResBlocks (3.6M params in Mini-MB-iSTFT-VITS), MCEP-VITS
uses a deep ResBlock backbone to predict mel-cepstral coefficients that are converted to
a spectral envelope via a fixed basis matrix, adds a bounded magnitude refinement, then
combines with a minimum-phase prior + learned phase residual to synthesize the waveform
through iSTFT.

## Quick Start

### 1. Install dependencies

```bash
pip install torch torchaudio pysptk scipy librosa unidecode soundfile wandb Cython
```

You also need [flite](http://www.festvox.org/flite/) installed and on your PATH for G2P:

```bash
# Ubuntu/Debian
sudo apt-get install flite

# Or build from source
git clone https://github.com/festvox/flite.git
cd flite && ./configure && make && sudo make install
```

### 2. Download LJSpeech

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjf LJSpeech-1.1.tar.bz2
```

### 3. Update filelists

The filelists in `filelists/` use `DUMMY1/` as a placeholder path. Replace it with the
actual path to your LJSpeech wavs directory:

```bash
cd src/falcon/mcep_vits

# Replace DUMMY1 with your LJSpeech wav path in all filelists
sed -i 's|DUMMY1|/path/to/LJSpeech-1.1/wavs|g' filelists/ljs_audio_text_train_filelist.txt
sed -i 's|DUMMY1|/path/to/LJSpeech-1.1/wavs|g' filelists/ljs_audio_text_val_filelist.txt
sed -i 's|DUMMY1|/path/to/LJSpeech-1.1/wavs|g' filelists/ljs_audio_text_test_filelist.txt
```

### 4. Preprocess text (flite G2P)

Generate the `.cleaned` filelists with phoneme sequences:

```bash
python preprocess.py \
    --filelists filelists/ljs_audio_text_train_filelist.txt \
                filelists/ljs_audio_text_val_filelist.txt \
                filelists/ljs_audio_text_test_filelist.txt \
    --text_cleaners flite_cleaners
```

This runs flite G2P on each utterance, producing space-delimited ARPAbet phone sequences
(42 phones, stress stripped). Output files are written as `*.txt.cleaned`.

### 5. Build Monotonic Alignment Search

```bash
cd monotonic_align && python setup.py build_ext --inplace && cd ..
```

Pre-compiled `core.c` is included for environments without Cython.

### 6. Train

```bash
# Single GPU
python train_latest.py -c configs/ljs_mcep_vits.json -m OUTPUT_DIR

# Multi-GPU (DDP)
python -m torch.distributed.run --nproc_per_node=N train_latest.py \
    -c configs/ljs_mcep_vits.json -m OUTPUT_DIR
```

Training logs to wandb automatically. The run name is derived from the config filename.

Default config uses batch_size=48 and fp32 (fp16 can cause CUDA assertion failures on
some PyTorch/CUDA versions). On an RTX 4090, this uses ~16.5 GB VRAM.

### 7. Resume from checkpoint

Checkpoints are saved as `G_STEPS.pth` and `D_STEPS.pth` in the output directory.
To resume, just re-run the same training command — it automatically picks up the latest
checkpoint from OUTPUT_DIR.

## Architecture

```
z [B, C, T]
  -> project: Conv1d(C, D, k=1)
  -> 5x ResBlock1(D, k=3, dilation=(1,3,5))   (deep backbone, RF > 100 frames)
  -> pad T -> T+1 (for iSTFT center=True)
  |
  +-- MCEP path:
  |     mcep_head(D, 40, k=1) -> MCEPBasis -> mcep_log_mag
  |
  +-- Magnitude refinement:
  |     mag_conv(D, D, k=3) + LReLU -> mag_head(D, 513, k=1)
  |     log_mag = mcep_log_mag + 0.3 * tanh(mag_refine)
  |     mag = exp(log_mag)
  |
  +-- Phase path:
        phase_conv1(D, D, k=3) + LReLU
        phase_conv2(D, D, k=3) + LReLU
        phase_head(D, 513, k=1)
        min_phase = compute_min_phase(log_mag)
        phase = min_phase + pi * sin(phase_residual)
  |
  iSTFT(n_fft=1024, hop=256) -> waveform
```

The mc2sp basis matrix is precomputed from pysptk and registered as a non-trainable buffer.
It maps 40 MCEPs to 513 log-magnitude frequency bins via a fixed linear transform.

Decoder params: ~3.9M (with D=192, 5 resblocks). Total generator: ~9.3M.

## Text Frontend

Uses [flite](http://www.festvox.org/flite/) for grapheme-to-phoneme conversion. The 42-phone
ARPAbet inventory (stress stripped) is defined in `text/symbols.py`. Phoneme sequences are
space-delimited (not character-by-character like IPA-based VITS).

Pipeline: raw text -> expand abbreviations -> ASCII transliteration -> flite G2P -> ARPAbet phones

## Results (LJSpeech)

| Model | Decoder | Gen Total | WER (%) | UTMOS |
|-------|---------|-----------|---------|-------|
| VITS (Coqui, 25K) | ~33M | 36.3M | 7.3 | 3.60 |
| Mini-MB-iSTFT-VITS (45K) | 3.62M | 9.07M | 19 | 3.08 |
| MCEP-VITS v8 (CG, 105K) | 3.9M | ~9.3M | 0-8 | 3.36 |
| MCEP-VITS v9 (CG, large dec) | 20.4M | 41.4M | TBD | TBD |

## Dependencies

- PyTorch >= 1.13
- flite (system binary, for G2P)
- pysptk (for MCEP basis matrix computation)
- scipy
- librosa
- unidecode
- soundfile
- Cython (optional, for building monotonic_align from .pyx)
- wandb (optional, for experiment tracking)

## File Structure

- `mcep_decoder.py` -- MCEPBasis, MCEPDecoder, compute_min_phase
- `models.py` -- SynthesizerTrn (main model), generators, discriminators
- `modules.py` -- ResBlock1/2, WaveNet
- `attentions.py` -- Transformer encoder attention
- `train_latest.py` -- Training loop with DDP, GAN loss, wandb logging
- `data_utils.py` -- TextAudioLoader, collate, distributed sampler
- `text/` -- Phoneme tokenization (cleaners, symbols)
- `monotonic_align/` -- Monotonic alignment search (Cython)
- `configs/` -- JSON config for LJSpeech
- `filelists/` -- LJSpeech train/val/test splits
