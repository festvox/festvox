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

## Results (LJSpeech)

| Model | Decoder | Gen Total | WER (%) | UTMOS |
|-------|---------|-----------|---------|-------|
| VITS (Coqui, 25K) | ~33M | 36.3M | 7.3 | 3.60 |
| Mini-MB-iSTFT-VITS (45K) | 3.62M | 9.07M | 19 | 3.08 |
| MCEP-VITS (10K) | 3.9M | ~9.3M | TBD | TBD |

## Usage

### Training

```bash
# Single GPU
python train_latest.py -c configs/ljs_mcep_vits.json -m OUTPUT_DIR

# Multi-GPU (DDP)
python -m torch.distributed.run --nproc_per_node=N train_latest.py \
    -c configs/ljs_mcep_vits.json -m OUTPUT_DIR
```

Training logs to wandb automatically. The run name is derived from the config filename.

### Data Setup

1. Download LJSpeech-1.1 to a local directory
2. Update the `training_files` and `validation_files` paths in the config JSON
3. Filelists are provided in `filelists/` (phoneme-cleaned format)

### Building Monotonic Alignment Search

```bash
cd monotonic_align && python setup.py build_ext --inplace
```

Pre-compiled `core.c` is included for environments without Cython.

## Dependencies

- PyTorch >= 1.13
- pysptk (for MCEP basis matrix computation)
- scipy
- librosa
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
