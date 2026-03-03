# Prefix-LM TTS: Text-to-Audio Token Generation for FestVox/Flite Integration

## Project Overview

This project implements a novel Text-to-Speech (TTS) system that treats speech synthesis as a prefix language modeling task. By representing both text and audio as discrete tokens in a unified sequence, we can leverage the power of causal Transformers to generate high-quality speech directly from text input.

## Objective

Build a simple, controllable, and exportable TTS pipeline with the following key characteristics:

- **Unified Token Representation**: Treat speech synthesis as sequence-to-sequence generation where the input sequence contains both text/phoneme tokens and audio tokens
- **Prefix Language Modeling**: Use a causal Transformer to predict the next token across the entire sequence
- **Direct Audio Generation**: Generate discrete audio tokens that can be decoded to waveform without intermediate mel-spectrogram representations
- **Production Ready**: Export the trained model to ONNX format for integration with FestVox/Flite as a neural voice pack

## Architecture Design

### Core Concept
```
Input Sequence:  [text/phoneme tokens] + [audio tokens]
Model:          Causal Transformer Language Model
Output:         Autoregressive audio token generation
```

### System Components

#### 1. Frontend Processing (FestVox/Flite Integration)
- **Text Normalization**: Convert raw text to normalized form
- **Phoneme Generation**: Transform text to phonemes using ARPAbet notation
- **Token Encoding**: Map phonemes to integer IDs for model input

#### 2. Audio Codec
- **Pretrained Neural Codec**: Bidirectional conversion between waveform and discrete audio tokens
- **Compression**: Efficient representation of audio information in token format
- **Quality Preservation**: Maintain audio fidelity through learned representations

#### 3. Language Model
- **Architecture**: Causal Transformer operating on unified token vocabulary
- **Training**: Next-token prediction across concatenated text and audio sequences
- **Inference**: Autoregressive generation of audio tokens given text prefix

#### 4. Decoder/Vocoder
- **Codec Decoder**: Convert generated audio tokens back to waveform
- **No Mel-Spectrogram**: Direct token-to-waveform conversion eliminating traditional vocoder training

#### 5. Runtime System
- **ONNX Export**: Model exported to ONNX format for cross-platform deployment
- **Flite Integration**: Bridge between ONNX model and FestVox/Flite framework
- **Manifest System**: Configuration and metadata management for voice packs

## Technical Advantages

1. **Simplicity**: Unified token representation eliminates complex multi-stage pipelines
2. **Controllability**: Prefix conditioning allows for fine-grained control over synthesis
3. **Efficiency**: Single model handles both linguistic and acoustic modeling
4. **Exportability**: ONNX format ensures broad compatibility and deployment flexibility
5. **Quality**: Leverages state-of-the-art language modeling techniques for audio generation

## Implementation Pipeline

### Training Phase
1. Prepare paired text-audio datasets
2. Extract phoneme sequences using FestVox/Flite frontend
3. Encode audio using pretrained neural codec
4. Train causal Transformer on concatenated sequences
5. Validate generation quality and controllability

### Deployment Phase
1. Export trained model to ONNX format
2. Package codec weights and model manifest
3. Integrate with FestVox/Flite runtime
4. Test end-to-end synthesis pipeline

## Expected Outcomes

- High-quality speech synthesis comparable to state-of-the-art TTS systems
- Reduced complexity compared to traditional multi-stage approaches
- Seamless integration with existing FestVox/Flite infrastructure
- Efficient inference suitable for real-time applications
- Flexible voice pack system for easy model distribution and deployment
