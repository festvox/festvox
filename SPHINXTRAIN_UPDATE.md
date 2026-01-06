# SphinxTrain Multi-Pronunciation Alignment Update

*January 2026*

## Overview

The `sphinxtrain` script in FestVox has been updated to support modern SphinxTrain (5.x) with multi-pronunciation alignment and automatic silence insertion. The script auto-detects available SphinxTrain versions and falls back to legacy Sphinx2 if needed.

## What's New

### 1. Modern SphinxTrain Support
- Uses `sphinx3_align` instead of deprecated `sphinx2-batch`
- Uses `sphinx_fe` for feature extraction
- Uses Python `sphinxtrain` setup script

### 2. Multi-Pronunciation Alignment
The aligner can now choose between alternate pronunciations based on acoustics:

```
READ  r eh d     # past tense "red"
READ(2)  r iy d  # present tense "reed"
```

When aligning "I tried to READ George Moore", SphinxTrain acoustically selects `READ(2)` because the speaker says "reed".

### 3. Automatic Silence Insertion
SphinxTrain detects natural pauses and inserts `<sil>` markers:

```
ROBBERY <sil> BRIBERY FRAUD
THE GIRL FACED HIM <sil> HER EYES SHINING
```

### 4. Stress Correction
A stress map is generated to correct syllable stress when a different pronunciation variant is selected:

```
RECORD (1 0)     # noun: REcord
RECORD(2) (0 1)  # verb: reCORD
```

If Festival predicted "REcord" but the speaker said "reCORD", the stress can be corrected during alignment merge.

## Usage

### Default Pipeline (Multi-Pronunciation)

```bash
export SPHINXTRAINDIR=/path/to/sphinxtrain
export ESTDIR=/path/to/speech_tools
export FESTVOXDIR=/path/to/festvox

# Run full pipeline
bin/sphinxtrain
```

This runs: `setup → files → multipron → feats → train → align → labs`

### Individual Steps

```bash
bin/sphinxtrain setup     # Initialize SphinxTrain directory
bin/sphinxtrain files     # Generate dict, phones, transcription
bin/sphinxtrain multipron # Convert to WORD(n) format
bin/sphinxtrain feats     # Extract MFCC features
bin/sphinxtrain train     # Train CI acoustic models
bin/sphinxtrain align     # Run forced alignment
bin/sphinxtrain labs      # Convert to FestVox format
```

### Legacy Mode (Sphinx2)

If `SPHINX2DIR` is set and `sphinx3_align` is not found, the script falls back to legacy Sphinx2 alignment automatically.

## Output Files

| File | Description |
|------|-------------|
| `lab/*.lab` | Phone-level alignments (FestVox format) |
| `wrd/*.wrd` | Word-level alignments |
| `st/falignout/*.alignoutput` | Selected pronunciations with WORD(n) notation |
| `st/etc/*.stressmap` | Stress patterns for each pronunciation variant |
| `st/phseg/*.phseg` | Raw phone segmentation from SphinxTrain |
| `st/wdseg/*.wdseg` | Raw word segmentation from SphinxTrain |

## Technical Details

### Dictionary Format Conversion

The `multipron` step converts Festival's format to SphinxTrain's:

```
# Festival/build_st.scm generates:
READ  r eh d
READ2  r iy d

# multipron step converts to:
READ  r eh d
READ(2)  r iy d
```

### Transcript Format

```
# Original (with pre-resolved pronunciations):
<s> I TRIED TO READ2 GEORGE MOORE </s> (arctic_a0479)

# Multi-pron (base words only):
<s> I TRIED TO READ GEORGE MOORE </s> (arctic_a0479)

# Alignment output (acoustically selected):
<s> I TRIED TO READ(2) GEORGE MOORE </s> (arctic_a0479)
```

### Stress Map Format

```
WORD (stress1 stress2 ...)
```

Examples:
```
FRAGMENTS (1 0)      # FRAGments
FRAGMENTS(2) (0 1)   # fragMENTS
PROGRESS (1 1)       # PROgress
PROGRESS(2) (0 1)    # proGRESS
A (0)                # schwa (unstressed)
A(2) (1)             # letter name (stressed)
```

## Files Changed

| File | Change |
|------|--------|
| `src/st/sphinxtrain` | Updated with modern SphinxTrain support, multi-pron |
| `src/st/build_st_multipron.scm` | New: Generate WORD(n) format files |
| `src/st/build_stress_map.scm` | New: Extract stress patterns |
| `src/st/align_with_stress.scm` | New: Merge with stress correction |

## Requirements

### Modern Mode (Recommended)
- SphinxTrain 5.x with `sphinx3_align` built
- Python 3 for `sphinxtrain` setup script

```bash
cd sphinxtrain
cmake -S . -B build
cmake --build build
```

### Legacy Mode (Fallback)
- Sphinx2 with `sphinx2-batch`
- Set `SPHINX2DIR` environment variable

## Example Results

### Arctic SLT Test (1132 utterances)

| Metric | Value |
|--------|-------|
| Files aligned | 1132 / 1132 (100%) |
| Utterances with variant pronunciations | 61 |
| Utterances with silence insertion | 225 |
| Stress map entries | 2,985 |

### Pronunciation Selection Examples

| Utterance | Word | Selected | Phones |
|-----------|------|----------|--------|
| arctic_a0479 | READ | READ(2) | r iy d |
| arctic_b0535 | READ | READ | r eh d |
| arctic_a0226 | A | A(2) | ey |

## Known Limitations

1. **Dictionary coverage**: Words not in the lexicon fall back to a single pronunciation
2. **Compound words**: May not have all variant forms
3. **Proper names**: Limited pronunciation variants

## Future Work

- Integrate stress correction into `align_utt` during UTT merge
- Add support for G2P fallback for OOV words
- Improve compound word handling
