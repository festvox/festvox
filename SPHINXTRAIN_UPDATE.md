# SphinxTrain Multi-Pronunciation Alignment

*January 2026*

The `sphinxtrain` script supports modern SphinxTrain (5.x) with multi-pronunciation
alignment and automatic silence insertion.

## Quick Start: Build Arctic SLT

```bash
# Set environment
export ESTDIR=/path/to/speech_tools
export FESTVOXDIR=/path/to/festvox
export SPHINXTRAINDIR=/path/to/sphinxtrain

# Download and unpack Arctic SLT
wget http://www.festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2
tar xjf cmu_us_slt_arctic-0.95-release.tar.bz2

# Set up voice directory
mkdir cmu_us_slt_arctic_cg
cd cmu_us_slt_arctic_cg
$FESTVOXDIR/src/clustergen/setup_cg cmu us slt_arctic

# Copy data
cp -p ../cmu_us_slt_arctic/wav/*.wav wav/
cp -p ../cmu_us_slt_arctic/etc/txt.done.data etc/

# Build everything (use label_sphinx instead of default label)
./bin/do_build build_prompts
./bin/do_build label_sphinx
./bin/do_build build_utts
./bin/do_clustergen parallel build

# Test
festival festvox/cmu_us_slt_arctic_cg.scm
festival> (voice_cmu_us_slt_arctic_cg)
festival> (SayText "Hello world.")
```

## do_build Commands

| Command | Description |
|---------|-------------|
| `./bin/do_build` | Run full pipeline (uses EHMM labeling) |
| `./bin/do_build build_prompts` | Generate Festival prompt utterances |
| `./bin/do_build label_sphinx` | Run SphinxTrain alignment (replaces `label`) |
| `./bin/do_build label` | Run EHMM alignment (default) |
| `./bin/do_build build_utts` | Build utterances from alignments |
| `./bin/do_clustergen parallel build` | Build ClusterGen voice (all features) |

To use SphinxTrain instead of EHMM, replace `./bin/do_build` with:
```bash
./bin/do_build build_prompts
./bin/do_build label_sphinx
./bin/do_build build_utts
./bin/do_clustergen parallel build
```

## What Multi-Pronunciation Does

### Acoustic Pronunciation Selection

When words have multiple pronunciations, SphinxTrain picks the one matching the audio:

```
READ    r eh d   # past tense
READ(2) r iy d   # present tense
```

### Automatic Silence Insertion

Natural pauses are detected and marked with `<sil>`:

```
ROBBERY <sil> BRIBERY FRAUD
```

### Stress Mapping

Syllable stress is recorded for each variant:

```
RECORD    (1 0)   # REcord (noun)
RECORD(2) (0 1)   # reCORD (verb)
```

## Individual SphinxTrain Steps

If you need more control, run steps individually:

```bash
./bin/sphinxtrain setup     # Initialize SphinxTrain directory
./bin/sphinxtrain files     # Generate dict, phones, transcription
./bin/sphinxtrain multipron # Convert to WORD(n) format
./bin/sphinxtrain feats     # Extract MFCC features
./bin/sphinxtrain train     # Train CI acoustic models
./bin/sphinxtrain align     # Run forced alignment
./bin/sphinxtrain labs      # Convert to FestVox format
```

## Output Files

| File | Description |
|------|-------------|
| `lab/*.lab` | Phone-level alignments |
| `wrd/*.wrd` | Word-level alignments |
| `st/falignout/*.alignoutput` | Selected pronunciations (WORD(n) format) |
| `st/etc/*.stressmap` | Stress patterns for variants |

## Requirements

SphinxTrain 5.x built with cmake:

```bash
cd $SPHINXTRAINDIR
cmake -S . -B build
cmake --build build
```

Verify:

```bash
ls $SPHINXTRAINDIR/build/sphinx3_align
ls $SPHINXTRAINDIR/scripts/10.falign_ci_hmm/slave_convg.pl
```

## Troubleshooting

**"train" step fails or does nothing:**

1. Check `SPHINXTRAINDIR` is set correctly
2. Verify `$SPHINXTRAINDIR/scripts/10.falign_ci_hmm/slave_convg.pl` exists
3. Debug with: `bash -x bin/sphinxtrain train`

**"No mdef-file" error during alignment:**

The train step didn't create acoustic models. Check for errors in training output.
