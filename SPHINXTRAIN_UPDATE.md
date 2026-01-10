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

## do_build Pipeline

Running `./bin/do_build` with no arguments builds the full voice:

```bash
./bin/do_build build_prompts   # Generate prompt utterances
./bin/do_build label           # EHMM alignment (default)
./bin/do_build build_utts      # Build utterances
./bin/do_clustergen parallel build  # Build ClusterGen voice
```

To use SphinxTrain multi-pronunciation instead of EHMM, replace `label` with `label_sphinx`:

```bash
./bin/do_build build_prompts
./bin/do_build label_sphinx    # <-- SphinxTrain instead of EHMM
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

### Stress Recovery

When the aligner selects a variant like `RECORD(2)`, stress can be recovered directly
from the lexicon by calling `(nth 1 (lex.lookup_all "record"))` in Festival. No separate
stress map file is needed - the lexicon entry contains the full syllable structure with stress.

## Individual SphinxTrain Steps

If you need more control, run steps individually:

```bash
./bin/sphinxtrain setup     # Initialize SphinxTrain directory
./bin/sphinxtrain files     # Generate dict with ALL lexicon pronunciations
./bin/sphinxtrain multipron # Copy files for downstream compatibility
./bin/sphinxtrain feats     # Extract MFCC features
./bin/sphinxtrain train     # Train CI acoustic models
./bin/sphinxtrain align     # Run forced alignment
./bin/sphinxtrain labs      # Convert to FestVox format
```

**Note:** The `files` step uses `build_st_multipron.scm` which calls `lex.lookup_all` to get
ALL pronunciations from the lexicon directly. The dictionary is generated in `WORD(n)` format
(e.g., `READ`, `READ(2)`, `READ(3)`). The `multipron` step just copies the files for
compatibility with downstream steps that expect `.multipron.dic`.

## Output Files

| File | Description |
|------|-------------|
| `lab/*.lab` | Phone-level alignments |
| `wrd/*.wrd` | Word-level alignments |
| `st/falignout/*.alignoutput` | Selected pronunciations (WORD(n) format) |
| `st/etc/*.dic` | Dictionary with all lexicon pronunciations |

Stress is recovered from the lexicon directly using `lex.lookup_all` - no separate stressmap file needed.

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
