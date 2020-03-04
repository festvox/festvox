#!/bin/bash

#### Create Stuff
mkdir -p data/train data/test
mkdir -p local

#### Copy stuff
ln -s ../../wsj/s5/steps/
ln -s ../../wsj/s5/utils/
cp ../../wsj/s5/cmd.sh .
cp ../../wsj/s5/path.sh .
cp -r ../../wsj/s5/conf/ .
cp ../../voxforge/s5/conf/decode.config conf
cp ../../yesno/s5/local/score.sh local/
cp -r ../../gale_arabic/s5/local/nnet local

#### Source stuff
. ./cmd.sh
. ./path.sh

# Prepare Train
for file in /home1/srallaba/challenges/msrcodeswitch2020/data/PartA_Telugu/Train/Audio/*.wav; do fname=$(basename "$file" .wav); echo $fname $file; done > data/train/wav.scp
cut -d' ' -f 1 data/train/wav.scp > data/train/text
paste -d' ' data/train/text data/train/text > data/train/utt2spk
./utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
./utils/fix_data_dir.sh data/train/

# Prepare Dev
for file in /home1/srallaba/challenges/msrcodeswitch2020/data/PartA_Telugu/Dev/Audio/*.wav; do fname=$(basename "$file" .wav); echo $fname $file; done > data/dev/wav.scp
cut -d' ' -f 1 data/dev/wav.scp > data/dev/text
paste -d' ' data/dev/text data/dev/text > data/dev/utt2spk
./utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
./utils/fix_data_dir.sh data/dev

# Extract features
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 100 --cmd "run.pl" data/train exp/mfcc mfcc_train
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 100 --cmd "run.pl" data/dev exp/mfcc mfcc_dev

