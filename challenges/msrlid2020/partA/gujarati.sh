







for file in /home1/srallaba/challenges/msrcodeswitch2020/data/PartA_Gujarati/Train/Audio/*.wav; do fname=$(basename "$file" .wav); echo $fname $file; done > data/train_gujarati/wav.scp
cut -d' ' -f 1 data/train_gujarati/wav.scp > data/train_gujarati/text
paste -d' ' data/train_gujarati/text data/train_gujarati/text > data/train_gujarati/utt2spk
./utils/utt2spk_to_spk2utt.pl data/train_gujarati/utt2spk > data/train_gujarati/spk2utt
./utils/fix_data_dir.sh data/train_gujarati/



for file in /home1/srallaba/challenges/msrcodeswitch2020/data/PartA_Gujarati/Dev/Audio/*.wav; do fname=$(basename "$file" .wav); echo $fname $file; done > data/dev_gujarati/wav.scp
cut -d' ' -f 1 data/dev_gujarati/wav.scp > data/dev_gujarati/text
paste -d' ' data/dev_gujarati/text data/dev_gujarati/text > data/dev_gujarati/utt2spk
./utils/utt2spk_to_spk2utt.pl data/dev_gujarati/utt2spk > data/dev_gujarati/spk2utt
./utils/fix_data_dir.sh data/dev_gujarati/




