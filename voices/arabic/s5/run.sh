#!/usr/bin/sh

filtered_file = '/home1/srallaba/projects/arabic/repos/arabic-tacotron-tts/tacotron_stuff/nawar_without_hag9/temp_filtered.csv'
vox_dir = 'vox'

# Data Preparation
python3.5 local/dataprep_addphones.py ${filtered_file} ${vox_dir}
python3.5 $FALCONDIR/utils/dataprep_addmspec.py ${vox_dir}/fnames ${vox_dir}
python3.5 $FALCONDIR/utils/dataprep_addlspec.py ${vox_dir}/fnames ${vox_dir}

# Train test split. For now, make val set and test set the same
./${vox_dir}/bin/traintest ${vox_dir}/ehmm/etc/txt.phseq.data
cat ${vox_dir}/ehmm/etc/txt.phseq.data | cut -d ' ' -f 1 > ${vox_dir}/fnames
cp ${vox_dir}/fnames.test ${vox_dir}/fnames.val

# Training
python3.5 local/train_phones.py --conf conf/arabic.conf --gpu-id 1 --exp-dir exp/exp_arabic_baseline

# Synthesis
python3.5 local/synthesize_phones.py exp/exp_arabic_baseline/checkpoints/checkpoint_step0140000.pth vox/ehmm/etc/txt.phseq.data.test exp/exp_arabic_baseline/samples_baseline
