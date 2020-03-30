source path.sh

mkdir -p ehmm/etc
rm -r ehmm/etc/txt.phseq.data
rm -r etc/fnames*
rm -r ehmm/etc/fnames*

for spk in awb rms #slt #iitm ljspeech
  do

    spk_dir='/home/srallaba/projects/text2speech/voices/cmu_us_'${spk}
    echo "Copying data from speaker " ${spk}

    # Copy lspec
    mkdir -p festival/falcon_lspec
    for file in ${spk_dir}/festival/falcon_lspec/*
      do
        fname=$(basename "$file")
        cp $file festival/falcon_lspec/${spk}_${fname}
      done

    # Copy mspec
    mkdir -p festival/falcon_mspec
    for file in ${spk_dir}/festival/falcon_mspec/*
      do
        fname=$(basename "$file")
        cp $file festival/falcon_mspec/${spk}_${fname}
      done

    # Copy phones
    mkdir -p festival/falcon_phones
    for file in ${spk_dir}/festival/falcon_phones/*
      do
        fname=$(basename "$file")
        cp $file festival/falcon_phones/${spk}_${fname}
      done

    # Copy txt.phseq.data
    cat  ${spk_dir}/ehmm/etc/txt.phseq.data | while read line
      do
         echo ${spk}_${line} >> ehmm/etc/txt.phseq.data
      done

  done

cat ehmm/etc/txt.phseq.data | awk '{print $1}' > fnames
./bin/traintest fnames
cp fnames.test fnames.val

# Edit etc/falcon_feats.desc
