source path.sh

mkdir -p vox
cd vox 
$FESTVOXDIR/src/clustergen/setup_cg cmu us multispeaker
cd ..

mkdir -p vox/ehmm/etc
rm -r vox/ehmm/etc/txt.phseq.data
rm -r vox/etc/fnames*
rm -r vox/fnames.*

for spk in awb rms #slt #iitm ljspeech
  do

    spk_dir=${arctic_dir}/${spk}/vox
    echo "Copying data from speaker " ${spk}

    # Copy lspec
    for file in ${spk_dir}/festival/falcon_lspec/*
      do
        fname=$(basename "$file" .feats.npy)
        echo ${spk}_$fname $file >> vox/etc/fnamesNlspec 
      done

    # Copy mspec
    for file in ${spk_dir}/festival/falcon_mspec/*
      do
        fname=$(basename "$file" .feats.npy)
        echo ${spk}_$fname $file >> vox/etc/fnamesNmspec
      done

    # Copy phones
    for file in ${spk_dir}/festival/falcon_phones/*
      do
        fname=$(basename "$file" .feats)
        echo ${spk}_$fname $file >> vox/etc/fnamesNphones
      done

    # Copy txt.phseq.data
    cat ${spk_dir}/ehmm/etc/txt.phseq.data | while read line
      do
         echo ${spk}_${line} >> vox/ehmm/etc/txt.phseq.data
      done

    # Copy fnames*
    cat ${spk_dir}/fnames.train | while read line 
      do 
         echo ${spk}_${line} >> vox/fnames.train
      done
    cat ${spk_dir}/fnames.test | while read line 
      do 
         echo ${spk}_${line} >> vox/fnames.test
      done


  done

  
python3.5 $FALCONDIR/utils/dataprep_addphones.py vox/ehmm/etc/txt.phseq.data vox
python3.5 local/dataprep_addspeakers.py vox/ehmm/etc/txt.phseq.data vox/
