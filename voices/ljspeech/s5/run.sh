VOXDIR=vox
spk=ljspeech

source path.sh

mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu us ${spk} || exit 1

cp ${data_dir}/LJSpeech-1.1/txt.done.data etc/ || exit 1
./bin/get_wavs ${data_dir}/LJSpeech-1.1/wavs/* || exit 1

./bin/do_build build_prompts || exit 1
./bin/do_build get_phseq || exit 1
cd ..

exit 1

python3.5 $FALCONDIR/utils/dataprep_addphones.py ${VOXDIR}/ehmm/etc/txt.phseq.data ${VOXDIR}
cat ${VOXDIR}/ehmm/etc/txt.phseq.data | awk '{print $1}' > ${VOXDIR}/fnames
python3.5 $FALCONDIR/utils/dataprep_addlspec.py ${VOXDIR}/fnames ${VOXDIR}
python3.5 $FALCONDIR/utils/dataprep_addmspec.py ${VOXDIR}/fnames ${VOXDIR}

${VOXDIR}/bin/traintest ${VOXDIR}/fnames 
cp ${VOXDIR}/fnames.test ${VOXDIR}/fnames.val

## Sort based on lengths and build a model for small utterances
cat ${VOXDIR}/fnames.train | while read fname; 
  do
      duration=`soxi -d ${VOXDIR}/wav/$fname.wav`
      echo $fname $duration 
  done | sort -k2 > ${VOXDIR}/etc/fnamesNdurations
#head -600 ${VOXDIR}/etc/fnamesNdurations > ${VOXDIR}/fnames.train


#python3.5 local/train_phones.py --exp-dir exp/taco_one_phseq 
#python3.5 local/synthesize_phones.py exp/taco_one_phseq/checkpoints/checkpoint_step30000.pth  vox/ehmm/etc/txt.phseq.data.test.head exp/taco_one_phseq/tts_phseq
