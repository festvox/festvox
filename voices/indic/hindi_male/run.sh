
source path.sh

VOXDIR=vox
spk=hin_male



mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg_indic cmu us hin ${spk} || exit 1

cp ${data_dir}/indic/cmu_indic_${spk}/etc/txt.done.data etc/ || exit 1
./bin/get_wavs ${data_dir}/indic/cmu_indic_${spk}/wav/* || exit 1

./bin/do_build build_prompts || exit 1
./bin/do_build get_phseq || exit 1
cd ..


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
head -600 ${VOXDIR}/etc/fnamesNdurations > ${VOXDIR}/fnames.train

## Train a neural barebones model
python3.5 local/train_phones.py --exp-dir exp/taco_one_phseq 
