
source path.sh

VOXDIR=vox
spk=rms

mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu us ${spk} || exit 1

cp ${data_dir}/arctic/cmu_us_${spk}_arctic/etc/txt.done.data etc/
./bin/get_wavs ${data_dir}/arctic/cmu_us_${spk}_arctic/wav/*

./bin/do_build build_prompts || exit 1
./bin/do_build get_phseq || exit 1
cd ..

python3.5 $FALCONDIR/utils/dataprep_addphones.py ${VOXDIR}/ehmm/etc/txt.phseq.data ${VOXDIR}
cat ${VOXDIR}/ehmm/etc/txt.phseq.data | awk '{print $1}' > ${VOXDIR}/fnames
python3.5 $FALCONDIR/utils/dataprep_addlspec.py ${VOXDIR}/fnames ${VOXDIR}
python3.5 $FALCONDIR/utils/dataprep_addmspec.py ${VOXDIR}/fnames ${VOXDIR}

${VOXDIR}/bin/traintest ${VOXDIR}/fnames 
cp ${VOXDIR}/fnames.test ${VOXDIR}/fnames.val

python3.5 local/train_phones.py --exp-dir exp/taco_one_phseq --preset conf/tacotron.conf
