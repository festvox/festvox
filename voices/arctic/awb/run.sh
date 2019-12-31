VOXDIR=vox
spk=awb

mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu us ${spk} || exit 1

cp /home_original/srallaba/data/arctic/cmu_us_${spk}_arctic/etc/txt.done.data etc/
./bin/get_wavs /home_original/srallaba/data/arctic/cmu_us_${spk}_arctic/wav/*

./bin/do_build build_prompts || exit 1
./bin/do_build label || exit 1


