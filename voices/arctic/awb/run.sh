

mkdir -p vox
cd vox
$FESTVOXDIR/src/clustergen/setup_cg cmu us awb

cp /home_original/srallaba/data/arctic/cmu_us_awb_arctic/etc/txt.done.data etc/
./bin/get_wavs /home_original/srallaba/data/arctic/cmu_us_awb_arctic/wav/*

./bin/do_build build_prompts
./bin/do_build label


