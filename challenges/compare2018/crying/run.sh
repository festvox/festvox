VOXDIR=vox
spk=crying
data_dir=/home2/srallaba/challenges/compare2018/data/ComParE2018_Crying/
feats_dir=/home2/srallaba/challenges/compare2018/feats/CRIED_soundnet/

# Make voice directory
mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu us ${spk} || exit 1
./bin/get_wavs ${data_dir}/wav/* || exit 1

# Prepare features and labels
python3.5 local/dataprep_addlabels.py ${data_dir}/lab/ComParE2018_Crying.tsv  vox/
python3.5 local/copy_soundnet.py ${feats_dir} vox/

# Get dev set
${VOXDIR}/bin/traintest ${VOXDIR}/fnames.train

# Train and validate
python3.5 local/train_baseline.py --conf conf/crying.conf --gpu-id 0 --exp-dir exp/exp_baseline  # Writes progress to exp/exp_baseline
