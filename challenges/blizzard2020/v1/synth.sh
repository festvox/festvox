
num=$1
python3.5 local/synthesize_vocoder7.py exp/exp_clone7b/checkpoints/checkpoint_step${num}_ema.pth vox/fnames.test.1 exp/exp_clone7b/samples_vocoder
sox exp/exp_clone7b/samples_vocoder/00100010_generated.wav -b 16 /tmp/t.wav
$FESTVOXDIR/src/clustergen/get_cd_dtw exp/exp_clone7b/samples_vocoder/00100010_original.wav /tmp/t.wav
