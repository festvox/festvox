
num=$1
python3.5 local/synthesize_vocoder8.py --logits-dim 60 exp/exp_clone8/checkpoints/checkpoint_step${num}_ema.pth vox/ehmm/etc/txt.phseq.data.test.head.1 exp/exp_clone8/samples_vocoder || exit 1
sox exp/exp_clone8/samples_vocoder/02200092_generated.wav -b 16 /tmp/t.wav || exit 1
$FESTVOXDIR/src/clustergen/get_cd_dtw exp/exp_clone8/samples_vocoder/02200092_original.wav /tmp/t.wav


