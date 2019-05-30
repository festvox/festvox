### For detailed setup instructions, check the script build_tacotronone.sh

### Setup global paths
export FALCONDIR=$FESTVOXDIR/src/falcon
export VOXDIR=`pwd`

### Prepare data
cat etc/txt.done.data | tr '(' ' ' | tr ')' ' ' | tr '"' ' ' > etc/tdd
python3.5 $FALCONDIR/prepare_data.py etc/tdd $VOXDIR

### Train Baseline Model
python3.5 $FALCONDIR/train_tacotronone.py --data-root $VOXDIR/etc --checkpoint-dir checkpoints_TacotronOneV2 > log_TacotronOneV2

### Test Baseline Model
python3.5 $FALCONDIR/synthesize_tacotronone.py checkpoints/checkpoint_step70000.pth data/dataprep_tacotron1/test.txt test/tacotronone
