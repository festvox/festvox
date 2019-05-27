### Setup global paths
export FALCONDIR=$FESTVOXDIR/src/falcon

### Prepare data
cat etc/txt.done.data | tr '(' ' ' | tr ')' ' ' | tr '"' ' ' > etc/tddmkdir -p data/dataprep_tacotron1
python3.5 $FALCONDIR/prepare_data.py etc/tdd data/dataprep_tacotron1 wav


### Train Baseline Model
python3.5 $FALCONDIR/train_tacotronone.py --data-root data/dataprep_tacotron1 --checkpoint-dir checkpoints > log_tacotronone

### Test Baseline Model
python3.5 $FALCONDIR/synthesize_tacotronone.py checkpoints/checkpoint_step70000.pth data/dataprep_tacotron1/test.txt test/tacotronone
