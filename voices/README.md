# Building a vanilla voice


### Sample build

### Building a vanilla phone level model

We follow a three step procedure: Data preparation, Training and Testing

#### Data Preparation

We do the following:

1) Phones preparation
2) Acoustic Feature Extraction. We extract mel and linear spectra as acoustic features.

The idea is to use the file 'txt.phseq.data'. Then we can iterate through the files and extract linear and mel spectra. 

```text
python3.5 $FALCONDIR/utils/dataprep_addphones.py ehmm/etc/txt.phseq.data vox 
```

#### Training

```text
python3.5 local/train_phones.py --exp-dir exp/exp_tacotron_phones > log_phones 2>&1&
```

This step will create a directory called exp, a subdirectory called exp_tacotron_phones. This sub directroy will house
all the information about training. For now, it contains three folders: checkpoints, tracking and samples. Checkpoints is to store 
the checkpoints during training. The Frequency of checkpoints can be modified in the file $FALCONDIR/hyperparameters. The folder 'tracking'
will log the loss value and other information. 'samples' is intended for synthesized samples

#### Testing

```text
python3.5 local/synthesize_phones.py exp/exp_tacotron_phones/checkpoints/checkpoint_step0025000.pth ehmm/etc/txt.phseq.data.test exp/exp_tacotron_phones/samples 
```

[Samples from voices](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/tts_phseq.html)

## 

