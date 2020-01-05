Falcon is a neural extension to festvox voice building suite. 

It is built upon [Ryuichi Yamamoto's tacotron repo](https://github.com/r9y9/tacotron_pytorch) and follows coding style of [Kaldi](https://github.com/kaldi-asr/kaldi) 


Layers -> Blocks -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

## Experiments
[Final frame Expt](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/local/train_phones_finalframe.py): Prediction of frame at time t+1 is dependent on only the final predicted frame at time t. [Samples](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/final_frame.html)
