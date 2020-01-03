Falcon is a neural extension to festvox voice building suite. 

It is inspired by [Ryuichi Yamamoto's tacotron repo](https://github.com/r9y9/tacotron_pytorch) and follows coding style of [Kaldi](https://github.com/kaldi-asr/kaldi) 


Layers -> Blocks -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

