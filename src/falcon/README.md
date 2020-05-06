Falcon is a neural extension to festvox voice building suite. 

It is built inspired by [Ryuichi Yamamoto's tacotron repo](https://github.com/r9y9/tacotron_pytorch) and follows coding style of [Kaldi](https://github.com/kaldi-asr/kaldi) 


Layers -> Blocks -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

Sample [run.sh](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/run.sh) can be found in any of the voices directories.


# 20.01 January
#### Acquisitions : 63% (14 blocks out of 22)
#### Experiments

[Barebones](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/local/train_phones.py): Our barebones implementation can be summarized simply as a clone of [Ryuichi Yamamoto's tacotron repo](https://github.com/r9y9/tacotron_pytorch) but at the level of phoneme sequences. [Checkout the samples](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/tts_phseq.html)

[Final Frame Expt](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/local/train_phones_finalframe.py): Prediction of frame at time t+1 is dependent on only the final predicted frame at time t. [Checkout the samples](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/final_frame.html)

[LSTMsBlock Expt](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/local/train_phones_lstmsblock.py) Replacing CBHGs in Encoder with 3 LSTMs based on [Tacotron 2](https://ai.googleblog.com/2017/12/tacotron-2-generating-human-like-speech.html). Checkout the samples: 
     (1) [replacing CBHG by LSTMsBlock in Encoder only while PostNet still has CBHG](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/lstms_encoder/tts_phseq_lstmsencoder_rms.html) 
      (2) [replacing CBHG by LSTMsBlock in both Encoder and PostNet](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/lstmsblock_encoderNpostnet/lstmsblockencoderNpostnet_rms.html)

[no ssil Expt]() Removing short silences obtained from EHMM alignment. Checkout the [samples](http://tts.speech.cs.cmu.edu/rsk/projects/falcon/exp/no_ssil.html)

# 20.02 February
#### Acquisitions : 58% (14 blocks out of 24)
#### Experiments

[Acoustic Model Baseline](https://github.com/festvox/festvox/blob/master/challenges/blizzard2020/v1/local/train_phones.py) Tokenize Mandarin, convert to pinyin, approximate phonemes using grapheme tools within festvox. [Checkout the samples](http://tts.speech.cs.cmu.edu/rsk/challenges/blizzard2020/exp/baseline.html)

Vocoder: [Checkout the samples](http://tts.speech.cs.cmu.edu/rsk/challenges/blizzard2020/exp/vocoder/baseline.html)

# 20.03 March
#### Acquisitions : 58% (14 blocks out of 24)
#### Experiments

[Vocoder](https://github.com/festvox/festvox/blob/master/challenges/blizzard2020/v1/local/train_quants.py): WaveLSTM with additional FC for fine [Checkout the samples](http://tts.speech.cs.cmu.edu/rsk/challenges/blizzard2020/exp/vocoder_wavernn/wavernn.html)

[Judith](https://youtu.be/irkrx-gvqig)

# 20.04 April
#### Acquisitions : 50% (15 blocks out of 30)
#### Experiments

[Vocoder](): WaveGlow 
