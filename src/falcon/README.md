Falcon is a neural extension to festvox voice building suite. 

It is built inspired by [Ryuichi Yamamoto's tacotron repo](https://github.com/r9y9/tacotron_pytorch) and follows the style of [Kaldi](https://github.com/kaldi-asr/kaldi) and [Google Research](https://arxiv.org/ftp/arxiv/papers/1702/1702.01715.pdf)

Extends the native HRG structure in Festival to neural systems. The hirerachy is Layers -> Blocks -> Models

For example,

Conv1d++ class is a layer that enables temporal convolutions during eval.<br>
ResidualDilatedCausalConv1d is a module built on top of Conv1d++ <br>
Wavenet is a model built on top of ResidualDilatedCausalConv1d

Sample [run.sh](https://github.com/festvox/festvox/blob/master/voices/arctic/rms/run.sh) can be found in any of the voices directories.

For the first 18 months, I have decided to track the amount of blocks that are indigenous to us as opposed to those borrowed from the giants. I refer to these as acquisitions. This is inspired by Marissa Mayer's notion of [acquisitions](https://gizmodo.com/heres-what-happened-to-all-of-marissa-mayers-yahoo-acqu-1781980352) at Yahoo.

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

# 20.04 April
#### Acquisitions : 50% (15 blocks out of 30)
#### Experiments

[Vocoder](): WaveGlow 


# 20.05 May
#### Acquisitions : 50% (15 blocks out of 30)
#### Integrating [Judith](http://www.cs.cmu.edu/~srallaba/ProjectAssistCore/) 

[Judith, give me a little juice](https://youtu.be/irkrx-gvqig)


# 20.06 June
#### Acquisitions : 50% (15 blocks out of 30)
#### Integrating Judith. 

[Judith based Lit Review](https://www.youtube.com/watch?v=A_idoFssTjE&list=PLOP55xdQB5RGDGnwUbK9dUn6t-7KuFn_F&index=8)

[Developer Cut](http://www.cs.cmu.edu/~srallaba/ProjectDeveloperCut/)

