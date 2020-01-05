import numpy as np

import torch
from torch.autograd import Variable
from torch import nn

from blocks import *
from layers  import *

'''Excerpts from the following sources
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py

'''

print_flag = 0


class TacotronOne(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOne, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        self.embedding = nn.Embedding(n_vocab, embedding_dim,
                                      padding_idx=padding_idx)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder_TacotronOne(embedding_dim)
        self.decoder = Decoder_TacotronOne(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    '''
    Section 4 in 'TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS' https://arxiv.org/pdf/1703.10135.pdf
      Its a common practice to train sequence models with a loss mask, which masks loss on zero-padded frames.
      However, we found that models trained this way dont know when to stop emitting outputs, causing
      repeated sounds towards the end. One simple trick to get around this problem is to also reconstruct
      the zero-padded frames.
    '''
    def forward(self, inputs, targets=None, input_lengths=None):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        encoder_outputs = self.encoder(inputs, input_lengths)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments




class TacotronOneSeqwise(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwise, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)

class TacotronOneFinalFrame(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneFinalFrame, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneFinalFrame(mel_dim, r)

