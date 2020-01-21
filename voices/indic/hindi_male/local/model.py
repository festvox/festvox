import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *

class Decoder_TacotronOneSeqwise(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneSeqwise, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim * r, sizes=[256, 128])


class TacotronOneSeqwise(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwise, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)

class TacotronOneSeqwiseqF0s(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_qF0s=20, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseqF0s, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.qF0_embedding = nn.Embedding(num_qF0s, 128)
        self.embeddings2inputs = SequenceWise(nn.Linear(embedding_dim + 128, embedding_dim))


    def forward(self, inputs, qF0s, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)

        # Embeddings for quantized F0s
        qF0s_embedding = self.qF0_embedding(qF0s)

        # Combination
        inputs = torch.cat([inputs, qF0s_embedding], dim=-1)
        inputs = self.embeddings2inputs(inputs) 

        # Text Encoder
        encoded_phonemes = self.encoder(inputs, input_lengths)
        decoder_inputs = encoded_phonemes

        # Decoder
        mel_outputs, alignments = self.decoder(decoder_inputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


class TacotronOneSeqwiseStress(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_qF0s=2, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseStress, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.qF0_embedding = nn.Embedding(num_qF0s, 128)
        self.embeddings2inputs = SequenceWise(nn.Linear(embedding_dim + 128, embedding_dim))


    def forward(self, inputs, qF0s, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)

        # Embeddings for quantized F0s
        qF0s_embedding = self.qF0_embedding(qF0s)

        # Combination
        inputs = torch.cat([inputs, qF0s_embedding], dim=-1)
        inputs = self.embeddings2inputs(inputs) 

        # Text Encoder
        encoded_phonemes = self.encoder(inputs, input_lengths)
        decoder_inputs = encoded_phonemes

        # Decoder
        mel_outputs, alignments = self.decoder(decoder_inputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments



class TacotronOneLSTMsBlockStress(TacotronOneLSTMsBlock):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_qF0s=2, padding_idx=None, use_memory_mask=False):
        super(TacotronOneLSTMsBlockStress, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.qF0_embedding = nn.Embedding(num_qF0s, 128)
        self.embeddings2inputs = SequenceWise(nn.Linear(embedding_dim + 128, embedding_dim))


    def forward(self, inputs, qF0s, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)

        # Embeddings for quantized F0s
        qF0s_embedding = self.qF0_embedding(qF0s)

        # Combination
        inputs = torch.cat([inputs, qF0s_embedding], dim=-1)
        inputs = self.embeddings2inputs(inputs) 

        # Text Encoder
        encoded_phonemes = self.encoder(inputs, input_lengths)
        decoder_inputs = encoded_phonemes

        # Decoder
        mel_outputs, alignments = self.decoder(decoder_inputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


