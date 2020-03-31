import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *


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

class TacotronOneSeqwiseMultispeaker(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_spk=2, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseMultispeaker, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.num_spk = num_spk
        self.spk_embedding = nn.Embedding(self.num_spk, 128)

        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.phonesNspk2embedding = SequenceWise(nn.Linear(embedding_dim + 128, embedding_dim)) 

        #self.decoder = Decoder_MultiSpeakerTacotronOne(mel_dim, r)

    def forward(self, inputs, spk, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)
        spk_embedding = self.spk_embedding(spk)
        spk_embedding = spk_embedding.unsqueeze(1).expand(-1, inputs.size(1), -1)

        # Text + Speaker
        inputs = torch.cat([inputs, spk_embedding], dim=-1)
        inputs = torch.tanh(self.phonesNspk2embedding(inputs))
 
        # Encoder
        encoder_outputs = self.encoder(inputs, input_lengths)
        #print("Output from encoder ", encoder_outputs)

        # Decoder
        mel_outputs, alignments = self.decoder(encoder_outputs, targets, memory_lengths=input_lengths)
        #print("Mel Outputs from the decoder: ", mel_outputs) 
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


