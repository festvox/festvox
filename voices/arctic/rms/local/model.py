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


class TacotronOneSeqwiseStress(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_qF0s=2, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseStress, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.qF0_embedding = nn.Embedding(num_qF0s, 32)
        self.embeddings2inputs = SequenceWise(nn.Linear(embedding_dim + 32, embedding_dim))


    def forward(self, inputs, qF0s, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)

        # Embeddings for quantized F0s
        qF0s_embedding = self.qF0_embedding(qF0s)

        # Combination
        inputs = torch.cat([inputs, qF0s_embedding], dim=-1)
        inputs = torch.tanh(self.embeddings2inputs(inputs))

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



class Transformer_Encoder_Barebones(nn.Module):

    def __init__(self, in_dim):

        super(Transformer_Encoder_Barebones, self).__init__()

        self.positional_encoder_dim = 16
        self.self_attention_dim = 64

        self.query_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))
        self.key_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))
        self.value_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))

        self.positional_embedding = nn.Embedding(90, self.positional_encoder_dim)

        self.feed_forward = SequenceWise(nn.Linear(self.self_attention_dim, in_dim))

        self.layer_norm = nn.LayerNorm(in_dim)

    def forward(self, x, lengths=None):

        #print("Shape of input to the transformer model: ", x.shape)

        # Figure out the positional embeddings thingy
        positions = torch.arange(x.shape[1]).float().cuda()
        positional_encoding = x.new(x.shape[0], x.shape[1]).zero_()
        positional_encoding += positions
        positional_embedding = self.positional_embedding(positional_encoding.long())

        # Concatenate
        inputs = torch.cat([x, positional_embedding], dim=-1)

        # Self Attention Mechanism
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)

        factor = torch.softmax(torch.bmm(query, key.transpose(1,2)) / 8, dim=-1)
        inputs = torch.bmm(factor, value)
        #inputs =       

        # Pass through feed forward layer
        inputs = torch.tanh(self.feed_forward(inputs))

        # Layer Norm
        inputs = self.layer_norm(inputs + x)

        return inputs 


class TacotronOneSelfAttention(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSelfAttention, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        #self.encoder = nn.ModuleList()
        #for i in range(6):
        #    self.encoder.append(Transformer_Encoder_Barebones(embedding_dim))
        self.dropout = 0.1
        self.activation = "relu"
        self.num_encoder_layers = 6
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, 8, 128, self.dropout, self.activation)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm)

    def forward(self, inputs, targets=None, input_lengths=None):

        B = inputs.size(0)

        # Embeddings for Text
        inputs = self.embedding(inputs)

        # Text Encoder
        #for transformer in self.encoder:
        #    inputs = transformer(inputs)
        #decoder_inputs = inputs
        #print("Shape of decoder_inputs: ", decoder_inputs.shape, " and that of targets: ", targets.shape)
        decoder_inputs = self.encoder(inputs)

        # Decoder
        mel_outputs, alignments = self.decoder(decoder_inputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        
        return mel_outputs, linear_outputs, alignments

