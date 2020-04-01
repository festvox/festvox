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


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TacotronOneSelfAttention(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_attention_heads = 6, num_encoder_layers = 6, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSelfAttention, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.dropout = 0.2
        self.activation = "relu"
        self.num_encoder_layers = num_encoder_layers
        self.src_mask = None
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(embedding_dim, self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, self.num_attention_heads, 128, self.dropout)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_encoder_layers, encoder_norm)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, inputs, targets=None, input_lengths=None):

        B = inputs.size(0)
        T = inputs.size(1)

        # Embeddings for text
        inputs = self.embedding(inputs)

        # Generate mask Transformer takes (T,B,C)
        inputs = inputs.transpose(0,1)

        if self.src_mask is None or self.src_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).cuda()
            self.src_mask = mask

        inputs = inputs  * math.sqrt(self.embedding_dim)
        inputs = self.pos_encoder(inputs)
        #decoder_inputs = self.transformer_encoder(inputs, self.src_mask) 
        decoder_inputs = self.transformer_encoder(inputs)
        decoder_inputs = decoder_inputs.transpose(0,1)
 
        if isnan(decoder_inputs):
           print("NANs in decoder inputs")
           sys.exit()
        #else:
        #   print(decoder_inputs)
        

        # Decoder
        input_lengths = None
        mel_outputs, alignments = self.decoder(decoder_inputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        
        return mel_outputs, linear_outputs, alignments

def isnan(x):
    return (x != x).any()


class TacotronOneSeqwiseAudiosearch(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseAudiosearch, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
 
        self.decoder_LSTM = nn.LSTM(256, 128, batch_first=True, bidirectional = True)
        self.decoder_fc = nn.Linear(256, 2)

    def forward(self, inputs):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        encoder_outputs = self.encoder(inputs)
 
        outputs, _ = self.decoder_LSTM(encoder_outputs)
        outputs = outputs[:, 0, :] 
        #print("Shape of final hidden state from decoder lstm: ", outputs.shape)
        
        return self.decoder_fc(outputs)
