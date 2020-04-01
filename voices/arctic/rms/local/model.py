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


class Decoder_Audiosearch(nn.Module):
    def __init__(self, in_dim, r):
        super(Decoder_Audiosearch, self).__init__()
        self.in_dim = in_dim
        self.r = r
        self.prenet = Prenet(in_dim * r, sizes=[256, 128])
        # (prenet_out + attention context) -> output
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttention(256)
        )
        self.memory_layer = nn.Linear(256, 256, bias=False)
        self.project_to_decoder_in = nn.Linear(512, 256)

        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])

        self.proj_to_mel = nn.Linear(256, in_dim * r)
        self.max_decoder_steps = 200

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        """
        Decoder forward step.
        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.
        Args:
            encoder_outputs: Encoder outputs. (B, T_encoder, dim)
            inputs: Decoder inputs. i.e., mel-spectrogram. If None (at eval-time),
              decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for
              attention masking.
        """
        B = encoder_outputs.size(0)
      
        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        if inputs is not None:
            #print("Shape of inputs and r: ", inputs.shape, self.r)
            # Grouping multiple frames if necessary
            if inputs.size(-1) == self.in_dim:
                inputs = inputs.view(B, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == self.in_dim * self.r
            T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_())

        # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else inputs[t - 1]
            # Prenet
            ####### Sai Krishna Rallabandi 15 June 2019 #####################
            #print("Shape of input to the decoder prenet: ", current_input.shape)
            if len(current_input.shape) < 3:
               current_input = current_input.unsqueeze(1)
            #################################################################
 
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = decoder_input
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        assert greedy or len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L273
def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()

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
        self.decoderfc_search = nn.Linear(256, 2)

        self.decoder = Decoder_Audiosearch(256, r)
        self.decoderfc_reconstruction = SequenceWise(nn.Linear(256, n_vocab))

    def forward(self, inputs, targets):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        targets = self.embedding(targets.long())
        encoder_outputs = self.encoder(inputs)
 
        outputs, _ = self.decoder_LSTM(encoder_outputs)
        outputs = outputs[:, 0, :] 
        #print("Shape of final hidden state from decoder lstm: ", outputs.shape)
        
        outputs_reconstructed, alignments  = self.decoder(encoder_outputs, targets)
        #print("Shape of reconstructed outputs : ", outputs_reconstructed.shape)

        return self.decoderfc_search(outputs), self.decoderfc_reconstruction(outputs_reconstructed.view(B, -1, 256))
