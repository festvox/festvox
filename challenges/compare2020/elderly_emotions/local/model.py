import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from blocks import *



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

# https://github.com/mkotha/WaveRNN/blob/master/layers/downsampling_encoder.py
class DownsamplingEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        prev_channels = 1
        total_scale = 1
        pad_left = 0
        self.skips = []
        for stride, ksz, dilation_factor in layer_specs:
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)

            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            self.skips.append(skip)
            total_scale *= stride
        self.pad_left = pad_left
        self.total_scale = total_scale

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)

    def forward(self, samples):
        x = samples.transpose(1,2) #.unsqueeze(1)
        #print("Shape of input: ", x.shape)
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec
            #print(i)
            x1 = conv_wide(x)
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            x3 = conv_1x1(x2)
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        return x.transpose(1, 2)




class ValenceSeq2Seq(nn.Module):

    def __init__(self, in_dim=80):
        super(ValenceSeq2Seq, self).__init__()
        self.encoder = Encoder_TacotronOne(in_dim)
        self.mel2output = nn.Linear(256, 3)

    def forward(self, mel):
        mel = self.encoder(mel)
        val_prediction = self.mel2output(mel)
        return val_prediction[:,-1,:]

class ValenceSeq2SeqTransformer(ValenceSeq2Seq):

    def __init__(self, in_dim=80, num_encoder_layers=6, num_attention_heads=4 ):
        super(ValenceSeq2SeqTransformer, self).__init__()

        self.encoder = nn.LSTM(80, 80, bidirectional=True, batch_first=True)
        self.mel2output = nn.Linear(160, 3)
        self.in_dim = in_dim
        self.dropout = 0.2
        self.activation = "relu"
        self.num_encoder_layers = num_encoder_layers
        self.src_mask = None
        self.num_attention_heads = num_attention_heads
        self.embedding_dim = in_dim
        self.pos_encoder = PositionalEncoding(in_dim, self.dropout)
        encoder_layers = nn.TransformerEncoderLayer(in_dim, self.num_attention_heads, 128, self.dropout)
        encoder_norm = nn.LayerNorm(in_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_encoder_layers, encoder_norm)


    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, mel):


        B = mel.size(0)
        T = mel.size(1)

        # Generate mask Transformer takes (T,B,C)
        mel = mel.transpose(0,1)

        if self.src_mask is None or self.src_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).cuda()
            self.src_mask = mask

        mel = mel  * math.sqrt(self.in_dim)
        mel = self.pos_encoder(mel)
        mel = self.transformer_encoder(mel)
        mel,_ = self.encoder(mel.transpose(0,1))
        mel = mel[:,-1,:]
        val_prediction = self.mel2output(mel)

        return val_prediction



class ValenceSeq2Seq_CPCLoss(nn.Module):

    def __init__(self, in_dim=39):
        super(ValenceSeq2Seq_CPCLoss, self).__init__()
        self.encoder = Encoder_TacotronOne(in_dim)
        self.mel2output = nn.Linear(256, 3)

    def forward(self, mel, mel_negative):
        mel = self.encoder(mel)
        mel_negative = self.encoder(mel_negative)

        feature_positive = mel[:,-1,:]
        feature_negative = mel[:,-1,:]

        val_prediction = self.mel2output(feature_positive)

        return val_prediction, feature_positive, feature_negative

    def forward_eval(self, mel):
        mel = self.encoder(mel)

        feature_positive = mel[:,-1,:]

        val_prediction = self.mel2output(feature_positive)

        return val_prediction

class ValenceNArousalSeq2Seq(nn.Module):

    def __init__(self, in_dim=80):
        super(ValenceNArousalSeq2Seq, self).__init__()
        self.encoder = Encoder_TacotronOne(in_dim)

        self.mel2output_valence = nn.Linear(256, 3)
        self.mel2output_arousal = nn.Linear(256, 3)

    def forward(self, mel):
        mel = self.encoder(mel)

        valence_prediction = self.mel2output_valence(mel)
        arousal_prediction = self.mel2output_arousal(mel)

        return valence_prediction[:,-1,:], arousal_prediction[:, -1,:]


class ValenceNArousalExperts(nn.Module):

    def __init__(self, in_dim=80):
        super(ValenceNArousalExperts, self).__init__()
 
        # Shared Encoder
        self.encoder = Encoder_TacotronOne(in_dim)

        # Expert 01
        self.exp1_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp1_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 02
        self.exp2_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp2_fc_b = SequenceWise(nn.Linear(128, 64))
        self.exp2_fc_c = SequenceWise(nn.Linear(64, 256))

        # Gates
        self.gate_arousal = nn.LSTM(in_dim, 128, bidirectional=True, batch_first=True)
        self.gate_valence = nn.LSTM(in_dim, 128, bidirectional=True, batch_first=True)

        self.mel2output_valence = nn.Linear(256, 3)
        self.mel2output_arousal = nn.Linear(256, 3)

    def forward(self, mel):

        mel_copy = mel
        mel = self.encoder(mel)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)

        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)
        exp2_logits = self.exp2_fc_c(exp2_logits)
        

        # Gates
        gate_arousal, _ = self.gate_arousal(mel_copy) 
        gate_arousal = gate_arousal[:, -1,:].unsqueeze(1)

        gate_valence, _ = self.gate_valence(mel_copy) 
        gate_valence = gate_valence[:, -1,:].unsqueeze(1)

        # Combine the experts
        #print("Shape of gate_arousal and exp1_logits: ", gate_arousal.shape, exp1_logits.shape)
   
        exp1_logits_arousal = gate_arousal + exp1_logits
        exp2_logits_arousal = gate_arousal + exp2_logits
        combination_arousal = torch.tanh(exp1_logits_arousal) * torch.sigmoid(exp2_logits_arousal)

        exp1_logits_valence = gate_valence + exp1_logits
        exp2_logits_valence = gate_valence + exp2_logits
        combination_valence = torch.tanh(exp1_logits_valence) * torch.sigmoid(exp2_logits_valence)

        valence_prediction = self.mel2output_valence(combination_valence)
        arousal_prediction = self.mel2output_arousal(combination_arousal)

        return valence_prediction[:,-1,:], arousal_prediction[:, -1,:]



class ValenceSeq2Seq_DownsamplingEncoder(ValenceSeq2Seq):

    def __init__(self, in_dim=80):
        super(ValenceSeq2Seq_DownsamplingEncoder, self).__init__(in_dim)

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(in_dim, encoder_layers)
        self.post_lstm_i = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_h = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_o = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)

        self.mel2output = nn.Linear(in_dim*2, 3)

    def forward(self, mel):
        mel = self.encoder(mel)

        mel_i, _ = self.post_lstm_i(mel)

        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h

        val_prediction = self.mel2output(mel_o)

        return val_prediction[:,-1,:]



class ValenceCPCMultitaskModel(nn.Module):
        
    def __init__(self, in_dim=80):
        super(ValenceCPCMultitaskModel, self).__init__()

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.mel_encoder = nn.LSTM(in_dim, int(in_dim/2), bidirectional=True, batch_first=True)
        self.quant_encoder = DownsamplingEncoder(1, encoder_layers)
        self.post_lstm_i = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_h = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_o = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.latents_armodel = nn.GRU(1, 256, batch_first = True)
        self.decoder_fc = nn.Linear(256, 256)
        self.mel2output = nn.Linear(in_dim*2, 3)
        self.lsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, mel, quant):

        # CPC Loss
        #print("Shape of quant: ", quant.shape) 
        encoded = self.quant_encoder(quant.unsqueeze(-1))
        #print("Encoded")
        latents, hidden = self.latents_armodel(encoded)
        z = latents[:,-1,:]
        predictions = self.decoder_fc(z)
        total = torch.mm(predictions, predictions.transpose(0,1))
        nce_loss = torch.sum(torch.diag(self.lsoftmax(total)))

        # Mel Encoding
        mel, _ = self.mel_encoder(mel)
        mel_i, _ = self.post_lstm_i(mel)
        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i
        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h
        mel_o = mel_o[:,-1,:]

        val_prediction = self.mel2output(mel_o)

        return val_prediction , nce_loss * -1

    def forward_eval(self, mel):
        mel, _ = self.mel_encoder(mel)
    
        mel_i, _ = self.post_lstm_i(mel)
    
        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h

        mel_o = mel_o[:,-1,:]

        val_prediction = self.mel2output(mel_o)

        return val_prediction 
