import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from blocks import *




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
        prev_channels = 39
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




class MaskSeq2Seq(nn.Module):

    def __init__(self, in_dim=80):
        super(MaskSeq2Seq, self).__init__()
        self.encoder = Encoder_TacotronOne(in_dim)
        self.mel2output = nn.Linear(256, 2)

    def forward(self, mel):
        mel = self.encoder(mel)
        val_prediction = self.mel2output(mel)
        return val_prediction[:,-1,:]



class ValenceSeq2Seq_DownsamplingEncoder(MaskSeq2Seq):

    def __init__(self, in_dim=80):
        super(ValenceSeq2Seq_DownsamplingEncoder, self).__init__(in_dim)

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(in_dim, encoder_layers)
        self.post_lstm_i = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_h = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_o = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)

        self.mel2output = nn.Linear(in_dim*2, 3)

    def forward(self, mel):

        mel = self.encoder(mel)

        # Classification part
        mel_i, _ = self.post_lstm_i(mel)

        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h

        val_prediction = self.mel2output(mel_o)



        return val_prediction[:,-1,:], pos_mel, neg_mel



class MaskSeq2Seq_Triplet(ValenceSeq2Seq_DownsamplingEncoder):

    def __init__(self, in_dim=39):
        super(MaskSeq2Seq_Triplet, self).__init__(in_dim)

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(in_dim, encoder_layers)
        self.post_lstm_i = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_h = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)
        self.post_lstm_o = nn.LSTM(in_dim*2, in_dim, bidirectional=True, batch_first=True)

        self.triplet_lstm = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)

        self.mel2output = nn.Linear(in_dim*2, 3)
        self.triplet_encoder = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)

    def forward(self, mel, pos_mel, neg_mel):

        mel = self.encoder(mel)

        # Triplet Loss part
        pos_mel = self.encoder(pos_mel)
        neg_mel = self.encoder(neg_mel)

        mel, _ = self.triplet_lstm(mel) 
        pos_mel, _ = self.triplet_lstm(pos_mel)
        neg_mel, _ = self.triplet_lstm(neg_mel)

        # Classification part
        mel_i, _ = self.post_lstm_i(mel)

        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h

        val_prediction = self.mel2output(mel_o)



        return val_prediction[:,-1,:], mel[:, -1,:], pos_mel[:,-1,:], neg_mel[:,-1,:]


    def forward_eval(self, mel):

        mel = self.encoder(mel)

        mel, _ = self.triplet_lstm(mel) 

        # Classification part
        mel_i, _ = self.post_lstm_i(mel)

        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)
        mel_o = mel_o + mel_h

        val_prediction = self.mel2output(mel_o)



        return val_prediction[:,-1,:]


