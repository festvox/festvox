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
        prev_channels = 80
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



class ValenceSeq2Seq_DownsamplingEncoder(ValenceSeq2Seq):

    def __init__(self):
        super(ValenceSeq2Seq_DownsamplingEncoder, self).__init__()

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
        self.encoder = DownsamplingEncoder(80, encoder_layers)
        self.post_lstm_i = nn.LSTM(80, 80, bidirectional=True, batch_first=True)
        self.post_lstm_h = nn.LSTM(160, 80, bidirectional=True, batch_first=True)
        self.post_lstm_o = nn.LSTM(160, 80, bidirectional=True, batch_first=True)

        self.mel2output = nn.Linear(160, 3)

    def forward(self, mel):
        mel = self.encoder(mel)

        mel_i, _ = self.post_lstm_i(mel)

        mel_h, _ = self.post_lstm_h(mel_i)
        mel_h = mel_h + mel_i

        mel_o,_ = self.post_lstm_o(mel_h)        
        mel_o = mel_o + mel_h

        val_prediction = self.mel2output(mel_o)

        return val_prediction[:,-1,:]


