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


class TacotronOnekothamspec(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOnekothamspec, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)


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

        return mel_outputs, alignments


# https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/upsample.py
class UpsampleNetwork(nn.Module):
    def __init__(self, upsample_scales, upsample_activation="none",
                 upsample_activation_params={}, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.up_layers = nn.ModuleList()
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, scale)
            stretch = Stretch2d(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / np.prod(k_size))
            conv = nn.utils.weight_norm(conv)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)
            if upsample_activation != "none":
                nonlinear = _get_activation(upsample_activation)
                self.up_layers.append(nonlinear(**upsample_activation_params))

    def forward(self, c):
   
        c = c.transpose(1,2)

        # B x 1 x C x T
        c = c.unsqueeze(1)
        for f in self.up_layers:
            c = f(c)
        # B x C x T
        c = c.squeeze(1)

        if self.indent > 0:
            c = c[:, :, self.indent:-self.indent]
        return c.transpose(1,2)


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


def _get_activation(upsample_activation):
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear

class WaveLSTM(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.encodedmel2logits_coarse = SequenceWise(nn.Linear(128, 256))
        self.encoder_coarse = nn.LSTM(81, 128, batch_first=True)

        self.encodedmel2logits_fine = SequenceWise(nn.Linear(128, 256))
        self.encoder_fine = nn.LSTM(256, 128, batch_first=True)


    def forward(self, mels, coarse, coarse_float, fine, fine_float):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        coarse_float = coarse_float[:, :-1].unsqueeze(-1)
        mels = torch.cat([mels, coarse_float], dim=-1)

        mels_encoded,_ = self.encoder_coarse(mels)
        coarse_logits = self.encodedmel2logits_coarse(mels_encoded)

        fine_logits, _ = self.encoder_fine(coarse_logits)
        fine_logits = self.encodedmel2logits_fine(fine_logits)

        return coarse_logits, coarse[:, 1:], fine_logits, fine[:, 1:]



    def forward_eval(self, mels):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        coarse_float = torch.zeros(mels.shape[0], 1).cuda() #+ 3.4
        output = []
        hidden = None

        for i in range(T):

           #print("Processing ", i, " of ", T, "Shape of coarse_float: ", coarse_float.shape)

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , coarse_float], dim=-1).unsqueeze(1)

           # Pass through encoder and logits
           mel_encoded, hidden = self.encoder(inp, hidden)
           logits = self.encodedmel2logits(mel_encoded)

           # Estimate the coarse and coarse_float
           posterior = F.softmax(logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution = torch.distributions.Categorical(probs=posterior)
           c_cat = distribution.sample().float()
           if i%1000 == 1:
              print("Predicted class at timestep ", i, "is :", c_cat)
           coarse_float = c_cat / 127.5 - 1.0
           #print("coarse_float and coarse_cat are: ", coarse_float, c_cat)
           coarse_float = coarse_float.unsqueeze(0).unsqueeze(0)
           #print("coarse_float and coarse_cat are: ", coarse_float, c_cat)
           sample = c_cat * 256 / 32767.5 - 1.0
           output.append(sample)

        output = torch.stack(output, dim=0) #.squeeze(1).squeeze(1)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()

