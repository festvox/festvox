import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *


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

class DNSLSTM(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(DNSLSTM, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.encodedmel2logits_coarse = SequenceWise(nn.Linear(128, 256))
        self.encoder_coarse = nn.LSTM(81, 128, batch_first=True)

        self.encodedmel2logits_fine = SequenceWise(nn.Linear(128,256))
        self.encoder_fine = nn.LSTM(82, 128, batch_first=True)


    def forward(self, mels, coarse, coarse_float, fine, fine_float):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        coarse_float = coarse_float[:, :-1].unsqueeze(-1)
        fine_float = fine_float[:, :-1].unsqueeze(-1)
        melsNcoarse = torch.cat([mels, coarse_float], dim=-1)

        mels_encoded,_ = self.encoder_coarse(melsNcoarse)
        coarse_logits = self.encodedmel2logits_coarse(mels_encoded)

        melsNfine = torch.cat([mels, coarse_float, fine_float], dim=-1)
        fine_logits, _ = self.encoder_fine(melsNfine)
        fine_logits = self.encodedmel2logits_fine(fine_logits)

        return coarse_logits, coarse[:, 1:], fine_logits, fine[:, 1:]



    def forward_eval(self, mels):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        coarse_float = torch.zeros(mels.shape[0], 1).cuda() #+ 3.4
        fine_float = torch.zeros(mels.shape[0], 1).cuda()
        output = []
        hidden = None

        for i in range(T):

           #print("Processing ", i, " of ", T, "Shape of coarse_float: ", coarse_float.shape)

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , coarse_float], dim=-1).unsqueeze(1)

           # Get coarse logits
           mel_encoded, hidden = self.encoder_coarse(inp, hidden)
           logits_coarse = self.encodedmel2logits_coarse(mel_encoded)

           # Estimate the coarse categorical
           posterior_coarse = F.softmax(logits_coarse.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_coarse = torch.distributions.Categorical(probs=posterior_coarse)
           categorical_coarse = distribution_coarse.sample().float()

           # Get fine logits
           inp = torch.cat([m , coarse_float, fine_float], dim=-1).unsqueeze(1)
           logits_fine, _ = self.encoder_fine(inp)
           logits_fine = self.encodedmel2logits_fine(logits_fine)

           # Estimate the fine categorical
           posterior_fine = F.softmax(logits_fine.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_fine = torch.distributions.Categorical(probs=posterior_fine)
           categorical_fine = distribution_fine.sample().float()


           if i%1000 == 1:
              print("Predicted coarse class at timestep ", i, "is :", categorical_coarse, " and fine class is ", categorical_fine)

           # Generate sample at current time step
           sample = (categorical_coarse * 256 + categorical_fine) / 32767.5  - 1.0
           output.append(sample)

           # Estimate the input for next time step
           coarse_float = categorical_coarse / 127.5 - 1.0
           coarse_float = coarse_float.unsqueeze(0).unsqueeze(0)
           fine_float = categorical_fine / 127.5 - 1.0
           fine_float = fine_float.unsqueeze(0).unsqueeze(0)


        output = torch.stack(output, dim=0)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()

