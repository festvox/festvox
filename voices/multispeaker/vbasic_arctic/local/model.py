import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *
from util import *

from torch.optim import SGD
import torch.nn.functional as F

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
        #self.encoder = Encoder_TacotronOne_nocudnn(embedding_dim)
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


class sgd_maml(SGD):

    def __init__(self, params, lr=0.01, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(sgd_maml, self).__init__(params, lr=0.01)

        self.param_groups_fast = self.param_groups


    @torch.no_grad()
    def step_maml(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = []
        for group in self.param_groups_fast:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    #print("Not computing grad since p.grad is None")
                    parameters.append(p)
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])
                parameters.append(p)
        return parameters


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




class WaveLSTM5(nn.Module):

    def __init__(self):
        super(WaveLSTM5, self).__init__()

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.logits_dim = 30
        self.joint_encoder = nn.LSTM(97, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.linear2logits =  SequenceWise(nn.Linear(64, self.logits_dim))

        self.spk_embedding = nn.Embedding(2, 16)

    def forward(self, mels, spk, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)

        spk = self.spk_embedding(spk.long()).unsqueeze(1).expand(-1, mels.size(1), -1)

        melsNxNspk = torch.cat([mels, inp, spk], dim=-1)
        outputs, hidden = self.joint_encoder(melsNxNspk)

        logits = torch.tanh(self.hidden2linear(outputs))
        return self.linear2logits(logits), x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, spk, log_scale_min=-50.0):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden = None
        output = []

        spk = self.spk_embedding(spk.long()).squeeze(0)

        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           #print("Shape of m and spk ", m.shape, spk.shape)
           inp = torch.cat([m , current_input, spk], dim=-1).unsqueeze(1)

           # Get logits
           outputs, hidden = self.joint_encoder(inp, hidden)
           logits = torch.tanh(self.hidden2linear(outputs))
           logits = self.linear2logits(logits)

           # Sample the next input
           sample = sample_from_discretized_mix_logistic(
                        logits.view(B, -1, 1), log_scale_min=log_scale_min)

           output.append(sample.data)
           current_input = sample

           if i%10000 == 1:
              print("  Predicted sample at timestep ", i, "is :", sample, " Number of steps: ", T)


        output = torch.stack(output, dim=0)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()

