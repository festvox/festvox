import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from blocks import *
from layers import *
from torch.optim import SGD



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
        prev_channels = channels
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




class LIDSeq2Seq(nn.Module):

    def __init__(self, in_dim=80):
        super(LIDSeq2Seq, self).__init__()

        self.encoder = Encoder_TacotronOne(in_dim)
        self.mel2output = nn.Linear(256, 2)

    def forward(self, mel):
        mel = self.encoder(mel)
        val_prediction = self.mel2output(mel)
        return val_prediction[:,-1,:]



class LIDSeq2SeqMixtureofExperts_basic(nn.Module):

    def __init__(self, in_dim=80):
        super(LIDSeq2SeqMixtureofExperts, self).__init__()

        # Shared encoder
        self.encoder = Encoder_TacotronOne(in_dim)

        # Expert 01
        self.exp1_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp1_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 02
        self.exp2_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp2_fc_b = SequenceWise(nn.Linear(128, 256))

        self.mel2output = nn.Linear(256, 2)


    def forward(self, mel):

        # Pass through shared layers
        mel = self.encoder(mel)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)

        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)

        # Combine the experts
        combination = torch.tanh(exp1_logits) * torch.sigmoid(exp2_logits)

        val_prediction = self.mel2output(combination)
        return val_prediction[:,-1,:]



class LIDSeq2SeqMixtureofExperts(nn.Module):

    def __init__(self, in_dim=80):
        super(LIDSeq2SeqMixtureofExperts, self).__init__()

        # Shared encoder
        self.encoder = Encoder_TacotronOne(in_dim)

        # Expert 01
        self.exp1_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp1_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 02
        self.exp2_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp2_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 03
        self.exp3_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp3_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 04
        self.exp4_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp4_fc_b = SequenceWise(nn.Linear(128, 256))

        self.mel2output = nn.Linear(256, 2)


    def forward(self, mel):

        # Pass through shared layers
        mel = self.encoder(mel)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)

        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)

        # Pass through expert 03
        exp3_logits = torch.tanh(self.exp3_fc_a(mel))
        exp3_logits = self.exp3_fc_b(exp3_logits)

        # Pass through expert 04
        exp4_logits = torch.tanh(self.exp4_fc_a(mel))
        exp4_logits = self.exp4_fc_b(exp4_logits)

        # Combine the experts
        experts = torch.stack([exp1_logits, exp2_logits, exp3_logits, exp4_logits], dim=0)
        weights_experts = torch.softmax(experts, dim=0)

        logits = weights_experts[0,:,:] * exp1_logits + weights_experts[1,:,:] * exp2_logits + weights_experts[2,:,:] * exp3_logits + weights_experts[3,:,:] * exp4_logits

        val_prediction = self.mel2output(logits)
        return val_prediction[:,-1,:]


class LIDSeq2SeqDownsampling(nn.Module):

    def __init__(self, in_dim=80):
        super(LIDSeq2SeqDownsampling, self).__init__()

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
        self.embedding_dim = 256
        self.encoder = DownsamplingEncoder(self.embedding_dim, encoder_layers)
        self.pre_encoder_fc = SequenceWise(nn.Linear(in_dim, self.embedding_dim))

        self.mel2output = nn.Linear(256, 2)

    def forward(self, mel):
        mel = torch.tanh(self.pre_encoder_fc(mel))
        mel = self.encoder(mel)
        val_prediction = self.mel2output(mel)
        return val_prediction[:,-1,:]


# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch/blob/a9dab4e759aaa68dce1b1ada46a8035076ba3296/src/model/model.py#L245
# https://github.com/ssp573/Contrastive-Predictive-Coding
# https://github.com/davidtellez/contrastive-predictive-coding
class CPCBaseline(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(CPCBaseline, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

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
        self.encoder = DownsamplingEncoder(embedding_dim, encoder_layers)
        self.decoder_fc = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.decoder_lstm = nn.GRU(embedding_dim, embedding_dim, batch_first = True)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.embedding_dim = 256 
        self.pre_encoder_fc = SequenceWise(nn.Linear(mel_dim, self.embedding_dim))

    def forward(self, inputs):
        mel = torch.tanh(self.pre_encoder_fc(inputs))
        #print("Shape of mel: ", mel.shape)
        encoded = self.encoder(mel)
        latents, hidden = self.decoder_lstm(encoded)
        #print("Shape of latents: ", latents.shape)
        z = latents[:,-1,:]
        #print("Shape of z: ", z.shape)
        predictions = self.decoder_fc(z)
        #print("Shape of predictions: ", predictions.shape)
        total = torch.mm(predictions, predictions.transpose(0,1))
        nce_loss = torch.sum(torch.diag(self.lsoftmax(total)))
        return -1 * nce_loss


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
