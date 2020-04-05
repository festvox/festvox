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

        self.hidden2coarse_hidden = SequenceWise(nn.Linear(128, 64))
        self.hidden2fine_hidden = SequenceWise(nn.Linear(128, 64))

        self.coarse_hidden2logits_coarse = SequenceWise(nn.Linear(64, 256))
        self.fine_hidden2logits_fine = SequenceWise(nn.Linear(64, 256))

        self.joint_encoder = nn.LSTM(82, 256, batch_first=True)


    def forward_old(self, mels, coarse, coarse_float, fine, fine_float):

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


    def forward(self, mels, coarse, coarse_float, fine, fine_float):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        coarse_float = coarse_float[:, :-1].unsqueeze(-1)
        fine_float = fine_float[:, :-1].unsqueeze(-1)
        melsNcoarseNfine = torch.cat([mels, coarse_float, fine_float], dim=-1)

        hidden,_ = self.joint_encoder(melsNcoarseNfine)
        coarse_hidden, fine_hidden = hidden.split(128, dim=-1)

        coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
        fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))

        coarse_logits = self.coarse_hidden2logits_coarse(coarse_hidden)
        fine_logits = self.fine_hidden2logits_fine(fine_hidden)

        return coarse_logits, coarse[:, 1:], fine_logits, fine[:, 1:]

    def forward_eval_old(self, mels):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        coarse_float = torch.zeros(mels.shape[0], 1).cuda() #+ 3.4
        fine_float = torch.zeros(mels.shape[0], 1).cuda()
        output = []
        hidden_coarse = None
        hidden_fine = None

        for i in range(T):

           #print("Processing ", i, " of ", T, "Shape of coarse_float: ", coarse_float.shape)

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , coarse_float], dim=-1).unsqueeze(1)

           # Get coarse logits
           mel_encoded, hidden_coarse = self.encoder_coarse(inp, hidden_coarse)
           logits_coarse = self.encodedmel2logits_coarse(mel_encoded)

           # Estimate the coarse categorical
           posterior_coarse = F.softmax(logits_coarse.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_coarse = torch.distributions.Categorical(probs=posterior_coarse)
           categorical_coarse = distribution_coarse.sample().float()

           # Get fine logits
           inp = torch.cat([m , coarse_float, fine_float], dim=-1).unsqueeze(1)
           logits_fine, hidden_fine = self.encoder_fine(inp, hidden_fine)
           logits_fine = self.encodedmel2logits_fine(logits_fine)

           # Estimate the fine categorical
           posterior_fine = F.softmax(logits_fine.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_fine = torch.distributions.Categorical(probs=posterior_fine)
           categorical_fine = distribution_fine.sample().float()


           if i%1000 == 1:
              print("Predicted coarse class at timestep ", i, "is :", categorical_coarse, " and fine class is ", categorical_fine, " Number of steps: ", T)

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
           inp = torch.cat([m , coarse_float, fine_float], dim=-1).unsqueeze(1)

           # Get coarse and fine logits
           mels_encoded, hidden = self.joint_encoder(inp, hidden)
           coarse_hidden, fine_hidden = mels_encoded.split(128, dim=-1)

           coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
           fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))

           coarse_logits = self.coarse_hidden2logits_coarse(coarse_hidden)
           fine_logits = self.fine_hidden2logits_fine(fine_hidden)


           # Estimate the coarse categorical
           posterior_coarse = F.softmax(coarse_logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_coarse = torch.distributions.Categorical(probs=posterior_coarse)
           categorical_coarse = distribution_coarse.sample().float()

           # Estimate the fine categorical
           posterior_fine = F.softmax(fine_logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_fine = torch.distributions.Categorical(probs=posterior_fine)
           categorical_fine = distribution_fine.sample().float()


           if i%1000 == 1:
              print("Predicted coarse class at timestep ", i, "is :", categorical_coarse, " and fine class is ", categorical_fine, " Number of steps: ", T)

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


class Transformer_Encoder_Barebones(nn.Module):

    def __init__(self, in_dim, out_dim):

        super(Transformer_Encoder_Barebones, self).__init__()

        self.positional_encoder_dim = 16
        self.self_attention_dim = 64

        self.query_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))
        self.key_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))
        self.value_layer = SequenceWise(nn.Linear(in_dim + self.positional_encoder_dim, self.self_attention_dim))

        self.positional_embedding = nn.Embedding(2400, self.positional_encoder_dim)

        self.feed_forward = SequenceWise(nn.Linear(self.self_attention_dim, out_dim))

    def forward(self, x, lengths=None):

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

        factor = torch.softmax(torch.bmm(query, key.transpose(1,2)) / self.self_attention_dim, dim=-1)
        inputs = torch.bmm(factor, value)

        # Pass through feed forward layer
        inputs = torch.tanh(self.feed_forward(inputs))

        return inputs

 
class WaveTransformer(WaveLSTM):
        
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveTransformer, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.encodedmel2logits_coarse = SequenceWise(nn.Linear(128, 256))
        self.encoder_coarse = Transformer_Encoder_Barebones(80, 128)

        self.encodedmel2logits_fine = SequenceWise(nn.Linear(128,256))
        self.encoder_fine = Transformer_Encoder_Barebones(80, 128)
    

    def forward(self, mels, coarse, coarse_float, fine, fine_float):
    
        B = mels.size(0)

        mels = self.upsample_network(mels)

        mels_encoded  = self.encoder_coarse(mels)
        coarse_logits = self.encodedmel2logits_coarse(mels_encoded)

        mels_encoded = self.encoder_fine(mels)
        fine_logits = self.encodedmel2logits_fine(mels_encoded)
    
        return coarse_logits, fine_logits

    def forward_eval(self, mels):


        mels = self.upsample_network(mels)

        T = mels.shape[1] / 1600

        output = []
        cnt = 0
        for mel in mels.split(1600, dim=1):
           cnt += 1
           #print("Processing chunk ", cnt, " of ", T, mel.shape)

           mels_encoded  = self.encoder_coarse(mel)
           coarse_logits = self.encodedmel2logits_coarse(mels_encoded)

           mels_encoded = self.encoder_fine(mel)
           fine_logits = self.encodedmel2logits_fine(mels_encoded)

           coarse_logits = F.softmax(coarse_logits, dim=-1)
           coarse_classes = torch.argmax(coarse_logits, dim=-1)

           fine_logits = F.softmax(fine_logits, dim=-1)
           fine_classes = torch.argmax(fine_logits, dim=-1)

           samples = (coarse_classes * 256 + fine_classes) / 32767.5  - 1.0
           print("Shape of samples: ", samples.shape)
           output.append(samples.squeeze(0))
 
        output = torch.stack(output[:-1], dim=0).view(-1)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()




class WaveLSTM2(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM2, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.hidden2coarse_hidden = SequenceWise(nn.Linear(128, 64))
        self.hidden2fine_hidden = SequenceWise(nn.Linear(128+1, 64))

        self.coarse_hidden2logits_coarse = SequenceWise(nn.Linear(64, 256))
        self.fine_hidden2logits_fine = SequenceWise(nn.Linear(64, 256))
           
        self.joint_encoder = nn.LSTM(82, 256, batch_first=True)

    def forward(self, mels, coarse, coarse_float, fine, fine_float):
        
        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        coarse_float = coarse_float[:, :-1].unsqueeze(-1)
        fine_float = fine_float[:, :-1].unsqueeze(-1)
        coarse = coarse[:, 1:]
        melsNcoarseNfine = torch.cat([mels, coarse_float, fine_float], dim=-1)

        hidden,_ = self.joint_encoder(melsNcoarseNfine)
        coarse_hidden, fine_hidden = hidden.split(128, dim=-1)

        #print("Shape of coarse and fine_hidden: ", coarse.shape, fine_hidden.shape)
        coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
        fine_input = torch.cat([fine_hidden, coarse.unsqueeze(-1).float()], dim=-1)
        fine_hidden = torch.relu(self.hidden2fine_hidden(fine_input))

        coarse_logits = self.coarse_hidden2logits_coarse(coarse_hidden)
        fine_logits = self.fine_hidden2logits_fine(fine_hidden)
        
        return coarse_logits, coarse, fine_logits, fine[:, 1:]


          
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
           inp = torch.cat([m , coarse_float, fine_float], dim=-1).unsqueeze(1)

           # Get coarse and fine logits
           mels_encoded, hidden = self.joint_encoder(inp, hidden)
           coarse_hidden, fine_hidden = mels_encoded.split(128, dim=-1)

           # Estimate the coarse categorical
           coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
           coarse_logits = self.coarse_hidden2logits_coarse(coarse_hidden)
           posterior_coarse = F.softmax(coarse_logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_coarse = torch.distributions.Categorical(probs=posterior_coarse)
           categorical_coarse = distribution_coarse.sample().float()


           cc = categorical_coarse.unsqueeze(0).unsqueeze(0).unsqueeze(0)
           #print("Shape of fine_hidden and categorical_coarse: ", fine_hidden.shape, cc.shape, cc)
           fine_input = torch.cat([fine_hidden, cc.float()], dim=-1)
           fine_hidden = torch.relu(self.hidden2fine_hidden(fine_input))
           fine_logits = self.fine_hidden2logits_fine(fine_hidden)

           # Estimate the fine categorical
           posterior_fine = F.softmax(fine_logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_fine = torch.distributions.Categorical(probs=posterior_fine)
           categorical_fine = distribution_fine.sample().float()


           if i%10000 == 1:
              print("  Predicted coarse class at timestep ", i, "is :", categorical_coarse, " and fine class is ", categorical_fine, " Number of steps: ", T)

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


