import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from util import *

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/attention.py#L46 
class AttentionWrapper(nn.Module):
    def __init__(self, rnn_cell, attention_mechanism,
                 score_mask_value=-float("inf")):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism
        self.score_mask_value = score_mask_value

    def forward(self, query, attention, cell_state, memory,
                processed_memory=None, mask=None, memory_lengths=None):

        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        # Concat input query and previous attention context
        ######### Sai Krishna 15 June 2019 #####################
        if len(query.shape) > 2:
              query = query.squeeze(1)
        #print("Shapes of query and attention: ", query.shape, attention.shape)
        ##########################################################
        cell_input = torch.cat((query, attention), -1)

        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)

        # Alignment
        # (batch, max_time)
        alignment = self.attention_mechanism(cell_output, processed_memory)

        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Normalize attention weight
        alignment = F.softmax(alignment,dim=-1)

        # Attention context vector
        # (batch, 1, dim)
        attention = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        attention = attention.squeeze(1)

        return cell_output, attention, alignment


# https://github.com/mozilla/TTS/blob/dev/layers/common_layers.py#L147
class GravesAttention(nn.Module):
    """ Discretized Graves attention:
        - https://arxiv.org/abs/1910.10288
        - https://arxiv.org/pdf/1906.01083.pdf
    """
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        super(GravesAttention, self).__init__()
        print("Using Graves Attention")
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
            nn.ReLU(),
            nn.Linear(query_dim, 3*K, bias=True))
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2*self.K):(3*self.K)], 1.)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K:(2*self.K)], 10)  # bias std

    def init_states(self, inputs):
        if self.J is None or inputs.shape[1]+1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1]+2).cuda().float() + 0.5
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs):
        """
        shapes:
            query: B x D_attention_rnn
            inputs: B x T_in x D_encoder
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # dropout to decorrelate attention heads
        g_t = torch.nn.functional.dropout(g_t, p=0.5, training=self.training)

        # attention GMM parameters
        sig_t = torch.nn.functional.softplus(b_t) + self.eps
        
        mu_t = self.mu_prev + torch.nn.functional.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[:inputs.size(1)+1]

        # attention weights
        #print("Type of sig_t: ", sig_t.type())
        #print("Type of mu_t: ", mu_t.type())
        #print("Type of g_t: ", g_t.type())
        #print("Type of j: ", j.type())
        #print("Type of self.J: ", self.J.type()) 
        phi_t = g_t.unsqueeze(-1) * (1 / (1 + torch.sigmoid((mu_t.unsqueeze(-1) - j) / sig_t.unsqueeze(-1))))

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = 1e-8

        #print("Shape of alpha_t: ", alpha_t.shape)
        return alpha_t
        sys.exit()

        # apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)
        self.attention_weights = alpha_t
        self.mu_prev = mu_t
        return context

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


class TacotronOneSeqwiseTones(TacotronOne):

    def __init__(self, n_vocab, n_tones, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseTones, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.tones_embedding = nn.Embedding(n_tones, 256)
        self.tonesNphones2inputs = nn.Linear(512,256)
        self.decoder.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            GravesAttention(256, 256)
           )
        self.decoder.graves_attention = 1

    def forward(self, inputs, tones, targets=None, input_lengths=None):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        tones = self.tones_embedding(tones)
        inputs = torch.cat([inputs, tones], dim=-1)
        inputs = torch.tanh(self.tonesNphones2inputs(inputs))

        encoder_outputs = self.encoder(inputs, input_lengths)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None
        
        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
      
        return mel_outputs, linear_outputs, alignments


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





class WaveLSTM3(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM3, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.hidden2coarse_hidden = SequenceWise(nn.Linear(128, 64))
        self.hidden2fine_hidden = SequenceWise(nn.Linear(128, 64))

        self.coarse_hidden2logits_coarse = SequenceWise(nn.Linear(64, 256))
        self.fine_hidden2logits_fine = SequenceWise(nn.Linear(64, 256))
           
        self.joint_encoder = nn.LSTM(82, 256, batch_first=True)
        self.fine_encoder = nn.LSTM(129, 128, batch_first=True)

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

        fine_hidden, _ = self.fine_encoder(fine_input)
        fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))

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
        hidden_fine = None           
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
           fine_input = torch.cat([fine_hidden, cc.float()], dim=-1)
           fine_hidden, hidden_fine = self.fine_encoder(fine_input, hidden_fine)
           fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))
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

class WaveLSTM4(WaveLSTM3):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM4, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        #self.fine_hidden2logits_fine = SequenceWise(nn.Linear(64, 30))
        self.fine_encoder = nn.LSTM(130, 128, batch_first=True)

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

        coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
        fine_input = torch.cat([fine_hidden, coarse.unsqueeze(-1).float(), fine_float], dim=-1)
        fine_hidden, _ = self.fine_encoder(fine_input)
        fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))

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
        hidden_fine = None
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
           fine_input = torch.cat([fine_hidden, cc.float(), fine_float.unsqueeze(0)], dim=-1)
           fine_hidden, hidden_fine = self.fine_encoder(fine_input, hidden_fine)
           fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))
           fine_logits = self.fine_hidden2logits_fine(fine_hidden)

           # Estimate the fine categorical
           posterior_fine = F.softmax(fine_logits.float(), dim=-1).squeeze(0).squeeze(0)
           distribution_fine = torch.distributions.Categorical(probs=posterior_fine)
           categorical_fine = distribution_fine.sample().float()

           if i%10000 == 1:
              print("  Predicted coarse class at timestep ", i, "is :", categorical_coarse, " and fine class is ", categorical_fine, " Number of steps: ", T)

           # Generate sample at current time step
           sample = (categorical_coarse * 256 + categorical_fine) / 65536 #/ 32767.5  - 1.0
           output.append(sample)

           # Estimate the input for next time step
           coarse_float = categorical_coarse / 127.5 - 1.0
           coarse_float = coarse_float.unsqueeze(0).unsqueeze(0)
           fine_float = categorical_fine / 127.5 - 1.0
           fine_float = fine_float.unsqueeze(0).unsqueeze(0)


        output = torch.stack(output, dim=0)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()



class WaveLSTM5(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025, logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM5, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.logits_dim = logits_dim
        self.joint_encoder = nn.LSTM(81, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.linear2logits =  SequenceWise(nn.Linear(64, self.logits_dim))

    def forward(self, mels, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)

        melsNx = torch.cat([mels, inp], dim=-1)
        outputs, hidden = self.joint_encoder(melsNx)

        logits = torch.tanh(self.hidden2linear(outputs))
        return self.linear2logits(logits), x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden = None
        output = []


        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , current_input], dim=-1).unsqueeze(1)

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


class WaveLSTM6(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM6, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
 
        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.melsNx2inputs = SequenceWise(nn.Linear(81,128))
        #self.joint_encoders = nn.ModuleList()
        #for i in range(3):
        #   enc = nn.LSTM(128, 128, batch_first=True)
        #   self.joint_encoders.append(enc)

        self.num_layers = 2
        self.rnns = [nn.LSTM(input_size=128 if l == 0 else 128,
             hidden_size=128, num_layers=1, batch_first=True,
             ) for l in range(self.num_layers)]
       
        self.joint_encoders = nn.ModuleList(self.rnns)
        self.hiddenfc = SequenceWise(nn.Linear(128,128))

        self.hidden2linear =  SequenceWise(nn.Linear(128, 64))
        self.logits_dim = logits_dim
        self.linear2logits =  SequenceWise(nn.Linear(64, logits_dim))

        self.highways = nn.ModuleList(
            [Highway(128, 128) for _ in range(4)])


    def forward(self, mels, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)
        melsNx = torch.cat([mels, inp], dim=-1)
        inputs = torch.tanh(self.melsNx2inputs(melsNx))

        # LSTM
        for enc in self.joint_encoders:
            enc.flatten_parameters()
            inputs, hidden = enc(inputs)
            #inputs = torch.tanh(self.hiddenfc(inputs))

        #print("Shape of inputs: ", inputs.shape)
        # Residual Connection            
        #inputs += residual

        # Highway Connection
        #for highway in self.highways:
        #    inputs = highway(inputs)

        #print("Shape of inputs: ", inputs.shape)
        logits = torch.tanh(self.hidden2linear(inputs))

        return self.linear2logits(logits), x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)
 
        mels = self.upsample_network(mels)
        T = mels.size(1)
 
        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hiddens = [None, None, None]
        output = []
 
 
        for i in range(T):
 
           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           melsNx = torch.cat([m , current_input], dim=-1).unsqueeze(1)
           inputs = torch.tanh(self.melsNx2inputs(melsNx))
 
           # Get logits
           for k in range(self.num_layers):
               inputs, hidden = self.joint_encoders[k](inputs, hiddens[k])
               hiddens[k] = hidden

           # Highway Connection
           #for highway in self.highways:
           #    inputs = highway(inputs)

           logits = torch.tanh(self.hidden2linear(inputs))
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
 

# Lets add dropout
class WaveLSTM7(TacotronOne):
    
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM7, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.melsNx2inputs = SequenceWise(nn.Linear(81,128))

        self.num_layers = 2
        self.rnns = [nn.LSTM(input_size=128 if l == 0 else 128,
             hidden_size=128, num_layers=1, batch_first=True,
             ) for l in range(self.num_layers)]

        self.joint_encoders = nn.ModuleList(self.rnns)

        self.hidden2linear =  SequenceWise(nn.Linear(128, 64))
        self.logits_dim = logits_dim
        self.linear2logits =  SequenceWise(nn.Linear(64, logits_dim))

        self.highways = nn.ModuleList(
            [Highway(128, 128) for _ in range(4)])

        self.dropout = nn.Dropout(0.3)

    def forward(self, mels, x):

        B = mels.size(0)

        # Mel Encoding
        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)
        melsNx = torch.cat([mels, inp], dim=-1)
        inputs = torch.tanh(self.melsNx2inputs(melsNx))
        #inputs = self.dropout(inputs)

        # LSTM
        for enc in self.joint_encoders:
            enc.flatten_parameters()
            inputs, hidden = enc(inputs)

        # Logits
        logits = torch.tanh(self.hidden2linear(inputs))
        logits = self.dropout(logits)
        return self.linear2logits(logits), x[:,1:].unsqueeze(-1)


    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)
 
        mels = self.upsample_network(mels)
        T = mels.size(1)
 
        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hiddens = [None, None, None]
        output = []
 
 
        for i in range(T):
 
           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           melsNx = torch.cat([m , current_input], dim=-1).unsqueeze(1)
           inputs = torch.tanh(self.melsNx2inputs(melsNx))
 
           # Get logits
           for k in range(self.num_layers):
               inputs, hidden = self.joint_encoders[k](inputs, hiddens[k])
               hiddens[k] = hidden

           # Highway Connection
           #for highway in self.highways:
           #    inputs = highway(inputs)

           logits = torch.tanh(self.hidden2linear(inputs))
           logits = self.linear2logits(logits)
    
           # Sample the next input
           sample = sample_from_discretized_mix_logistic(
                        logits.view(B, -1, 1), log_scale_min=log_scale_min)
 
           output.append(sample.data)
           current_input = sample

 
           if i%10000 == 1:
              print("  Predicted sampxle at timestep ", i, "is :", sample, " Number of steps: ", T)
 
 
        output = torch.stack(output, dim=0)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()
 



# Lets add 2 MoLs and sample randomly at test time
class WaveLSTM8(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025, logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM8, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.logits_dim = logits_dim
        self.joint_encoder = nn.LSTM(81, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.linear2logits1 =  SequenceWise(nn.Linear(64, self.logits_dim))
        self.linear2logits2 =  SequenceWise(nn.Linear(64, 30))


    def forward(self, mels, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)

        melsNx = torch.cat([mels, inp], dim=-1)
        outputs, hidden = self.joint_encoder(melsNx)

        logits = torch.tanh(self.hidden2linear(outputs))
        return self.linear2logits1(logits),  self.linear2logits2(logits), x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden = None
        output = []


        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , current_input], dim=-1).unsqueeze(1)

           # Get logits
           outputs, hidden = self.joint_encoder(inp, hidden)
           logits = torch.tanh(self.hidden2linear(outputs))
           logits1 = self.linear2logits1(logits)
           logits2 = self.linear2logits2(logits)
           #logits3 = logits1 + logits2
           logits = [logits1, logits2] #, logits3]
           logits = random.choice(logits)

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



    def forward_eval_sampling1(self, mels, log_scale_min=-50.0):

        B = mels.size(0)
 
        mels = self.upsample_network(mels)
        T = mels.size(1)
 
        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden = None
        output = []

 
        for i in range(T):
 
           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , current_input], dim=-1).unsqueeze(1)
 
           # Get logits
           outputs, hidden = self.joint_encoder(inp, hidden)
           logits = torch.tanh(self.hidden2linear(outputs))
           logits1 = self.linear2logits1(logits)
           logits2 = self.linear2logits2(logits)
           #logits3 = logits1 + logits2
           #logits = [logits1, logits2] #, logits3]
           #logits = random.choice(logits)
 
           # Sample the next input
           sample1 = sample_from_discretized_mix_logistic(
                        logits1.view(B, -1, 1), log_scale_min=log_scale_min)
           sample2 = sample_from_discretized_mix_logistic(
                        logits2.view(B, -1, 1), log_scale_min=log_scale_min)

           samples = [sample1, sample2,  0.5 * (sample1 + sample2)]
           sample = random.choice(samples)
 
           output.append(sample.data)
           current_input = sample
 
           if i%10000 == 1:
              print("  Predicted sample at timestep ", i, "is :", sample, " Number of steps: ", T)
 
 
        output = torch.stack(output, dim=0)
        print("Shape of output: ", output.shape)
        return output.cpu().numpy()
 




# Lets add 2 encoders and sample randomly at test time
class WaveLSTM8b(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025, logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM8b, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.logits_dim = logits_dim
        self.joint_encoder1 = nn.LSTM(81, 256, batch_first=True)
        self.joint_encoder2 = nn.GRU(81, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.linear2logits =  SequenceWise(nn.Linear(64, self.logits_dim))


    def forward(self, mels, x):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)

        melsNx = torch.cat([mels, inp], dim=-1)
        outputs1, hidden1 = self.joint_encoder1(melsNx)
        outputs2, hidden2 = self.joint_encoder2(melsNx)

        logits1 = torch.tanh(self.hidden2linear(outputs1))
        logits2 = torch.tanh(self.hidden2linear(outputs2))

        return self.linear2logits(logits1),  self.linear2logits(logits2), x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        T = mels.size(1)

        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden1 = None
        hidden2 = None
        output = []


        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , current_input], dim=-1).unsqueeze(1)

           # Get logits
           outputs1, hidden1 = self.joint_encoder1(inp, hidden1)
           outputs2, hidden2 = self.joint_encoder2(inp, hidden2)

           logits1 = torch.tanh(self.hidden2linear(outputs1))
           logits2 = torch.tanh(self.hidden2linear(outputs2))

           logits1 = self.linear2logits(logits1)
           logits2 = self.linear2logits(logits2)

           logits = [logits1, logits2]
           logits = random.choice(logits)

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




# Lets add 2 encoders and sample based on a gating function
class WaveLSTM8c(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025, logits_dim=30,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(WaveLSTM8c, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.logits_dim = logits_dim
        self.joint_encoder1 = nn.LSTM(81, 256, batch_first=True)
        self.joint_encoder2 = nn.GRU(81, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.linear2logits =  SequenceWise(nn.Linear(64, self.logits_dim))


    def forward(self, mels, x):
        
        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        inp = x[:, :-1].unsqueeze(-1)

        melsNx = torch.cat([mels, inp], dim=-1)
        outputs1, hidden1 = self.joint_encoder1(melsNx)
        outputs2, hidden2 = self.joint_encoder2(melsNx)

        logits1 = torch.tanh(self.hidden2linear(outputs1))
        logits2 = torch.tanh(self.hidden2linear(outputs2))

        combination = torch.tanh(logits1) * torch.sigmoid(logits2)

        return self.linear2logits(combination),  x[:,1:].unsqueeze(-1)

    def forward_eval(self, mels, log_scale_min=-50.0):

        B = mels.size(0)
 
        mels = self.upsample_network(mels)
        T = mels.size(1)

        current_input = torch.zeros(mels.shape[0], 1).cuda()
        hidden1 = None
        hidden2 = None
        output = []


        for i in range(T):

           # Concatenate mel and coarse_float
           m = mels[:, i,:]
           inp = torch.cat([m , current_input], dim=-1).unsqueeze(1)

           # Get logits
           outputs1, hidden1 = self.joint_encoder1(inp, hidden1)
           outputs2, hidden2 = self.joint_encoder2(inp, hidden2)
 
           logits1 = torch.tanh(self.hidden2linear(outputs1))
           logits2 = torch.tanh(self.hidden2linear(outputs2))

           combination = torch.tanh(logits1) * torch.sigmoid(logits2)

           logits = self.linear2logits(combination)
 
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


# MoL for fine
class WaveLSTM10(nn.Module):

    def __init__(self, logits_dim=30):
        super(WaveLSTM10, self).__init__()

        self.logits_dim = logits_dim

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)

        self.hidden2coarse_hidden = SequenceWise(nn.Linear(128, 64))
        self.hidden2fine_hidden = SequenceWise(nn.Linear(128, 64))

        self.coarse_hidden2logits_coarse = SequenceWise(nn.Linear(64, 256))
        self.fine_hidden2logits_fine = SequenceWise(nn.Linear(64, self.logits_dim))

        self.joint_encoder = nn.LSTM(81, 256, batch_first=True)
        self.fine_encoder = nn.LSTM(129, 128, batch_first=True)

    def forward(self, mels, coarse, coarse_float, fine, fine_float):

        B = mels.size(0)

        mels = self.upsample_network(mels)
        mels = mels[:,:-1,:]
        coarse_float = coarse_float[:, :-1].unsqueeze(-1)
        coarse = coarse[:, 1:]
        melsNcoarse = torch.cat([mels, coarse_float], dim=-1)

        self.joint_encoder.flatten_parameters()
        hidden,_ = self.joint_encoder(melsNcoarse)
        coarse_hidden, fine_hidden = hidden.split(128, dim=-1)

        coarse_hidden = torch.relu(self.hidden2coarse_hidden(coarse_hidden))
        fine_input = torch.cat([fine_hidden, coarse.unsqueeze(-1).float()], dim=-1)

        self.fine_encoder.flatten_parameters()
        fine_hidden, _ = self.fine_encoder(fine_input)
        fine_hidden = torch.relu(self.hidden2fine_hidden(fine_hidden))

        coarse_logits = self.coarse_hidden2logits_coarse(coarse_hidden)
        fine_logits = self.fine_hidden2logits_fine(fine_hidden)

        return coarse_logits, coarse, fine_logits, fine_float[:, 1:]




class DurationAcousticModel(TacotronOne):

    def __init__(self, n_vocab):
        super(DurationAcousticModel, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
 
        self.r = 5
        self.mel_dim=80  
        self.lstm = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.decoder_fc = nn.Linear(512, self.mel_dim)
       

    def forward(self, x, mel=None):
        B = x.shape[0]

        x = self.embedding(x)
        hidden, _ = self.lstm(x)
        #print("Shape of hidden: ", hidden.shape)
   
        mel_outputs = self.decoder_fc(hidden)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
    
        return mel_outputs, linear_outputs


class DurationTonesAcousticModel(DurationAcousticModel): 
 
    def __init__(self, n_vocab, n_tones): 
        super(DurationTonesAcousticModel, self).__init__(n_vocab) 

        self.tones_embedding = nn.Embedding(n_tones, 256)
        self.tonesNphones2inputs = nn.Linear(512,256)
        self.lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.decoder_fc = nn.Linear(256, self.mel_dim)


    def forward(self, inputs, tones, targets=None, input_lengths=None):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        tones = self.tones_embedding(tones)
        inputs = torch.cat([inputs, tones], dim=-1)
        #print("Shape of inputs to tonesNphones2inputs: ", inputs.shape)
        inputs = torch.tanh(self.tonesNphones2inputs(inputs))

        hidden, _ = self.lstm(inputs)
        hidden = self.encoder(hidden)

        mel_outputs = self.decoder_fc(hidden)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
    
        return mel_outputs, linear_outputs

