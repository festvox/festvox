import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import os
from layers import *
import random
from torch.nn import init

'''Excerpts from the following sources
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py

'''

print_flag = 0
cuda_version = None
if "CUDAVERSION" in os.environ:
    cuda_version = float(os.environ.get('CUDAVERSION'))
 

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/attention.py#L7
class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, processed_memory):
        """
        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)

        # (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        # (batch, max_time)
        return alignment.squeeze(-1)

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/attention.py#L33
def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    # Changing based on https://github.com/jiesutd/NCRFpp/issues/137 This is not currently handled neatly. Will revisit
    if cuda_version is not None and cuda_version >= 9.2:
       mask = memory.data.new(memory.size(0), memory.size(1)).bool().zero_()
    else:
       try:
          mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
       except Exception as e:
          print("Think you should check cuda version. Mask calculation needs this to be specified explicitly. ")
          sys.exit()

    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask

# Type: Indigenous Based on https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/attention.py#L33
def get_floatmask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).zero_()

    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return mask


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

# Type: Acquisition_CodeBorrowed Source:https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L12
class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs

# Type: Indigenous
class Prenet_seqwise(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet_seqwise, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [SequenceWise(nn.Linear(in_size, out_size))
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            #print("Shape of inputs: ", inputs.shape)
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs

# Type: Indigenous
class Prenet_tones(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs, inputs_tones):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L28
class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L45
class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L61
class CBHG(nn.Module):

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs

# Type: Indigenous
class LSTMsBlock(nn.Module):
    """Replacement for the CBHG module
        - 3 bidirectional lstm layers
    """

    def __init__(self, in_dim, linear_dim=None):
        super(LSTMsBlock, self).__init__()
        self.in_dim = in_dim
        self.lstm_i = nn.LSTM(self.in_dim, self.in_dim, bidirectional=True, batch_first=True)
        self.lstm_h = nn.LSTM(self.in_dim*2, self.in_dim, bidirectional=True, batch_first=True)
        self.lstm_o = nn.LSTM(self.in_dim*2, 256, bidirectional=True, batch_first=True)
        if linear_dim:
           self.linear_dim = linear_dim
        else:
            self.linear_dim = 256
        self.final_linear = SequenceWise(nn.Linear(512, self.linear_dim))

    def forward(self, x, input_lengths=None):

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        x, hidden = self.lstm_i(x)
        x, hidden = self.lstm_h(x)
        outputs, hidden = self.lstm_o(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        outputs = torch.tanh(self.final_linear(outputs))
        return outputs

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/wavenet_vocoder
class residualconvmodule(nn.Module):

    def __init__(self,  in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(residualconvmodule, self).__init__()

        self.conv = self.weightnorm_conv1d( in_channels, out_channels, kernel_size, stride, padding, dilation )
        self.prefinal_fc = SequenceWise(nn.Linear(128, 128))


    def weightnorm_conv1d(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        dropout = 0
        std_mul = 1.0
        m = Conv1dplusplus(in_channels,out_channels, kernel_size=kernel_size, stride=stride, padding = padding, dilation = dilation)
        std = math.sqrt((std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
        m.weight.data.normal_(mean=0, std=std)
        m.bias.data.zero_()
        return nn.utils.weight_norm(m)

    def clear_buffer(self):
        self.conv.clear_buffer()


    def forward(self, x, c, g=None):
        return self._forward(x, c, g, False)


    def incremental_forward(self, x, c=None, g=None):
        return self._forward(x, c, g, True)


    def _forward(self,x, c, g, incremental_flag):

        residual = x        

        # Feed to the module
        if incremental_flag:
           if print_flag:
              print("   Module: The shape of residual in the module is ", residual.shape, " and that of x is ", x.shape) 
           assert residual.shape[1] == x.shape[1]
           x = F.relu(self.conv.incremental_forward(x))

        else:
           x = F.relu(self.conv(x))
           x = x.transpose(1,2)
           # Causal
           x = x[:,:residual.shape[2],:] 

        if print_flag:
           print("   Module: The shape of residual in the module is ", residual.shape)
           print("   Module: Shape of x after residual convs is ", x.shape)
           print("   Module: Shape of x before prefinal fc is ", x.shape)
 
        x = self.prefinal_fc(x)

        if incremental_flag:
           pass
        else:
           x = x.transpose(1,2)

        if print_flag:
           print("   Module: Shape of x right before adding the residual and the residual: ", x.shape, residual.shape)
        assert x.shape == residual.shape

        x = (x + residual) * math.sqrt(0.5)

        return x

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/wavenet_vocoder
class UpsampleNetwork_r9y9(nn.Module):

     def __init__(self, feat_dims, upsample_scales):
         super().__init__()
         self.upsample_conv = nn.ModuleList()
         freq_axis_kernel_size=3
         weight_normalization = True
         for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace=True))


     def forward(self, c):

            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1) 
            return c

# Type: Acquisition_CodeBorrowed Source: https://github.com/mkotha/WaveRNN/blob/a06e6b867592654d123fd6c57c755d02db3bf7ec/layers/vector_quant.py#L7
class quantizer_kotha(nn.Module):
    """
        Input: (B, T, n_channels, vec_len) numeric tensor n_channels == 1 usually
        Output: (B, T, n_channels, vec_len) numeric tensor
    """
    def __init__(self, n_channels, n_classes, vec_len, normalize=False, scale=None):
        super().__init__()
        if normalize:
            target_scale = scale if scale is not None else  0.06
            self.embedding_scale = target_scale
            self.normalize_scale = target_scale
        else:
            self.embedding_scale = 1e-3 #1e-3
            self.normalize_scale = None
        self.embedding0 = nn.Parameter(torch.randn(n_channels, n_classes, vec_len, requires_grad=True) * self.embedding_scale)
        self.offset = torch.arange(n_channels).cuda() * n_classes
        # self.offset: (n_channels) long tensor
        self.n_classes = n_classes
        self.after_update()


    def forward(self, x0):
        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0
        #logger.log(f'std[x] = {x.std()}')
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        # x1: (N*samples, n_channels, 1, vec_len) numeric tensor
        #print("Shape of x1 and embedding: ", x1.shape, embedding.shape)

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        # index: (N*samples, n_channels) long tensor
        if True: # compute the entropy
            hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
            prob = hist.masked_select(hist > 0) / len(index)
            entropy = - (prob * prob.log()).sum().item()
            #logger.log(f'entrypy: {entropy:#.4}/{math.log(self.n_classes):#.4}')
        else:
            entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        return (out0, out1, out2, entropy)


    def forward_modified(self, x0):

        if torch.isnan(x0).any():
           print("Something went wrong here with x0")
           sys.exit()


        if self.normalize_scale:
            target_norm = self.normalize_scale * math.sqrt(x0.size(3))
            x = target_norm * x0 / x0.norm(dim=3, keepdim=True)
            embedding = target_norm * self.embedding0 / self.embedding0.norm(dim=2, keepdim=True)
        else:
            x = x0
            embedding = self.embedding0

        if torch.isnan(torch.Tensor([target_norm])).any():
           print("Something went wrong here with target norm")
           sys.exit()


        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        #print("Shape of x and x1: ", x.shape, x1.shape)

        # Perform chunking to avoid overflowing GPU RAM.
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        entropy = 0
        index1 = (index + self.offset).view(index.size(0) * index.size(1))
        #print("Shape of x1_chunk and embedding: ", x1_chunk.shape, embedding.shape)
        #print("Shape of index1: ", index1.shape)

        # index1: (N*samples*n_channels) long tensor
        output_flat = embedding.view(-1, embedding.size(2)).index_select(dim=0, index=index1)
        # output_flat: (N*samples*n_channels, vec_len) numeric tensor
        #print("Shapes of output_flat and x: ", output_flat.shape, x.shape)
        output = output_flat.view(x.size())

        out0 = (output - x).detach() + x
        out1 = (x.detach() - output).float().norm(dim=3).pow(2)
        out2 = (x - output.detach()).float().norm(dim=3).pow(2) + (x - x0).float().norm(dim=3).pow(2)
        #logger.log(f'std[embedding0] = {self.embedding0.view(-1, embedding.size(2)).index_select(dim=0, index=index1).std()}')
        #print("Shape of out0: ", out0.shape)

        return (out0.squeeze(2), out1, out2, entropy)

    def after_update(self):
        if self.normalize_scale:
            with torch.no_grad():
                target_norm = self.embedding_scale * math.sqrt(self.embedding0.size(2))
                self.embedding0.mul_(target_norm / self.embedding0.norm(dim=2, keepdim=True))


    def get_quantizedindices(self, x0):

        x = x0
        embedding = self.embedding0
        x1 = x.reshape(x.size(0) * x.size(1), x.size(2), 1, x.size(3))
        index_chunks = []
        for x1_chunk in x1.split(512, dim=0):
            index_chunks.append((x1_chunk - embedding).norm(dim=3).argmin(dim=2))
        index = torch.cat(index_chunks, dim=0)
        hist = index.float().cpu().histc(bins=self.n_classes, min=-0.5, max=self.n_classes - 0.5)
        prob = hist.masked_select(hist > 0) / len(index)
        entropy = - (prob * prob.log()).sum().item()
        arr = (index.squeeze(1) + self.offset).cpu().numpy().tolist()
        #print("Predicted quantized Indices are: ", (index.squeeze(1) + self.offset).cpu().numpy())
        #print('\n')
        return self.deduplicate(arr), entropy

    # Remove repeated entries
    def deduplicate(self, arr):
       arr_new = []
       current_element = None
       for element in arr:
          if current_element is None:
            current_element = element
            arr_new.append(element)
          elif element == current_element:
            continue
          else:
            current_element = element
            arr_new.append(element)
       return arr_new


# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L139
class Encoder_TacotronOne(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)

# Type: Indigenous
class Encoder_TacotronOne_nocudnn(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne_nocudnn, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)

# Type: Indigenous
class Encoder_TacotronOne_Tones(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne, self).__init__()
        self.prenet = Prenet_tones(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, tones, input_lengths=None):
        inputs = self.prenet(inputs, tones)
        return self.cbhg(inputs, input_lengths)

# Type: Indigenous
class Encoder_TacotronOne_LSTMsBlock(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne_LSTMsBlock, self).__init__()

        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.recurrent_block = LSTMsBlock(128)

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.recurrent_block(inputs, input_lengths)

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L150
class Decoder_TacotronOne(nn.Module):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOne, self).__init__()
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

# Type: Indigenous
class Decoder_TacotronOneSeqwise(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneSeqwise, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim * r, sizes=[256, 128])

# Type: Indigenous
class Decoder_TacotronOneFinalFrame(Decoder_TacotronOneSeqwise):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneFinalFrame, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim, sizes=[256, 128])
        self.in_dim = in_dim

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):

        B = encoder_outputs.size(0)
      
        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        if inputs is not None:
            # Grouping multiple frames if necessary
            if inputs.size(-1) == self.in_dim:
                inputs = inputs.view(B, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == self.in_dim * self.r
            T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim).zero_())

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
                current_input = current_input[:,-self.in_dim:]
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

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/wavenet_vocoder
class GatedCombinationConv(nn.Module):

    def __init__(self, in_channels, out_channels, global_cond_channels):
        super().__init__()
        ksz = 3
        self.out_channels = out_channels
        if 0 < global_cond_channels:
            self.w_cond = nn.Linear(global_cond_channels, 2 * out_channels, bias=False)
        self.conv_wide = nn.Conv1d(in_channels, 2 * out_channels, ksz, stride=1, padding=1)
        wsize = 2.967 / math.sqrt(ksz * in_channels)
        self.conv_wide.weight.data.uniform_(-wsize, wsize)
        self.conv_wide.bias.data.zero_()

    def forward(self, x, global_cond):
        x1 = self.conv_wide(x.transpose(1, 2)).transpose(1, 2)
        if global_cond is not None:
            x2 = self.w_cond(global_cond).unsqueeze(1).expand(-1, x1.size(1), -1)
        else:
            x2 = torch.zeros_like(x1)
        a, b = (x1 + x2).split(self.out_channels, dim=2)
        return torch.sigmoid(a) * torch.tanh(b)

# Type: Indigenous
def reparameterize(self, mu, sigma):

    std = torch.exp(0.5*sigma)
    eps = torch.rand_like(std)
    z = eps.mul(std).add_(mu)
    return z

# Type: Indigenous
class Decoder_TacotronOneVQ(Decoder_TacotronOneSeqwise):

    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneVQ, self).__init__(in_dim, r)
        self.project_attnNlatent_to_attn = nn.Linear(512,256)

    def forward(self, encoder_outputs, latent_output, inputs=None, memory_lengths=None):

        B = encoder_outputs.size(0)

        latent_output = latent_output.squeeze(1)

        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        if inputs is not None:
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
             
            current_attention = torch.cat([current_attention, latent_output], dim=-1)
            current_attention = torch.tanh(self.project_attnNlatent_to_attn(current_attention))

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

# Type: Indigenous
class LSTMDiscriminator(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(self.in_dim*2, self.hidden_dim, batch_first=True)
        self.cbhg = CBHG(self.in_dim, K=16, projections=[128, 128])
        self.output_linear = SequenceWise(nn.Linear(self.hidden_dim, self.out_dim))
        self.drop = nn.Dropout(0.5)

    def forward(self,x):
        x = self.cbhg(x)
        #x = self.drop(x)
        x, _ = self.lstm(x)
        #x = self.drop(x)
        x = self.output_linear(x)
        return x[:,-1,:]



# Type Acquisition_IdeaBorrowed Source: https://arxiv.org/abs/1807.03039
class ActNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, choice=True):
        super(ActNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.scale = nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.bias =  nn.Parameter(torch.Tensor(num_features), requires_grad=True)
        self.num_features = num_features
        self.initialized = None
        self.register_parameter('scale', self.scale)
        self.register_parameter('bias', self.bias)

        #self.choice = choice

        #init.kaiming_uniform_(self.scale, a=math.sqrt(5))
        #fan_in, _ = init._calculate_fan_in_and_fan_out(self.scale)
        #bound = 1 / math.sqrt(fan_in)
        #init.uniform_(self.bias, -bound, bound)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        if 'scale' not in state_dict:
            state_dict['scale'] = self.scale
            state_dict['bias'] = self.bias

        super(ActNorm1d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


    def forward(self, input):

        self._check_input_dim(input)

        if self.initialized is None:
             std = input.std([0, 2]) # Not sure about unbiases vs biased
             self.scale.data = torch.pow(std, -1)
             input = input * self.scale[None,:,None]
             mean = input.mean([0, 2])
             self.bias.data = -1.0 * mean
             input = input + self.bias[None,:,None]
             self.initialized = 1
             assert input.mean([0,2]).sum().item() < 0.1 # Zero mean and unit variance for initial minibatch
        else:
             input = input * self.scale[None,:,None] + self.bias[None,:,None]

        return input

# Type Indigenous
class ActNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(ActNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.an = ActNorm1d(out_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.an(x)

# Type: Indigenous
class CBHGActNorm(CBHG):

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHGActNorm, self).__init__(in_dim, K, projections)

        self.tanh = nn.Tanh() 
        self.conv1d_banks = nn.ModuleList(
            [ActNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.tanh)
             for k in range(1, K + 1)])
        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.tanh] * (len(projections) - 1) + [None]

        self.conv1d_projections = nn.ModuleList(
            [ActNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])


# Type: Indigenous
class Encoder_TacotronOne_ActNorm(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne_ActNorm, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHGActNorm(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)

