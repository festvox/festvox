import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from layers import *
'''Excerpts from the following sources
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py

'''

print_flag = 0

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


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


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
        alignment = F.softmax(alignment)

        # Attention context vector
        # (batch, 1, dim)
        attention = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        attention = attention.squeeze(1)

        return cell_output, attention, alignment

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


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

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
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


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


