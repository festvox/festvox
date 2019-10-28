import numpy as np

import torch
from torch.autograd import Variable
from torch import nn

from blocks import *
from layers  import *

'''Excerpts from the following sources
# https://github.com/r9y9/tacotron_pytorch/blob/master/tacotron_pytorch/tacotron.py

'''

print_flag = 0

class Encoder_TacotronOne(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)

class Encoder_TacotronOne_Tones(nn.Module):
    def __init__(self, in_dim):
        super(Encoder_TacotronOne, self).__init__()
        self.prenet = Prenet_tones(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, tones, input_lengths=None):
        inputs = self.prenet(inputs, tones)
        return self.cbhg(inputs, input_lengths)


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


    def forward_multispeaker(self, encoder_outputs, spk_batch, inputs=None, memory_lengths=None):
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
        #print("Shape of encoders and spk: ", encoder_outputs.shape, spk_batch.shape) 
        assert B == spk_batch.shape[0]

        spk_embedding = self.spk_embedding(spk_batch)
        spk_embedding = self.spk_linear(spk_embedding)

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
        #print("Shape of initial_input and the speaker embedding: ", initial_input.shape, spk_embedding.shape)
        #### Cat the speaker embedding #########
        initial_input = torch.cat([initial_input, spk_embedding], dim = 1)
        current_input = torch.tanh(self.cond2inp(initial_input))
        ########################################

        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else inputs[t - 1]
                current_input = torch.cat([current_input, spk_embedding], dim = 1)        
                current_input = torch.tanh(self.cond2inp(current_input))
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

            #print("Shape of decoder input and spk_embeding: ", decoder_input.shape, spk_embedding.shape)
            decoder_output = torch.cat([decoder_input, spk_embedding], dim = -1)
            output = self.proj_to_mel(decoder_output)

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

        return outputs, alignments, spk_embedding




def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class TacotronOne(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOne, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        self.embedding = nn.Embedding(n_vocab, embedding_dim,
                                      padding_idx=padding_idx)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder_TacotronOne(embedding_dim)
        self.decoder = Decoder_TacotronOne(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    '''
    Section 4 in 'TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS' https://arxiv.org/pdf/1703.10135.pdf
      Its a common practice to train sequence models with a loss mask, which masks loss on zero-padded frames.
      However, we found that models trained this way dont know when to stop emitting outputs, causing
      repeated sounds towards the end. One simple trick to get around this problem is to also reconstruct
      the zero-padded frames.
    '''
    def forward(self, inputs, targets=None, input_lengths=None):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)
        #print("Shape of encoder outputs: ", encoder_outputs.shape)
        memory_lengths = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments



    def forward_nomasking_multispeaker(self, inputs, spk, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)
        #print("Shape of encoder outputs: ", encoder_outputs.shape)
        memory_lengths = None

        mel_outputs, alignments = self.decoder.forward_multispeaker(
            encoder_outputs, spk, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments



class TacotronOne_tones(nn.Module):
    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOne_tones, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.use_memory_mask = use_memory_mask
        self.embedding = nn.Embedding(n_vocab, embedding_dim,
                                      padding_idx=padding_idx)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder_TacotronOne(embedding_dim)
        self.decoder = Decoder_TacotronOne(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

        self.toneNchar2embedding = SequenceWise(nn.Linear(embedding_dim*2,embedding_dim))

    def forward(self, inputs, tones, targets=None, input_lengths=None):
        B = inputs.size(0)
 
        inputs = self.embedding(inputs)

        '''Sai Krishna Rallabandi
        This might not be the best place to handle concatenation of inputs and tones
        '''
        #print("Shapes of inputs and tones: ", inputs.shape, tones.shape)
        assert inputs.shape[1] == tones.shape[1]
        inputs_tones = self.embedding(tones)
        inputs = torch.cat([inputs, inputs_tones], dim=-1)
        inputs = self.toneNchar2embedding(inputs)

        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None
        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


    def forward_nomasking(self, inputs, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)

        memory_lengths = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments



class Wavenet_Barebones(nn.Module):

    def __init__(self):
        super(Wavenet_Barebones, self).__init__()

        self.embedding = nn.Embedding(259, 128)
        self.encoder_fc = SequenceWise(nn.Linear(128, 128))
        self.encoder_dropout = nn.Dropout(0.3)
        self.kernel_size = 3
        self.stride = 1

        layers = 24
        stacks = 4
        layers_per_stack = layers // stacks

        self.conv_modules = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            self.padding = int((self.kernel_size - 1) * dilation)
            conv = residualconvmodule(128,128, self.kernel_size, self.stride, self.padding,dilation)
            self.conv_modules.append(conv)

        self.final_fc1 = SequenceWise(nn.Linear(128, 512))
        self.final_fc2 = SequenceWise(nn.Linear(512, 259))
        
        self.upsample_fc = SequenceWise(nn.Linear(80,60))
        upsample_factors = [4,4,4,4]
        self.upsample_network = UpsampleNetwork_r9y9(80, upsample_factors)        

    def encode(self, x, teacher_forcing_ratio):
        x = self.embedding(x.long())
        if len(x.shape) < 3:
           x = x.unsqueeze(1)
        x = F.relu(self.encoder_fc(x))
        if teacher_forcing_ratio > 0.1:
           #print("Dropping out")
           x = self.encoder_dropout(x)
        return x

    def upsample_ccoeffs(self, c, frame_period=80):
        c = self.upsample_network(c)
        print("Shape of upsampled c: ", c.shape)
        return c

        if print_flag:
           print("Shape of ccoeffs in upsampling routine is ", c.shape)
        c = c.transpose(1,2)
        c = F.interpolate(c, size=[c.shape[-1]*frame_period])
        c = c.transpose(1,2)
        if print_flag:
           print("Shape of ccoeffs after upsampling is ", c.shape)
        c = self.upsample_fc(c)   
        return c #[:,:-1,:]


    def upsample_ccoeffs_conv(self, c, frame_period=80):
        c = self.upsample_network(c)
        return c


    def forward(self,x, c, tf=1):


       # Do something about the wav
       x = self.encode(x.long(), 1.0)

       # Do something about the ccoeffs
       frame_period = 256
       c = self.upsample_ccoeffs(c, frame_period)       

       # Feed to Decoder
       x = x.transpose(1,2)
       for module in self.conv_modules:
          x = F.relu(module(x, c))

       x = x.transpose(1,2)

       x = F.relu(self.final_fc1(x))
       x = self.final_fc2(x)

       return x[:,:-1,:]
 

    def forward_convupsampling(self,x, c, tf=1):


       # Do something about the wav
       x = self.encode(x.long(), 1.0)

       # Do something about the ccoeffs
       frame_period = 256
       c = self.upsample_ccoeffs_conv(c, frame_period)       

       # Feed to Decoder
       x = x.transpose(1,2)
       for module in self.conv_modules:
          x = module(x, c)

       x = x.transpose(1,2)

       x = torch.tanh(self.final_fc1(x))
       x = self.final_fc2(x)

       return x[:,:-1,:]

    def clear_buffers(self):

       for module in self.conv_modules:
           module.clear_buffer()

    def forward_incremental(self, c):
 
       self.clear_buffers()

       max_length = c.shape[1] * 256
       print("Max Length is ", max_length)

       bsz = c.shape[0]
       x = c.new(bsz,1)
       a = 0
       x.fill_(a)

       outputs = []
       samples = []

       # Do something about the ccoeffs
       frame_period = 256
       if print_flag:
          print(" Model: Shape of c before upsampling: ", c.shape)
       c = self.upsample_ccoeffs_conv(c.transpose(1,2), frame_period).transpose(1,2)
       if print_flag:
          print(" Model: Shape of c after upsampling: ", c.shape)

       for i in range(max_length-1):

          # Do something about the wav
          x = self.encode(x.long(), 0.0)

          # Feed to Decoder
          ct = c[:,i,:].unsqueeze(1)

          assert len(x.shape) == 3

          for module in self.conv_modules:
             x = module.incremental_forward(x, ct)

          #x = x.transpose(1,2)
          if print_flag:
             print(" Model: Shape of output from the modules: ", x.shape)  
          x = torch.tanh(self.final_fc1(x))
          x = self.final_fc2(x)

          probs = F.softmax(x.view(bsz, -1), dim=1)
          predicted = torch.max(x.view(bsz, -1), dim=1)[1]
          sample = np.random.choice(np.arange(259), p = probs.view(-1).data.cpu().numpy())
          predicted = np.array([sample])
          sample_onehotk = x.new(x.shape[0], 259)
          sample_onehotk.zero_()
          sample_onehotk[:,predicted] = 1
          outputs.append(x)
          #samples.append(predicted)
          samples.append(sample_onehotk)
          x = torch.LongTensor(predicted).cuda()

       outputs = torch.stack(outputs)
       samples = torch.stack(samples)
       return outputs



class Encoder_TacotronOneSeqwise(Encoder_TacotronOne):
    def __init__(self, in_dim):
        super(Encoder_TacotronOneSeqwise, self).__init__(in_dim)
        self.prenet = Prenet_seqwise(in_dim, sizes=[256, 128])


class Decoder_TacotronOneSeqwise(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneSeqwise, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim * r, sizes=[256, 128])


class Decoder_TacotronOneMultispeaker(Decoder_TacotronOneSeqwise):
    def __init__(self, in_dim, r, num_spk):
        super(Decoder_TacotronOneMultispeaker, self).__init__(in_dim, r)
        self.spkemb_dim = 128
        self.spk_embedding = nn.Embedding(num_spk, self.spkemb_dim)
        self.spk_linear = nn.Linear(self.spkemb_dim, self.spkemb_dim)
        self.cond2inp = nn.Linear(in_dim * r + self.spkemb_dim  , in_dim * r)
        self.proj_to_mel = nn.Linear(256 + self.spkemb_dim, in_dim * r)


class TacotronOneSeqwise(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwise, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        #self.encoder = Encoder_TacotronOneSeqwise(embedding_dim)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)


class TacotronOneMultispeaker(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_spk = 5,  padding_idx=None, use_memory_mask=False):
        super(TacotronOneMultispeaker, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneMultispeaker(mel_dim, r, num_spk)
        self.last_linear = nn.Linear(mel_dim * 2 + self.decoder.spkemb_dim, linear_dim)
       

    def forward_nomasking_multispeaker(self, inputs, spk, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        encoder_outputs = self.encoder(inputs, input_lengths)
        memory_lengths = None

        mel_outputs, alignments, spk_embedding = self.decoder.forward_multispeaker(
            encoder_outputs, spk, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        #spk_embedding = spk_embedding.repeat(1, linear_outputs.shape[1]).view(spk_embedding.shape[0], linear_outputs.shape[1], spk_embedding.shape[1])
        spk_embedding = spk_embedding.unsqueeze(-1) if spk_embedding.dim() == 2 else g
        spk_embedding = spk_embedding.expand(B, -1, linear_outputs.shape[1]).transpose(1,2).contiguous()
        #print(spk_embedding.shape, linear_outputs.shape)
        assert spk_embedding.shape[1] == linear_outputs.shape[1]
        linear_outputs = torch.cat([linear_outputs, spk_embedding], dim = -1)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


class TacotronOneSeqwiseVQ(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseVQ, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.num_classes = 60
        self.quantizer = quantizer_kotha(1, self.num_classes, 256, normalize=True)

    def forward_nomasking(self, inputs, targets=None, input_lengths=None):
        B = inputs.size(0)

        inputs = self.embedding(inputs)
        inputs_quantized = inputs.unsqueeze(2)
        inputs_quantized, vq_pen, encoder_pen, entropy = self.quantizer(inputs_quantized)
        inputs = inputs_quantized.squeeze(2)

        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs, input_lengths)
        #print("Shape of encoder outputs: ", encoder_outputs.shape)
        memory_lengths = None

        mel_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments,  vq_pen.mean(), encoder_pen.mean(), entropy

    def get_indices(self, inputs):

        B = inputs.size(0)

        inputs = self.embedding(inputs)

        return self.quantizer.get_quantizedindices(inputs.unsqueeze(2))


