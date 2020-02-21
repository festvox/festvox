import os, sys
from hparams_arctic import hparams, hparams_debug_string

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from models import *
from blocks import BatchNormConv1d


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


class Decoder_TacotronOneSeqwise(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_TacotronOneSeqwise, self).__init__(in_dim, r)
        self.prenet = Prenet_seqwise(in_dim, sizes=[256, 128])
        self.decoder_inputsNspk_embedding2decoder_inputs = nn.Linear(in_dim + 128, in_dim)

    def forward(self, encoder_outputs, spk_embedding, inputs=None, memory_lengths=None):

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
            encoder_outputs.data.new(B, self.in_dim ).zero_()) # Picking just the last frame

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

            #print("Shape of current_input before combining with spk_embedding: ", current_input.shape) 
            # Add speaker
            ####### Sai Krishna Rallabandi 16 December 2019 #####################
            current_input = torch.cat([current_input, spk_embedding], dim=-1)
            current_input = torch.tanh(self.decoder_inputsNspk_embedding2decoder_inputs(current_input))
            current_input = current_input.unsqueeze(1)
            #################################################################
            #print("Shape of current_input before prenet: ", current_input.shape) 

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


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).cuda()

    def forward(self, tensor):

        return tensor + torch.randn(tensor.size()).cuda() * self.sigma #+ self.mean

        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size().cpu()).normal_() * scale
            x = x + sampled_noise
        return x


class ZeroSpeechVQVAE(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_spk=2, padding_idx=None, use_memory_mask=False):
        super(ZeroSpeechVQVAE, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            #(2, 4, 1),
            #(1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(80, encoder_layers)

        self.spk_embedding = nn.Embedding(num_spk, 128)

        self.lstm1_encoder = nn.LSTM(80, 128, batch_first=True, bidirectional=True)
        self.lstm2_encoder = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.encoder_outputs_quantized_to_decoder_inputs = SequenceWise(nn.Linear(256,256))

        self.n_channels = 1
        self.num_classes = 100
        self.vector_length = 80
        self.normalize=False
        self.quantizer_scale = 0.006
        self.quantizer = quantizer_kotha(self.n_channels, self.num_classes, self.vector_length, self.normalize, self.quantizer_scale)

        self.targetsNspkembedding2decodeinputs = SequenceWise(nn.Linear(208, 80))

        self.jitter = GaussianNoise()

    def forward(self, spk, targets, input_lengths=None):

        B = spk.size(0)

        spk_embedding = self.spk_embedding(spk)

        # Encoder
        encoder_outputs = self.encoder(targets)

        # Vector Quantization
        encoder_outputs_quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(encoder_outputs.unsqueeze(2))
        encoder_outputs_quantized = encoder_outputs_quantized.squeeze(2)


        if input_lengths is not None:
            encoder_outputs_quantized = nn.utils.rnn.pack_padded_sequence(
                encoder_outputs_quantized, input_lengths, batch_first=True)

        encoder_outputs_desired, _ = self.lstm1_encoder(encoder_outputs_quantized)
        encoder_outputs_desired, _ = self.lstm2_encoder(encoder_outputs_desired)


        if input_lengths is not None:
            encoder_outputs_desired, _ = nn.utils.rnn.pad_packed_sequence(
                encoder_outputs_desired, batch_first=True)

        encoder_outputs_desired = torch.tanh(self.encoder_outputs_quantized_to_decoder_inputs(encoder_outputs_desired))

        # Decoder
        modified_targets = self.jitter(targets)
        mel_outputs, alignments = self.decoder(encoder_outputs_desired, spk_embedding, modified_targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments, vq_penalty.mean(), encoder_penalty.mean(), entropy

    def forward_generate(self, targets, spk, input_lengths=None):

        B = spk.size(0)

        spk_embedding = self.spk_embedding(spk)

        # Encoder
        encoder_outputs = self.encoder(targets)

        # Vector Quantization
        encoder_outputs_quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(encoder_outputs.unsqueeze(2))
        encoder_outputs_quantized = encoder_outputs_quantized.squeeze(2)

        encoder_outputs_desired, _ = self.lstm1_encoder(encoder_outputs_quantized)
        encoder_outputs_desired, _ = self.lstm2_encoder(encoder_outputs_desired)
        encoder_outputs_desired = torch.tanh(self.encoder_outputs_quantized_to_decoder_inputs(encoder_outputs_desired))
        targets = None

        # Decoder
        mel_outputs, alignments = self.decoder(encoder_outputs_desired, spk_embedding, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments 


    def forward_v2(self, spk, targets, input_lengths=None):

        B = spk.size(0)

        spk_embedding = self.spk_embedding(spk)

        # Encoder
        encoder_outputs = self.encoder(targets)

        # Vector Quantization
        encoder_outputs_quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(encoder_outputs.unsqueeze(2))
        encoder_outputs_quantized = encoder_outputs_quantized.squeeze(2)
        encoder_outputs_desired = self.encoder_outputs_quantized_to_decoder_inputs(encoder_outputs_quantized)
 
        # Decoder
        spk_embedding = spk_embedding.unsqueeze(1).expand(-1, targets.size(1), -1)
        targets = torch.cat([targets, spk_embedding],dim=-1)
        targets = self.targetsNspkembedding2decodeinputs(targets)
        mel_outputs, alignments = self.decoder(encoder_outputs_desired, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments, vq_penalty.mean(), encoder_penalty.mean(), entropy

    def forward_generate_v2(self, targets, spk, input_lengths=None):

        B = spk.size(0)

        spk_embedding = self.spk_embedding(spk)

        # Encoder
        encoder_outputs = self.encoder(targets)

        # Vector Quantization
        encoder_outputs_quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(encoder_outputs.unsqueeze(2))
        encoder_outputs_quantized = encoder_outputs_quantized.squeeze(2)
        encoder_outputs_desired = self.encoder_outputs_quantized_to_decoder_inputs(encoder_outputs_quantized)
        targets = None
        # Decoder
        #spk_embedding = spk_embedding.unsqueeze(1).expand(-1, targets.size(1), -1)
        #targets = torch.cat([targets, spk_embedding],dim=-1)
        #targets = self.targetsNspkembedding2decodeinputs(targets)
        mel_outputs, alignments = self.decoder(encoder_outputs_desired, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments 

