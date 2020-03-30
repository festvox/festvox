import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *


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


class SimpleEncoder(nn.Module):

    def __init__(self, in_dim):
       super(SimpleEncoder, self).__init__()
       self.encoder_lstm = nn.LSTM(in_dim, in_dim, bidirectional=True, batch_first=True)
       self.encoder_conv = BatchNormConv1d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, lengths):
       x,h = self.encoder_lstm(x)
       #x = torch.tanh(x)
       x = self.encoder_conv(x.transpose(1,2)).transpose(1,2)
       #print("Shape of x after encoder: ", x.shape)
       return x



class Decoder_MultiSpeakerTacotronOne(Decoder_TacotronOne):
    def __init__(self, in_dim, r):
        super(Decoder_MultiSpeakerTacotronOne, self).__init__(in_dim, r)

        self.combiner = GatedCombinationConv(in_dim*r, in_dim*r, 128)
        self.project_to_decoder_in = nn.Linear(640, 256)

    def forward(self, encoder_outputs, spk_embedding, emb2linear, inputs=None, memory_lengths=None, combine_flag=True):

        B = encoder_outputs.size(0)

        processed_memory = self.memory_layer(encoder_outputs)

        mask = get_mask_from_lengths(processed_memory, memory_lengths)

        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(B, inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r
        T_decoder = inputs.size(1)

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_()) + 2

        # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input

        while True:
            if t > 0:
                current_input = inputs[t - 1]

            # Prenet
            if len(current_input.shape) < 3:
               current_input = current_input.unsqueeze(1)
               #print("Shape of current_input before combining: ", current_input.shape)

            if combine_flag:
               #print("Combining")
               current_input = torch.cat([current_input, spk_embedding.unsqueeze(1)], dim = -1)
               current_input = torch.tanh(emb2linear(current_input))

            current_input = self.prenet(current_input)

            #print("Shape of current_input after combining: ", current_input.shape)
            #print("Shape of Processed memory: ", processed_memory.shape)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(current_input, current_attention, attention_rnn_hidden, encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            #print("Shape of attention_rnn_hidden, current_attention: ", attention_rnn_hidden.shape, current_attention.shape)
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention, spk_embedding), -1))

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

            if t >= T_decoder:
               break

        assert len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments


    def forward_generate(self, encoder_outputs, spk_embedding, emb2linear, combine_flag=False):

        B = encoder_outputs.size(0)

        processed_memory = self.memory_layer(encoder_outputs)

        mask = None

        greedy = True

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_()) + 2

        # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input

        while True:
            if t > 0:
                current_input = outputs[-1]

            # Prenet
            if len(current_input.shape) < 3:
               current_input = current_input.unsqueeze(1)
               #print("Shape of current_input before combining: ", current_input.shape)

            if combine_flag:
               #print("Combining")
               current_input = torch.cat([current_input, spk_embedding.unsqueeze(1)], dim = -1)
               current_input = torch.tanh(emb2linear(current_input))

            current_input = self.prenet(current_input)

            #print("Shape of current_input after combining: ", current_input.shape)
            #print("Shape of Processed memory: ", processed_memory.shape)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(current_input, current_attention, attention_rnn_hidden, encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention, spk_embedding), -1))

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

            if t > 1 and is_end_of_frames(output):
                break
            elif t > self.max_decoder_steps:
                print("Warning! doesn't seems to be converged")
                break


        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments



def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()


class TacotronOneSeqwiseMultispeaker(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, num_spk=2, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwiseMultispeaker, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.num_spk = num_spk
        self.spk_embedding = nn.Embedding(self.num_spk, 128)

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

        # Decoder
        mel_outputs, alignments = self.decoder(encoder_outputs, targets, memory_lengths=input_lengths)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        # PostNet
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments


