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

# Type: Acquisition_CodeBorrowed Source: https://github.com/r9y9/tacotron_pytorch/blob/62db7217c10da3edb34f67b185cc0e2b04cdf77e/tacotron_pytorch/tacotron.py#L277
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

# Type: Indigenous
class TacotronOneSeqwise(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneSeqwise, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneSeqwise(mel_dim, r)

# Type: Indigenous
class TacotronOneFinalFrame(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneFinalFrame, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.decoder = Decoder_TacotronOneFinalFrame(mel_dim, r)

# Type: Indigenous
# Note: Only the CBHG in the encoder is replaced
class TacotronOneLSTMsBlockEncoder(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneLSTMsBlockEncoder, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)
        self.encoder = Encoder_TacotronOne_LSTMsBlock(embedding_dim)

# Type: Indigenous
# Note: CBHG in the encoder and postnet are replaced
class TacotronOneLSTMsBlock(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneLSTMsBlock, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.encoder = Encoder_TacotronOne_LSTMsBlock(embedding_dim)
        self.postnet = LSTMsBlock(mel_dim, mel_dim*2)

# Type: Indigenous
# Note: Vector Quantization in latent space
class TacotronOneVQ(TacotronOneSeqwise):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneVQ, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.num_channels = 1
        self.num_classes = 200
        self.vec_len = 256
        self.normalize = False
        self.quantizer = quantizer_kotha(self.num_channels, self.num_classes, self.vec_len, self.normalize)


    def forward(self, inputs, targets=None, input_lengths=None):

        B = inputs.size(0)

        inputs = self.embedding(inputs)
        encoder_outputs = self.encoder(inputs, input_lengths)

        # Latent Vector Quantization 
        latent_outputs, vq_penalty, encoder_penalty, entropy = self.quantizer(encoder_outputs.unsqueeze(2))
        latent_outputs = latent_outputs.squeeze(2)

        if self.use_memory_mask:
            memory_lengths = input_lengths
        else:
            memory_lengths = None

        mel_outputs, alignments = self.decoder(
            latent_outputs, targets, memory_lengths=memory_lengths)

        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments, vq_penalty.mean(), encoder_penalty.mean(), entropy


# Type: Indigenous
class TacotronOneActNorm(TacotronOne):

    def __init__(self, n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False):
        super(TacotronOneActNorm, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                 r=5, padding_idx=None, use_memory_mask=False)

        self.encoder = Encoder_TacotronOne_ActNorm(embedding_dim)
        self.postnet = CBHGActNorm(mel_dim, K=8, projections=[256, mel_dim])


#Type : Indigenous 
# vocoder part is inspired by https://github.com/mkotha/WaveRNN
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
       
    def forward(self, mels, coarse, coarse_float, fine, fine_float):

        B = mels.size(0)
        outputs = {}

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
        outputs['coarse_logits'] = coarse_logits
        outputs['coarse'] = coarse[:,1:]

        fine_logits = self.fine_hidden2logits_fine(fine_hidden)
        outputs['fine_logits'] = fine_logits
        outputs['fine'] = fine[:,1:] 

        return outputs

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
        
           if i%10000 == 1:
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

    
