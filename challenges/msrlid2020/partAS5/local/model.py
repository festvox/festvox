import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from models import *
from layers import *
from util import *

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
            #print(i, "Stride, ksz, DF and shape of input: ", stride, ksz, dilation_factor, x.shape)
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

class Attention(nn.Module):

    def __init__(self, dim):
       super(Attention, self).__init__()
       self.attention_fc = nn.Linear(dim, 1)

    def forward(self, decoded):

        processed = torch.tanh(self.attention_fc(decoded))
        alignment = F.softmax(processed,dim=-1)
        attention = torch.bmm(alignment.transpose(1,2), decoded)
        attention = attention.squeeze(1)
        return attention

class MaskedAttention(Attention):

    def __init__(self, dim):
       super(MaskedAttention, self).__init__(dim)

       self.attention_bias = nn.Parameter(torch.Tensor(dim), requires_grad=True)

    def forward(self, encoder_outputs, lengths):

        processed = torch.tanh(self.attention_fc(encoder_outputs + self.attention_bias[None, None,:]))
        #print(processed)
        #sys.exit()

        if lengths is not None:
          mask = get_floatmask_from_lengths(encoder_outputs, lengths)
          mask = mask.view(processed.size(0), -1, 1)
          processed.data.masked_fill_(mask == 0.0, -2000.0)

        alignment = F.softmax(processed,dim=-1)
        if isnan(alignment):
           print("NANs in alignment")
           sys.exit()

        #print(alignment)
        attention = torch.bmm(alignment.transpose(1,2), encoder_outputs)
        attention = attention.squeeze(1)
        return attention

# masked_fill(mask == 1, float(0.0))

class LIDlatents(TacotronOne):

    def __init__(self, n_vocab):
       super(LIDlatents, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                r=5, padding_idx=None, use_memory_mask=False)

       self.decoder_lstm = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
       self.linear2logits = nn.Linear(32, 2)
       #self.encoder = Encoder_TacotronOne_ActNorm(256)
       self.attention = Attention(128)
       self.attention2linear = nn.Linear(128, 32)

       self.drop = nn.Dropout(0.0)

    def forward(self, latents, lengths=None):

        B = latents.size(0)
        T = latents.size(1)

        latents = self.drop(self.embedding(latents))
        encoder_outputs = self.encoder(latents, lengths)

        self.decoder_lstm.flatten_parameters()
        decoded, _ = self.decoder_lstm(encoder_outputs)

        # Attention pooling
        attention = self.attention(decoded)
        #mask = get_mask_from_lengths(decoded, lengths)
        #processed = torch.tanh(self.attention_fc(decoded))
        #mask = mask.view(processed.size(0), -1, 1)
        #processed.data.masked_fill_(mask, -float("inf"))
        #alignment = F.softmax(processed,dim=-1)
        #attention = torch.bmm(alignment.transpose(1,2), decoded)
        #attention = attention.squeeze(1)
        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)


class LIDlatentsB(TacotronOne):

    def __init__(self, n_vocab):
       super(LIDlatentsB, self).__init__(n_vocab, embedding_dim=256, mel_dim=80, linear_dim=1025,
                r=5, padding_idx=None, use_memory_mask=False)

       self.decoder_lstm = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
       self.linear2logits = nn.Linear(128, 2)

       self.attention_fc1 = nn.Linear(128, 1)
       self.attention_fc2 = nn.Linear(128, 1)

       self.relu = nn.ReLU()
       self.conv1x1 = BatchNormConv1d(128,128, kernel_size=1, stride=1, padding=0, activation = self.relu)

    def forward(self, latents, lengths=None):

        B = latents.size(0)
        T = latents.size(1)

        latents = self.embedding(latents)
        encoder_outputs = self.encoder(latents, lengths)

        self.decoder_lstm.flatten_parameters()
        decoded, _ = self.decoder_lstm(encoder_outputs)
        #decoded = self.conv1x1(decoded.transpose(1,2)).transpose(1,2)

        # Attention pooling
        #mask = get_mask_from_lengths(decoded, lengths)
        processed1 = torch.tanh(self.attention_fc1(decoded))
        processed2 = torch.tanh(self.attention_fc2(decoded))

        #mask = mask.view(processed.size(0), -1, 1)
        #processed.data.masked_fill_(mask, -float("inf"))

        alignment1 = F.softmax(processed1,dim=-1)
        alignment2 = F.softmax(processed2,dim=-1)

        attention1 = torch.bmm(alignment1.transpose(1,2), decoded)
        attention2 = torch.bmm(alignment2.transpose(1,2), decoded)

        #attention = self.conv1x1(attention.transpose(1,2)).transpose(1,2)
        attention1 = attention1.squeeze(1)
        attention2 = attention2.squeeze(1)
        #print("Shape of attention1 and attention2: ", attention1.shape, attention2.shape)        
        attention = torch.tanh(attention1) * torch.sigmoid(attention2)

        return self.linear2logits(attention)



class LIDmfcc(nn.Module):

    def __init__(self, in_dim=39):
        super(LIDmfcc, self).__init__()
        self.encoder = Encoder_TacotronOne(in_dim)
        self.attention = MaskedAttention(256)
        self.attention2linear = nn.Linear(256, 32)
        self.linear2logits = nn.Linear(32, 2)
        self.lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True) 

    def forward(self, mel, lengths=None):
        mel = self.encoder(mel.float(), lengths)
        self.lstm.flatten_parameters()
        mel, _ = self.lstm(mel)
        attention = self.attention(mel, lengths)
        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)


class LIDmfcclatents(nn.Module):

    def __init__(self, in_dim=39, vocab_size=257):
        super(LIDmfcclatents, self).__init__()
        self.mfcc_encoder = Encoder_TacotronOne(in_dim)
        self.latent_encoder = Encoder_TacotronOne(256)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.mfcc_attention = MaskedAttention(256)
        self.latent_attention = MaskedAttention(256)
        self.attention2linear = nn.Linear(256, 32)
        self.linear2logits = nn.Linear(32, 2)
        self.mfcc_lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True) 
        self.latent_lstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True) 
        self.drop = nn.Dropout(0.2)

    def forward(self, mel, latents, mfcc_lengths=None, latent_lengths=None):

        mel = self.drop(self.mfcc_encoder(mel.float(), mfcc_lengths))
        latents = self.embedding(latents)
        #print("Shape of latents: ", latents.shape)
        latents = self.drop(self.latent_encoder(latents))

        self.mfcc_lstm.flatten_parameters()
        self.latent_lstm.flatten_parameters()
        mel, _ = self.mfcc_lstm(mel)
        latents,_ = self.latent_lstm(latents)
        #print("Shape of latent_lengths and mfcc lengths: ", latent_lengths.shape, mfcc_lengths.shape)
        #mfcc_attention = self.mfcc_attention(mel, mfcc_lengths)
        #latent_attention = self.latent_attention(latents, latent_lengths)
        mfcc_attention = mel[:,-1,:]
        latent_attention = latents[:,-1,:]

        #attention = mfcc_attention + latent_attention
        attention = torch.tanh(mfcc_attention) * torch.sigmoid(latent_attention)
        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)



class LIDmfccmelmol(nn.Module):

    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmol, self).__init__()


        self.encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1)
            ]
        self.mfcc_preencoder = DownsamplingEncoder(mfcc_dim, self.encoder_layers)

        self.mfcc_encoder = Encoder_TacotronOne(mfcc_dim)
        self.mfcc_attention = Attention(256)

        self.attention2linear = nn.Linear(256, 32)
        self.linear2logits = nn.Linear(32, 2)

        self.drop = nn.Dropout(0.2)

    def forward(self, mfcc, mel, mol, mfcc_lengths=None):

        mfcc = self.mfcc_preencoder(mfcc)

        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)

        attention = mfcc_attention

        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)

class LIDmfccmelmol1(LIDmfccmelmol):

    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmol1, self).__init__(mfcc_dim, mel_dim)

        self.mel_preencoder = DownsamplingEncoder(mel_dim, self.encoder_layers)
        self.mel_encoder = Encoder_TacotronOne(mel_dim)
        self.mel_attention = Attention(256)
        self.conv1x1 = nn.Conv1d(256, 256, 1)
        self.conv1x1.bias.data.zero_()
        nn.init.orthogonal_(self.conv1x1.weight)
        self.bn = nn.BatchNorm1d(256, momentum=0.9)

    def forward(self, mfcc, mel, mol, mfcc_lengths=None):

        # MFCC
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)

        # Mel
        mel = self.mel_encoder(mel)
        mel_attention = self.mel_attention(mel)

        attention = mfcc_attention + mel_attention
        attention = attention.unsqueeze(1)
        attention = self.conv1x1(attention.transpose(1,2))
        attention = self.bn(attention).transpose(1,2)
        attention = attention.squeeze(1)

        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)


# MFCC + latents from mel + combination
class LIDmfccmelmol2(LIDmfccmelmol1):
        
    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmol2, self).__init__(mfcc_dim, mel_dim)

        self.quantizer = quantizer_kotha(n_channels=1, n_classes=200, vec_len=256)
        self.an = ActNorm1d(256, momentum=0.9)



    def forward(self, mfcc, mel, mol, mfcc_lengths=None):

        # MFCC
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)

        # Latents
        mel = self.mel_encoder(mel)
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mel.unsqueeze(2))
        quantized = quantized.squeeze(2)
        quantized_attention = self.mfcc_attention(quantized)

        # Pooling
        attention = mfcc_attention + quantized_attention
        attention = attention.unsqueeze(1)
        attention = self.conv1x1(attention.transpose(1,2))
        attention = self.bn(attention).transpose(1,2)
        attention = attention.squeeze(1)

        # Logits
        logits = torch.tanh(self.attention2linear(attention))
        logits = self.linear2logits(logits)

        return logits, vq_penalty.mean(), encoder_penalty.mean(), entropy


# MFCC + latents from mfcc + combination
class LIDmfccmelmol2C(LIDmfccmelmol2):

    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmol2C, self).__init__(mfcc_dim, mel_dim)
        
        self.quantized_attention = Attention(256)
        
    
    def forward(self, mfcc, mel, mol, mfcc_lengths=None):
       
        # MFCC
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)

        # Latents
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mfcc.unsqueeze(2))
        quantized = quantized.squeeze(2)
        quantized_attention = self.quantized_attention(quantized)

        # Pooling
        attention = mfcc_attention + quantized_attention
        attention = attention.unsqueeze(1)
        attention = self.conv1x1(attention.transpose(1,2))
        attention = self.bn(attention).transpose(1,2)
        attention = attention.squeeze(1)

        # Logits
        logits = torch.tanh(self.attention2linear(attention))
        logits = self.linear2logits(logits)

        return logits, vq_penalty.mean(), encoder_penalty.mean(), entropy


class LIDmfccmelmol3(LIDmfccmelmol2):

    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmol3, self).__init__(mfcc_dim, mel_dim)

        self.upsample_scales = [2,4,5,5]
        self.upsample_network = UpsampleNetwork(self.upsample_scales)
    
        self.logits_dim = 60
        self.joint_encoder = nn.LSTM(64, 256, batch_first=True)
        self.hidden2linear =  SequenceWise(nn.Linear(256, 64))
        self.mels2conv = nn.Linear(81,81)
        self.conv_1x1 = nn.Conv1d(81, 81, 1)
        self.conv_1x1.bias.data.zero_()
        self.linear2logits_reconstruction =  SequenceWise(nn.Linear(64, self.logits_dim))

        self.mels2encoderinput = nn.Linear(257,64)


    def forward(self, mfcc, mel, mol, mfcc_lengths=None):
       
        # MFCC
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)
        
        # Latents
        mel = self.mel_encoder(mel)
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mel.unsqueeze(2))
        quantized = quantized.squeeze(2)
        quantized_attention = self.mfcc_attention(quantized)

        # Pooling
        attention = mfcc_attention + quantized_attention
        attention = attention.unsqueeze(1)
        attention = self.conv1x1(attention.transpose(1,2))
        attention = self.bn(attention).transpose(1,2)
        attention = attention.squeeze(1)

        # Logits
        logits = torch.tanh(self.attention2linear(attention))
        logits = self.linear2logits(logits)

        # Reconstruction
        mel = self.upsample_network(mel)
        
        # Adjust inputs
        mel = mel[:,:-1,:]
        inp = mol[:, :-1].unsqueeze(-1)

        # Prepare inputs for LSTM
        melsNx = torch.cat([mel, inp], dim=-1)
        melsNx = torch.tanh(self.mels2encoderinput(melsNx))
        
        # Feed to encoder
        self.joint_encoder.flatten_parameters()
        outputs, hidden = self.joint_encoder(melsNx)

        # get logits
        mol_logits = torch.tanh(self.hidden2linear(outputs))
        mol_logits = self.linear2logits_reconstruction(mol_logits)
 
        return logits, vq_penalty.mean(), encoder_penalty.mean(), entropy, mol_logits, inp

    def forward_eval(self, mfcc, mel, mol, mfcc_lengths=None):

        # MFCC
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = self.drop(self.mfcc_encoder(mfcc))
        mfcc_attention = self.mfcc_attention(mfcc)
        
        # Latents
        mel = self.mel_encoder(mel)
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mel.unsqueeze(2))
        quantized = quantized.squeeze(2)
        quantized_attention = self.mfcc_attention(quantized)

        # Pooling
        attention = mfcc_attention + quantized_attention
        attention = attention.unsqueeze(1)
        attention = self.conv1x1(attention.transpose(1,2))
        attention = self.bn(attention).transpose(1,2)
        attention = attention.squeeze(1)

        # Logits
        logits = torch.tanh(self.attention2linear(attention))
        logits = self.linear2logits(logits)

        return logits, vq_penalty.mean(), encoder_penalty.mean(), entropy



class LIDmfccmelmoltransformer(nn.Module):
       
    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmoltransformer, self).__init__()

        self.encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.mfcc_preencoder = DownsamplingEncoder(mfcc_dim, self.encoder_layers)

        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=12)
        self.mfcc_fc = nn.Linear(mfcc_dim, 512)
        self.attention = Attention(512)
        self.attention2linear = nn.Linear(512, 32)
        self.linear2logits = nn.Linear(32, 2)

    def forward(self, mfcc, mel, mol, mfcc_lengths=None):
 
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = torch.tanh(self.mfcc_fc(mfcc))
        #print("Shape of mfcc before transformer: ", mfcc.shape)
        mfcc = self.transformer.encoder(mfcc)

        #mfcc_attention = self.attention(mfcc)
        attention = mfcc[:, 0,:]

        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits)


class LIDmfccmelmoltransformer2(LIDmfccmelmoltransformer):

    def __init__(self, mfcc_dim=39, mel_dim=80):
        super(LIDmfccmelmoltransformer2, self).__init__()

        self.quantizer = quantizer_kotha(n_channels=1, n_classes=200, vec_len=512)
        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=4)

    def forward(self, mfcc, mel, mol, mfcc_lengths=None):

        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = torch.tanh(self.mfcc_fc(mfcc))
        quantized, vq_penalty, encoder_penalty, entropy = self.quantizer(mfcc.unsqueeze(2))
        quantized = quantized.squeeze(2)
        mfcc = self.transformer.encoder(mfcc)
       
        #mfcc_attention = self.attention(mfcc)
        attention = mfcc[:, 0,:]
        
        logits = torch.tanh(self.attention2linear(attention))
        return self.linear2logits(logits),  vq_penalty.mean(), encoder_penalty.mean(), entropy
        

class LIDlatentstransformer(nn.Module):

    def __init__(self, n_vocab):
        super(LIDlatentstransformer, self).__init__()

        self.embedding = nn.Embedding(n_vocab, 128)
        self.transformer = nn.Transformer(nhead=4, num_encoder_layers=4)
        self.attention = MaskedAttention(128)
        self.attention2linear = nn.Linear(128, 32)
        self.linear2logits = nn.Linear(32, 2)
        self.lstm = nn.LSTM(128,64, bidirectional=True, batch_first=True)

        self.convmodules = nn.ModuleList() # nn.ModuleList 
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.selu = nn.SELU() 
        for i in range(30):
          self.conv = BatchNormConv1d(128, 128, kernel_size=1, stride=1, padding=0, activation=self.selu)
          #self.conv = nn.Conv1d(128,128,1)
          self.convmodules.append(self.conv) 
        self.conv_1x1 = nn.Conv1d(128,128,1)
        self.conv_1x1.bias.data.zero_()
        nn.init.orthogonal_(self.conv_1x1.weight)


        self.drop = nn.Dropout(0.2)

    def forward(self, latents, lengths=None):

        latents = self.drop(self.embedding(latents))
        #latents = self.transformer.encoder(latents)
        self.lstm.flatten_parameters()
        latents,_ = self.lstm(latents)

        latents = latents.transpose(1,2)
        for module in self.convmodules:
            x = latents
            latents = module(latents)
            latents = x + latents
        latents = self.conv_1x1(latents)
        latents = latents.transpose(1,2)

        latent_attention = self.attention(latents, lengths)
        attention = latent_attention

        logits = torch.tanh(self.attention2linear(attention))
        #print("Shape of logits: ", logits.shape)
        return self.linear2logits(logits)



class LIDlatentstransformerhuggingface(nn.Module):
        
    def __init__(self, n_vocab):
        super(LIDlatentstransformerhuggingface, self).__init__()

        self.embedding = nn.Embedding(n_vocab, 768)
        self.encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.preencoder = DownsamplingEncoder(768, self.encoder_layers)

        self.transformer = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
         
    def forward(self, latents, lengths=None):

        latents = self.embedding(latents)
        latents = self.preencoder(latents)
        outputs = self.transformer(inputs_embeds=latents) 
        #print("Shape of outputs: ", outputs[0].shape, len(outputs))
        return outputs[0]


class LIDMixtureofExpertslatents(nn.Module):

    def __init__(self, n_vocab):
        super(LIDMixtureofExpertslatents, self).__init__()
 
        # Embedding 
        self.embedding = nn.Embedding(201, 256)

        # Shared encoder
        self.encoder = Encoder_TacotronOne(256)

        # Expert 01
        self.exp1_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp1_fc_b = SequenceWise(nn.Linear(128, 256))

        # Expert 02
        self.exp2_fc_a = SequenceWise(nn.Linear(256, 128))
        self.exp2_fc_b = SequenceWise(nn.Linear(128, 256))

        self.mel2output = nn.Linear(256, 2)


    def forward(self, latents, lengths=None):

        # Embedding
        latents = self.embedding(latents) 

        # Pass through shared layers
        mel = self.encoder(latents, lengths)

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


class LIDMixtureofExpertslatentsattention(LIDMixtureofExpertslatents):

    def __init__(self, n_vocab=256):
        super(LIDMixtureofExpertslatentsattention, self).__init__(n_vocab)
        self.attention = MaskedAttention(256)

    def forward(self, latents, lengths=None):

        # Embedding
        latents = self.embedding(latents) 

        # Pass through shared layers
        mel = self.encoder(latents, lengths)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)
        
        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)
       
        # Combine the experts
        combination = torch.tanh(exp1_logits) * torch.sigmoid(exp2_logits)

        # Pooling
        attention = self.attention(combination, lengths)

        val_prediction = self.mel2output(attention)

        return val_prediction


class LIDMixtureofExpertsmfccattention(LIDMixtureofExpertslatents):

    def __init__(self, n_vocab=256):
        super(LIDMixtureofExpertsmfccattention, self).__init__(n_vocab)
        self.attention = MaskedAttention(256)
        self.mel2encoder = nn.Linear(39, 256)
        self.encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1)
            ]
        self.mfcc_preencoder = DownsamplingEncoder(39, self.encoder_layers)

    def forward(self, mfcc, lengths=None):

        # Pass through shared layers
        mfcc = self.mfcc_preencoder(mfcc)
        mfcc = torch.tanh(self.mel2encoder(mfcc)) 
        mel = self.encoder(mfcc)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)

        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)

        # Combine the experts
        combination = torch.tanh(exp1_logits) * torch.sigmoid(exp2_logits)
        
        # Pooling
        attention = self.attention(combination, lengths)
    
        val_prediction = self.mel2output(attention)

        return val_prediction

class LIDMixtureofExpertslatentsattentionconv(LIDMixtureofExpertslatentsattention):

    def __init__(self, n_vocab=256):
        super(LIDMixtureofExpertslatentsattentionconv, self).__init__(n_vocab)

        self.conv_1x1 = nn.Conv1d(256, 256, 1)
        self.conv_1x1.bias.data.zero_()
        nn.init.orthogonal_(self.conv_1x1.weight)
   
        self.bn = nn.BatchNorm1d(256, momentum=0.9) 
        self.encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1)
            ]
        self.latent_preencoder = DownsamplingEncoder(256, self.encoder_layers)
        self.drop = nn.Dropout(0.1)
   
    def forward(self, latents, lengths=None):

        # Embedding
        latents = self.embedding(latents)
        latents = self.latent_preencoder(latents)
        latents = self.drop(latents)

        # Pass through shared layers
        mel = self.encoder(latents)

        # Pass through expert 01
        exp1_logits = torch.tanh(self.exp1_fc_a(mel))
        exp1_logits = self.exp1_fc_b(exp1_logits)

        # Pass through expert 02
        exp2_logits = torch.tanh(self.exp2_fc_a(mel))
        exp2_logits = self.exp2_fc_b(exp2_logits)
        
        # Combine the experts
        combination = torch.tanh(exp1_logits) * torch.sigmoid(exp2_logits)
        #combination = F.tanh(self.conv_1x1(combination.transpose(1,2)))
        #combination = combination.transpose(1,2)

        # Pooling
        attention = self.attention(combination, lengths)
    
        val_prediction = self.mel2output(attention)

        return val_prediction

