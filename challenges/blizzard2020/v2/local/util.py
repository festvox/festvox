import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment


import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


### Text Processing Stuff
def populate_phonesarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary") 
       sys.exit()
    #print("Looking for fname ", fname)
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats

def populate_quantsarray(fname, feats_dir):

    arr = {}
    arr['fname'] = fname
    quant = np.load(fname)
    quant = quant.astype(np.int64) + 2**15

    assert len(quant) > 1
    #print("Shape of quant: ", quant.shape)
    coarse = quant // 256
    coarse_float = coarse.astype(np.float) / 127.5 - 1.
  
    fine = quant % 256
    fine_float = fine.astype(float) / 127.5 - 1.
 
    arr['coarse'] = coarse
    arr['coarse_float'] = coarse_float
    arr['fine'] = fine
    arr['fine_float'] = fine_float

    return arr


### Data Source Stuff
class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):
        #print("Feat type is ", self.feat_type, " and the feat name is ", self.feat_name)
        assert self.feat_type == 'categorical'
        fname =  str(self.filenames_array[idx])
        #fname = ''.join(k for k in fname[2:])
        fname = fname.lstrip("0")
        fname = self.feats_dir + '/' + fname.strip().zfill(8) + '.feats'
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'durations':
            f = open(fname)
            for line in f:
               line = line.split('\n')[0].split()
               return line
        elif self.feat_name == 'tones':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'phonesnossil':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'quants':
            fname += '.npy'
            return populate_quantsarray(fname, self.feats_dir)
        elif self.feat_name == 'r9y9inputmol':
            fname += '.npy'
            return np.load(fname)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)


    def __getitem__(self, idx):

        if self.feat_name == 'mspeckotha':
            fname =  str(self.filenames_array[idx]).zfill(8)
            fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
            #print("Shape I am loading is ", np.load(fname).T.shape)
            mspec = np.load(fname).T
            #print("Shape I loaded is ", mspec.shape)
            return mspec
        elif self.feat_name == 'mspec':
            fname =  str(self.filenames_array[idx]).zfill(8)
            fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
            return np.load(fname)

        elif self.feat_name == 'lspec':
            fname =  str(self.filenames_array[idx]).zfill(8)
            fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
            return np.load(fname)

        elif self.feat_name == 'r9y9outputmel':
            fname =  str(self.filenames_array[idx]).zfill(8)
            fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
            return np.load(fname)


#### Visualization Stuff

def visualize_phone_embeddings(model, checkpoints_dir, step):

    print("Computing TSNE")
    phone_embedding = model.embedding
    phone_embedding = list(phone_embedding.parameters())[0].cpu().detach().numpy()
    phone_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(phone_embedding)

    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phones_dict = json.load(f)

    ids2phones = {v:k for (k,v) in phones_dict.items()}
    phones = list(phones_dict.keys())
    y = phone_embedding[:,0]
    z = phone_embedding[:,1]

    fig, ax = plt.subplots()
    ax.scatter(y, z)

    for i, phone in enumerate(phones):
        ax.annotate(phone, (y[i], z[i]))

    path = checkpoints_dir + '/step' + str(step) + '_embedding_phones.png'
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

### Misc
def visualize_latent_embeddings(model, checkpoints_dir, step):
    return
    print("Computing TSNE")
    latent_embedding = model.quantizer.embedding0.squeeze(0).detach().cpu().numpy()
    num_classes = model.num_classes

    ppl_array = [5, 10, 40, 100, 200]
    for ppl in ppl_array:

       embedding = TSNE(n_components=2, verbose=1, perplexity=ppl).fit_transform(latent_embedding)

       y = embedding[:,0]
       z = embedding[:,1]

       fig, ax = plt.subplots()
       ax.scatter(y, z)

       for i  in range(num_classes):
          ax.annotate(i, (y[i], z[i]))

       path = checkpoints_dir + '/step' + str(step) + '_latent_embedding_perplexity_' + str(ppl) + '.png'
       plt.tight_layout()
       plt.savefig(path, format="png")
       plt.close()


class Blzrd2020Dataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)


class WaveLSTMDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)

def save_alignment(global_step, attn, checkpoint_dir=None):

    step = str(global_step).zfill(7)
    print("Save intermediate states at step {}".format(step))

    idx = 0

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(step))

    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment, step)



def collate_fn_mspeckotha(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return x_batch, input_lengths, mel_batch

def _pad(seq, max_len):
    #print("Shape of seq: ", seq.shape, " and the max length: ", max_len)     
    assert len(seq) < max_len
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn_r9y9melNmol(batch):
    """Create batch"""

    r = hparams.outputs_per_step
    seq_len = 4
    max_offsets = [x[1].shape[0] - seq_len for x in batch]
    mel_lengths = [x[1].shape[0] for x in batch]

    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [int(offset * hparams.frame_shift_ms * hparams.sample_rate / 1000) for offset in mel_offsets]
    sig_lengths = [x[0].shape[0] for x in batch]
    sig_length = int(seq_len * hparams.frame_shift_ms * hparams.sample_rate / 1000)

    mels = torch.FloatTensor([x[1][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])
    x = torch.FloatTensor([x[0][sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(batch)])

    return mels, x



def collate_fn_mspecNquant(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    seq_len = 4
    max_offsets = [x[1].shape[0] - seq_len for x in batch]
    mel_lengths = [x[1].shape[0] for x in batch]
    #print("Shortest utterance in this batch: ", min(mel_lengths))
    #print("Max offsets and lengths: ", ' '.join(str(k) + '_' + str(x) + '_' + str(x-k) for (k,x) in list(zip(max_offsets, mel_lengths))))
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [int(offset * hparams.frame_shift_ms * hparams.sample_rate / 1000) for offset in mel_offsets]
    sig_lengths = [x[0]['coarse'].shape[0] for x in batch]
    #print(sig_offsets, sig_lengths)
    sig_length = int(seq_len * hparams.frame_shift_ms * hparams.sample_rate / 1000)
    #print("Length of the raw samples I am taking in: ", sig_length)

    coarse = [x[0]['coarse'] for x in batch]
    fine = [x[0]['fine'] for x in batch]
    coarse_float = [x[0]['coarse_float'] for x in batch]  
    fine_float = [x[0]['fine_float'] for x in batch]

    mels = torch.FloatTensor([x[1][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])
    coarse = torch.LongTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(coarse)])
    fine = torch.LongTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(fine)])
    coarse_float = torch.FloatTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(coarse_float)])
    fine_float = torch.FloatTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(fine_float)])
    #print("Max value in coarse: ", torch.max(coarse.view(-1)))
    return mels, coarse, coarse_float, fine, fine_float

def collate_fn_tacotron(batch):

    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths) + 1

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, y_batch


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
# https://github.com/r9y9/wavenet_vocoder/blob/c4c148792c6263afbedb9f6bf11cd552668e26cb/train.py
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta



# https://github.com/r9y9/wavenet_vocoder/blob/c4c148792c6263afbedb9f6bf11cd552668e26cb/train.py#L365
class DiscretizedMixturelogisticLoss(nn.Module):
    def __init__(self):
        super(DiscretizedMixturelogisticLoss, self).__init__()

    def forward(self, input, target, lengths=None, mask=None, max_len=None):
        #if lengths is None and mask is None:
        #    raise RuntimeError("Should provide either lengths or mask")

        # (B, T, 1)
        #if mask is None:
        #    mask = sequence_mask(lengths, max_len).unsqueeze(-1)

        # (B, T, 1)
        #print("Shape of mask and target: ", mask.shape, target.shape)
        #mask_ = mask.expand_as(target)

        losses = discretized_mix_logistic_loss(
            input, target, num_classes=256,
            log_scale_min=-16.0, reduce=False)
        assert losses.size() == target.size()
        return losses.mean()

        #return ((losses * mask_).sum()) / mask_.sum()



def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

# https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
def discretized_mix_logistic_loss(y_hat, y, num_classes=256,
                                  log_scale_min=-7.0, reduce=True):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].
    Args:
        y_hat (Tensor): Predicted output (B x C x T)
        y (Tensor): Target (B x T x 1).
        num_classes (int): Number of classes
        log_scale_min (float): Log scale minimum value
        reduce (bool): If True, the losses are averaged or summed for each
          minibatch.
    Returns
        Tensor: loss
    """
    #print("Shapes of y and y_hat: ", y.shape, y_hat.shape)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.sum(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

# https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
def sample_from_discretized_mix_logistic(y, log_scale_min=-7.0,
                                         clamp_log_scale=False):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = to_one_hot(argmax, nr_mix)
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1)
    if clamp_log_scale:
        log_scales = torch.clamp(log_scales, min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x



# https://github.com/r9y9/wavenet_vocoder/blob/c4c148792c6263afbedb9f6bf11cd552668e26cb/train.py#L307
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class PhonesNTonesDataset(object):
    def __init__(self, X, T, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.T = T

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)



class PhonesNTonesDataset_dict(object):
    def __init__(self, X, T, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.T = T
        self.word2ids = self.X.feats_dict
        self.ids2words = {v:k for (k,v) in self.word2ids.items()}

        self.tone2ids = self.T.feats_dict
        self.ids2tones = {v:k for (k,v) in self.tone2ids.items()}

        with open('word2phones.json') as  f:
          word2phones = json.load(f)
        self.word2phones = dict(word2phones)

        self.replace_feats_dict()

        print("Tone2ids: ", self.tone2ids)

    def replace_feats_dict(self):
        phone2ids = defaultdict(lambda: len(phone2ids))
        phone2ids['<']
        phone2ids['>']
        for (k,v) in self.word2phones.items():
            phones = v.split()
            for p in phones:
                phone2ids[p]
        self.phone2ids = dict(phone2ids)

    def return_phoneids(self):
        return self.phone2ids


    def get_phonesNtones4mwords(self, pinyin_words, pinyin_tones):
        phseq = []
        toneseq = []
        for (w,t) in list(zip(pinyin_words, pinyin_tones)):
          #print("Word and tone are: ", w, t)
          if w == '<':
            phseq += [self.phone2ids[w]]
            toneseq += [self.tone2ids[str(t)]]
          elif w == '>':
            phseq += [self.phone2ids[w]]
            toneseq += [self.tone2ids[str(t)]]
          else:
            phones = self.word2phones[w]
            #print("Word and phones: ", w, phones)
            phones = phones.replace('pau',' ').split()
            for p in phones:
                phseq += [self.phone2ids[p]]
                toneseq += [self.tone2ids[str(t)]]
        assert len(phseq) == len(toneseq)
        return phseq, toneseq

    def __getitem__(self, idx):
        pinyin_sentence = self.X[idx]
        pinyin_words = [self.ids2words[k] for k in pinyin_sentence]
        pinyin_tones = self.T[idx]
        assert len(pinyin_words) == len(pinyin_tones)
        phones, tones = self.get_phonesNtones4mwords(pinyin_words, pinyin_tones)
        return phones, tones, self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

def collate_fn_phonesNtones(batch):

    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    
    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    a = np.array([_pad(x[1], max_input_len) for x in batch], dtype=np.int)
    tones_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)
    
    c = np.array([_pad_2d(x[3], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, tones_batch, input_lengths, mel_batch, y_batch


 
def collate_fn_phonesNdurations(batch):
    
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    target_lengths =  [len(x[1]) for x in batch]

    phones = [x[0] for x in batch]
    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch]) + 1
    #print("Max input length and max target length: ", max_input_len, max_target_len, input_lengths, target_lengths)
 
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    
    a = np.array([_pad(x[0], max_target_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)
 
    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)
    
    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, mel_batch, y_batch


class PhonesNDurationsDataset(object):
    def __init__(self, X, T, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.T = T

    def __getitem__(self, idx):
        phones = self.X[idx]
        durations = self.T[idx]
        assert len(phones) == len(durations) + 2

        # Extend durations
        durations_extended = []
        for i, d in enumerate(durations):
            d = int(d)
            while d > 0:
              durations_extended += [i+1]
              d = d -1
        phones = phones[durations_extended]
        # Extend phones to durations length
        return phones,  self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


class PhonesTonesNDurationsDataset(object):
    def __init__(self, X, Tones, T,  Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.T = T
        self.Tones = Tones
    
    def __getitem__(self, idx):
        phones = self.X[idx]
        durations = self.T[idx]
        tones =  self.Tones[idx]
        #print("Lengths of phones, tones and durations: ", len(phones), len(tones), len(durations))
        #print(phones, tones)
        assert len(phones) == len(durations) + 2
        assert len(phones) == len(tones)

        # Extend durations
        durations_extended = []
        for i, d in enumerate(durations):
            d = int(d)
            while d > 0:
              durations_extended += [i+1]
              d = d -1
        phones = phones[durations_extended]
        tones = tones[durations_extended]
        # Extend phones to durations length
        return phones, tones, self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


  
def collate_fn_phonestonesNdurations(batch): 
     
    r = hparams.outputs_per_step 
    input_lengths = [len(x[0]) for x in batch] 
    target_lengths =  [len(x[2]) for x in batch] 
 
    phones = [x[0] for x in batch]
    tones = [x[1] for x in batch]  

    max_input_len = np.max(input_lengths) + 1 
    # Add single zeros frame at least, so plus 1 
    max_target_len = np.max([len(x[2]) for x in batch]) + 1 
    #print("Max input length and max target length: ", max_input_len, max_target_len, input_lengths, target_lengths) 
  
    if max_target_len % r != 0: 
        max_target_len += r - max_target_len % r 
        assert max_target_len % r == 0 
     
    assert max_target_len - max_input_len <= 150

    a = np.array([_pad(x[0], max_target_len) for x in batch], dtype=np.int) 
    x_batch = torch.LongTensor(a) 
  
    a = np.array([_pad(x[1], max_target_len) for x in batch], dtype=np.int) 
    tones_batch = torch.LongTensor(a) 

    b = np.array([_pad_2d(x[2], max_target_len) for x in batch], 
                 dtype=np.float32) 
    mel_batch = torch.FloatTensor(b) 
     
    c = np.array([_pad_2d(x[3], max_target_len) for x in batch], 
                 dtype=np.float32) 
    y_batch = torch.FloatTensor(c) 
    return x_batch, tones_batch, mel_batch, y_batch 

