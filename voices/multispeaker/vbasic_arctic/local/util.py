import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment

import torch.nn as nn
import torch.nn.functional as F


### Text Processing Stuff
def populate_phonesarray(fname, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary") 
       sys.exit()

    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats


def get_fnames(fnames_file, scp_file):
    scp_dict = {}
    f = open(scp_file)
    for line in f:
      line = line.split('\n')[0]
      fname, location = line.split()[0], line.split()[1]
      scp_dict[fname] = location

    #print(scp_dict)
    filenames_array = []
    f = open(fnames_file)
    for line in f:
      line = line.split('\n')[0]
      #print(line, fnames_file, scp_file)
      filenames_array.append(scp_dict[line])
    return filenames_array


### Data Source Stuff
class categorical_datasource(Dataset):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dict=None, spk_dict=None):

      self.fnames_file = fnames_file
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.vox_dir = 'vox'
      self.filenames_array = get_fnames(self.fnames_file, self.vox_dir + '/etc/fnamesN' + self.feat_name) # Filenames is a placeholder really.
      print("Feat Name is ", feat_name) 
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)
      self.feats_dict = defaultdict(lambda: len(self.feats_dict)) if feats_dict is None else feats_dict
      self.spk_dict = defaultdict(lambda: len(self.spk_dict)) if spk_dict is None else spk_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dict)
        elif self.feat_name == 'speaker' :
            return int(fname)
        elif self.feat_name == 'r9y9inputmol' :
            return np.load(fname)

        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

    def __len__(self):
       return len(self.filenames_array)

    def get_spkdict(self):
        return self.spk_dict
 
class float_datasource(Dataset):

    def __init__(self, fnames_file, desc_file, feat_name):

      self.fnames_file = fnames_file
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.vox_dir = 'vox'
      self.filenames_array = get_fnames(self.fnames_file, self.vox_dir + '/etc/fnamesN' + self.feat_name)
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)

    def __getitem__(self, idx):

        fname = self.filenames_array[idx]
        return np.load(fname)

    def __len__(self):
       return len(self.filenames_array)


class MultispeakerDataset(object):
    def __init__(self, X, spk, Mel, Y):
        self.X = X
        self.spk = spk
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.spk[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


class MultispeakerVocoderDataset(object):
    def __init__(self, X, spk, Mel):
        self.X = X
        self.spk = spk
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.spk[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)


### Collate Stuff


def _pad(seq, max_len):
    #print("Shape of seq: ", seq.shape, " and the max length: ", max_len)     
    #assert len(seq) < max_len
    if len(seq) == max_len:
        return seq
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


# We get (x_array, spk_id), y, mel
def collate_fn_spk(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    phones = [x for (x,_,_,_) in batch]
    mels =  [x for (_, _, x,_) in batch]
    linears = [x for (_,_,_, x) in batch]
    spk_ids =  [x for (_,x,_,_) in batch]

    input_lengths = [len(p) for p in phones]
    mel_lengths = [len(mel) for mel in mels]
    if np.all(mel_lengths) is False:
       print("Check this ", mel_lengths)
       sys.exit()

    max_input_len = np.max(input_lengths)

    max_target_len = np.max([len(x) for x in mels]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    try:
      a = np.array([_pad(p, max_input_len) for p in phones], dtype=np.int)
    except Exception as e:
      print("Exception here : ", chars)
      print(e)
      sys.exit()
    x_batch = torch.LongTensor(a)
    #print(spk_ids)
    spk_batch = torch.LongTensor(spk_ids)

    input_lengths = torch.LongTensor(input_lengths)
    b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(linear, max_target_len) for linear in linears],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    return x_batch, spk_batch, input_lengths, mel_batch, y_batch



def visualize_latent_embeddings(model, checkpoints_dir, step):

    print("Computing TSNE")
    #latent_embedding = model.quantizer.
    spk_embedding = list(spk_embedding.parameters())[0].cpu().detach().numpy()
    spk_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(spk_embedding)

    with open(checkpoints_dir + '/spk_ids') as  f:
       speakers_dict = json.load(f)

    ids2speakers = {v:k for (k,v) in speakers_dict.items()}
    speakers = list(speakers_dict.keys())
    y = spk_embedding[:,0]
    z = spk_embedding[:,1]

    fig, ax = plt.subplots()
    ax.scatter(y, z)

    for i, spk in enumerate(speakers):
        ax.annotate(spk, (y[i], z[i]))

    path = checkpoints_dir + '/step' + str(step) + '_speaker_embedding.png'
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def collate_fn_r9y9melNmolNspk(batch):
    """Create batch"""

    r = hparams.outputs_per_step
    seq_len = 4
    max_offsets = [x[2].shape[0] - seq_len for x in batch]
    mel_lengths = [x[2].shape[0] for x in batch]

    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [int(offset * hparams.frame_shift_ms * hparams.sample_rate / 1000) for offset in mel_offsets]
    sig_lengths = [x[0].shape[0] for x in batch]
    sig_length = int(seq_len * hparams.frame_shift_ms * hparams.sample_rate / 1000)

    spk_ids =  [x for (_,x,_) in batch]

    mels = torch.FloatTensor([x[2][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])
    x = torch.FloatTensor([x[0][sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(batch)])
    spk_batch = torch.LongTensor(spk_ids)

    return mels, x, spk_batch




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
