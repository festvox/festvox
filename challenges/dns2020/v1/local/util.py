import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment



def populate_quantsarray(fname, feats_dir):

    arr = {}
    arr['fname'] = fname
    quant = np.load(fname)
    quant = quant.astype(np.int64) + 2**15

    assert len(quant) > 1
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

        assert self.feat_type == 'categorical'
        fname =  str(self.filenames_array[idx]).strip()
        clean_fname = 'clean_fileid_' + fname.split('_fileid_')[1]

        clean_fname = self.feats_dir + '/' + clean_fname + '.feats'

        if self.feat_name == 'quant':
            clean_fname += '.npy'
            return populate_quantsarray(clean_fname, self.feats_dir)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)



class DNSDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)


def collate_fn_mspecNquant(batch):
    """Create batch"""

    r = hparams.outputs_per_step
    seq_len = 8
    max_offsets = [x[1].shape[0] - seq_len for x in batch]
    mel_lengths = [x[1].shape[0] for x in batch]
    #print(max_offsets, mel_lengths)

    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [int(offset * hparams.frame_shift_ms * hparams.sample_rate / 1000) for offset in mel_offsets]
    sig_lengths = [x[0]['coarse'].shape[0] for x in batch]
    #print(sig_lengths)

    sig_length = int(seq_len * hparams.frame_shift_ms * hparams.sample_rate / 1000)

    coarse_clean = [x[0]['coarse'] for x in batch]
    fine_clean = [x[0]['fine'] for x in batch]
    coarse_float_clean = [x[0]['coarse_float'] for x in batch]
    fine_float_clean = [x[0]['fine_float'] for x in batch]

    mels_noisy = torch.FloatTensor([x[1][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])
    coarse_clean = torch.LongTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(coarse_clean)])
    fine_clean = torch.LongTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(fine_clean)])
    coarse_float_clean = torch.FloatTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(coarse_float_clean)])
    fine_float_clean = torch.FloatTensor([x[sig_offsets[i]:int(sig_offsets[i] + sig_length)] for i, x in enumerate(fine_float_clean)])

    return mels_noisy,  coarse_clean, fine_clean, coarse_float_clean, fine_float_clean



