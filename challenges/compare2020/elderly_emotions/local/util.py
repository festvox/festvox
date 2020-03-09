import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment
from sklearn.metrics import *


### Data Source Stuff
class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        f = open(fname)
        for line in f:
           valence = line.split('\n')[0]
           return valence

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)



class ValenceDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)

class ValenceNArousalDataset(object):
    def __init__(self, Xa, Xv, Mel):
        self.Xa = Xa
        self.Xv = Xv
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.Xa[idx], self.Xv[idx], self.Mel[idx]

    def __len__(self):
        return len(self.Xa)



def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

def collate_fn_valence(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch])

    a = np.array([x[0] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return x_batch, mel_batch

def collate_fn_valenceNarousal(batch):
    """Create batch"""
    r = hparams.outputs_per_step


    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch])

    a = np.array([x[0] for x in batch], dtype=np.int)
    xa_batch = torch.LongTensor(a)

    a = np.array([x[1] for x in batch], dtype=np.int)
    xv_batch = torch.LongTensor(a)


    b = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return xa_batch, xv_batch, mel_batch


# Utility to return predictions
def return_classes(logits, dim=-1):
   _, predicted = torch.max(logits,dim)    
   return predicted.view(-1).cpu().numpy()


def get_metrics(predicteds, targets):
   print(classification_report(targets, predicteds))
   print("Accuracy is ", accuracy_score(targets, predicteds))
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   print("EER is ", EER)
   return recall_score(predicteds, targets,average='macro')

