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
        arr = {}
        arr['fname'] = fname
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        f = open(fname)
        for line in f:
           mask = line.split('\n')[0]
           arr['mask'] = mask
           return arr

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)


class MaskDataset(object):

    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)


class mask_dataset_triplet(MaskDataset):

    def __init__(self, X, Mel, train_mask_file='vox/fnames.train.mask', train_clear_file='vox/fnames.train.clear'):
       super(mask_dataset_triplet, self).__init__(X, Mel)

       self.train_mask = self.return_fnamesarray(train_mask_file)
       self.train_clear = self.return_fnamesarray(train_clear_file)

    def __getitem__(self, idx):

        label = self.X[idx]

        if int(label['mask']) == 0:

           #print("Label is 0")
           ridx = np.random.randint(len(self.train_mask))
           positive_fname = self.train_mask[ridx]
           fname = 'vox/festival/falcon_mfcc/' + positive_fname + '.feats.npy'
           positive_mel = np.load(fname)

           ridx = np.random.randint(len(self.train_clear))
           negative_fname = self.train_clear[ridx]
           fname = 'vox/festival/falcon_mfcc/' + negative_fname + '.feats.npy'
           negative_mel = np.load(fname)
           #print("Got my pos and neg from label 0", label)

        elif int(label['mask']) == 1:

           #print("Label is 1")
           ridx = np.random.randint(len(self.train_clear))
           positive_fname = self.train_clear[ridx]
           fname = 'vox/festival/falcon_mfcc/' + positive_fname + '.feats.npy'
           positive_mel = np.load(fname)

           ridx = np.random.randint(len(self.train_mask))
           negative_fname = self.train_mask[ridx]
           fname = 'vox/festival/falcon_mfcc/' + negative_fname + '.feats.npy'
           negative_mel = np.load(fname)
           #print("Got my pos and neg from label 1", label)

        else:

          print("I dont know this one ", label)
          sys.exit()

        return self.X[idx], self.Mel[idx], positive_mel, negative_mel


    def return_fnamesarray(self, fname):
       f = open(fname)
       arr = []
       for line in f:
           line = line.split('.')[0]
           arr.append(line)
       return arr



def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

def collate_fn_mask(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch])

    a = np.array([x[0]['mask'] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    f = [x[0]['fname'] for x in batch]


    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return x_batch, mel_batch, f



def collate_fn_mask_triplet(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    max_target_len = np.max([len(x[1]) for x in batch])

    a = np.array([x[0]['mask'] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    f = [x[0]['fname'] for x in batch]


    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    max_target_len = np.max([len(x[2]) for x in batch])
    positive_mel = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    positive_mel = torch.FloatTensor(positive_mel)


    max_target_len = np.max([len(x[3]) for x in batch])
    negative_mel = np.array([_pad_2d(x[3], max_target_len) for x in batch],
                 dtype=np.float32)
    negative_mel = torch.FloatTensor(negative_mel)

    return x_batch, mel_batch, positive_mel, negative_mel, f



# Utility to return predictions
def return_classes(logits, dim=-1):
   _, predicted = torch.max(logits,dim)
   return predicted.view(-1).cpu().numpy()


def get_metrics(predicteds, targets):
   print(classification_report(targets, predicteds))
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   print("EER is ", EER)
   return recall_score(predicteds, targets,average='macro')

