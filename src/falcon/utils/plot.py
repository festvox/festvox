import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .audio import *
from sklearn.manifold import TSNE
import json

def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')

def save_alignment(path, attn, global_step):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))


def save_spectrogram(path, linear_output):
    spectrogram = denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()



def visualize_speaker_embeddings(model, checkpoints_dir, step):

    print("Computing TSNE")
    spk_embedding = model.spk_embedding
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


