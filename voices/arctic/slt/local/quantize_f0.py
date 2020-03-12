""" Quantize F0s

Usage: local/quantize_f0.py [options] <f0_ascii_dir> <qF0s_dir>

options:
    --factor=<N>             factor to bin F0 [default: 50].
    -h, --help               Show help message.

"""
from docopt import docopt
import os, sys
import numpy as np


src_dir = sys.argv[1]
tgt_dir = sys.argv[2]

args = docopt(__doc__)
src_dir = args['<f0_ascii_dir>']
tgt_dir = args['<qF0s_dir>']
factor = int(args['--factor'])

if not os.path.exists(tgt_dir):
   os.mkdir(tgt_dir)


files = sorted(os.listdir(src_dir))
l = len(files)
for i, file in enumerate(files):
   A = np.loadtxt(src_dir + '/' + file, usecols=0)
   A += 10
   A //= factor
   np.savetxt(tgt_dir + '/' + file, A)

   if i % 100 == 0:
      print("Processed ", i, "files out of ", l)

