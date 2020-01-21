""" Quantize F0s

Usage: quantize_f0.py [options] <F0_dir> <qF0_dir>

options:
    --bin-size=<N>             Size of F0 bin [default: 50].
    -h, --help               Show help message.

"""
from docopt import docopt
import os, sys
import numpy as np

args = docopt(__doc__)
src_dir = args['<F0_dir>']
tgt_dir = args['<qF0_dir>']
binsize = int(args['--bin-size'])

if not os.path.exists(tgt_dir):
   os.mkdir(tgt_dir)

files = sorted(os.listdir(src_dir))
l = len(files)
for i, file in enumerate(files):
   A = np.loadtxt(src_dir + '/' + file, usecols=0)
   A += 10
   A //= binsize
   np.savetxt(tgt_dir + '/' + file, A)

   if i % 100 == 0:
      print("Processed ", i, "files out of ", l)
