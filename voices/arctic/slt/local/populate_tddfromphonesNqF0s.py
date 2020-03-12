""" Populate tdd file from phones and quantized F0s

Usage: local/populate_tddfromphonesNqF0s.py [options] <qF0sNphones_dir> <tdd_file>

options:
    -h, --help               Show help message.

"""
from docopt import docopt
import os, sys
import numpy as np

args = docopt(__doc__)

import os

tonesNphones_dir = args['<qF0sNphones_dir>']
files = sorted(os.listdir(tonesNphones_dir))

tdd_file = args['<tdd_file>']
g = open(tdd_file, 'w')

for file in files:
   if file.endswith('.qF0sNphones'):
       fname = file.split('.')[0]
       l = '<_0.0'
       f = open(tonesNphones_dir + '/' + file)
       for line in f:
           line = line.split('\n')[0]
           try:
             phone, qF0 = line.split()[0], line.split()[1]
           except IndexError:
             print("Check this file for errors", fname)
           l += ' ' + phone + '_' + qF0
       l += ' >_0.0'
       g.write(fname + ' ' + l + '\n')
       f.close()
g.close()

