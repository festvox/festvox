""" Populate tdd file from phones and stress

Usage: local/populate_tddfromphonesNstress.py [options] <phonesNstress_dir> <tdd_file>

options:
    -h, --help               Show help message.

"""
from docopt import docopt
import os, sys
import numpy as np

args = docopt(__doc__)

import os

phonesNstress_dir = args['<phonesNstress_dir>']
files = sorted(os.listdir(phonesNstress_dir))

tdd_file = args['<tdd_file>']
g = open(tdd_file, 'w')

for file in files:
   if file.endswith('.stress'):
       fname = file.split('.')[0]
       l = '<_0.0'
       f = open(phonesNstress_dir + '/' + file)
       for line in f:
           line = line.split('\n')[0]
           try:
             phone, stress = line.split()[0], line.split()[1]
           except IndexError:
             print("Check this file for errors", fname)
           l += ' ' + phone + '_' + stress
       l += ' >_0.0'
       g.write(fname + ' ' + l + '\n')
       f.close()
g.close()
