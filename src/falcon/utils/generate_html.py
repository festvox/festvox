""" Generate a html page with sample wavefiles

Usage: generate_html.py [options] <fnames_file> <original_directory> <samples_directory> <output_file>
       generate_html.py vox/fnames.test vox/wav exp/taco_one_phseq/tts_phseq exp/taco_one_phseq/tts_phseq.html

Options:
    --max-files=<N>          Maximum files to plot [default: 3].
    --shuffle                Flag to shuffle the filenames [default: None].
    --link=<l>               Link to append to the path of audio file
    --exp-name=<exp>         Title of experiment [default: samples]
    --add-exp=<e>            Location of additional experiment to compare
    -h, --help               Show help message.

"""
from docopt import docopt

import os, sys
from xml.etree import ElementTree as ET
import random



def write_html(fnames, original_dir, samples_dir, exp_name, out_file, link=None, additional_exp=None):
   
   print("Writing html")

   html = ET.Element('html')

   body = ET.Element('body')
   html.append(body)

   ## Initial spacing
   div_space = ET.Element('div', attrib={'class': 'col-md-1'})
   body.append(div_space)

   ## Title

   div = ET.Element('div')
   desc = ET.Element('h3')
   desc.text = exp_name  
   div.append(desc)
   body.append(div)

   for fname in fnames:
       fname = fname.split('\n')[0].strip()
       div = ET.Element('div', attrib={'class': 'row'})
       print("Adding ", fname)
       desc = ET.Element('text')
       desc.text = fname
       div.append(desc)
       body.append(div)
       div = ET.Element('div', attrib={'class': 'row'})

       # Original
       audio = ET.Element('audio', attrib={'controls': 'controls'})
       path = os.path.basename(original_dir)
       if link:
          path = link + path
       source = ET.Element('source', src = path + '/' + fname + '.wav')
       audio.append(source)
       div.append(audio)

       # Sample
       audio = ET.Element('audio', attrib={'controls': 'controls'})
       path = os.path.basename(samples_dir)
       if link:
          path = link + path
       source = ET.Element('source', src = path + '/' + fname + '.wav')
       audio.append(source)
       div.append(audio)

       # Additional Experiment
       if additional_exp:
         audio = ET.Element('audio', attrib={'controls': 'controls'})
         path = os.path.basename(additional_exp)
         if link:
           path = link + path
         source = ET.Element('source', src = path + '/' + fname + '.wav')
         audio.append(source)
         div.append(audio)


       body.append(div)

   ET.ElementTree(html).write(out_file, encoding='unicode', method='html')





args = docopt(__doc__)
fnames_file = args['<fnames_file>']
original_dir = args['<original_directory>']
samples_dir = args['<samples_directory>']
max_files = int(args['--max-files'])
shuffle = args['--shuffle']
link = args['--link']
exp_name = args['--exp-name']
out_file = args['<output_file>']
additional_exp = args['--add-exp']

with open(fnames_file) as f:
    fnames = f.readlines()

print(shuffle, max_files)
if shuffle:
   random.shuffle(fnames)
fnames = fnames[0:max_files]

write_html(fnames, original_dir, samples_dir, exp_name, out_file, link, additional_exp)

