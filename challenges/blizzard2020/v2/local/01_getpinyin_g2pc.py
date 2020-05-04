import os,sys
#import pinyin
#import pinyin.cedict
from g2pc import G2pC

g2p = G2pC()

input_file = '../data/Hub/train/train_text.txt'
output_file = '../output.g2pc'

f = open(input_file, encoding='gb18030')
g = open(output_file, 'w')
g.close()
for line in f:
   line = line.split('\n')[0]
   content = ' '.join(k for k in line.split()[1:])
   fname = line.split()[0]
   data = g2p(content) # [(_, _, pin, _, _, _ )]
   pin = ' '.join(k[3] for k in data)
   pin = pin.replace('\uff0c', 'pau').replace('\u3002', '.').replace('\u3001', ',')
   g = open(output_file, 'a')
   g.write(fname + ' ' + pin + '\n')
   #print(fname, pin)
   g.close()
