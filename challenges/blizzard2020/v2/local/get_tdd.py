import os
from collections import defaultdict
import json

input_file = '../output.g2pc'
underscores_file = 'tdd.underscores'
tdd_file = 'tdd'

words_dict = defaultdict(lambda: len(words_dict))


def remove_tones(sentence):
  words = sentence.split()
  content = ''
  underscored_content = ''
  for word in words:
    if word == '.':
       word = 'pau'
    last_char = word[-1]
    #print(last_char)
    if last_char.isdigit():
       c_word = ''.join(k for k in word[:-1])
       u_word = c_word + '_' + last_char
    else:
       c_word = word
       u_word = c_word + '_0'
    words_dict[c_word] 
    content += ' ' + c_word
    underscored_content += ' ' + u_word

  return underscored_content, content


f = open(input_file)
g = open(tdd_file, 'w')
h = open(underscores_file, 'w')
for line in f:
   line = line.split('\n')[0]
   fname = line.split()[0]
   content = ' '.join(k for k in line.split()[1:])
   underscored_content, content = remove_tones(content)
   print(fname, content, underscored_content)
   g.write('( ' + fname + ' " ' + content + ' " )' + '\n')
   h.write(fname + ' ' + underscored_content + '\n')
f.close() 
g.close()
h.close()


idsdict_file = 'wordids.json'
with open(idsdict_file, 'w') as outfile:
  json.dump(words_dict, outfile)

print(len(words_dict))
