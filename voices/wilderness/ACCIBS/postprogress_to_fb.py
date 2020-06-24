#!usr/bin/env python
# Program to mine data from your own facebook account
            
import json
import facebook
import os   
import sys  
import random

token = os.environ.get('FACEBOOK_TOKEN_JUDITH')
group_id = str(os.environ.get('DeEntanglement_GROUP_ID'))
log_file = 'exp/exp_baseline_ACCIBS/tracking/logfile'
alignments_path = 'exp/exp_baseline_ACCIBS/checkpoints/'
expt_name = 'Wilderness'
print("The group is ", group_id)


def get_progress(log_file):
    f = open(log_file)
    for line in f:
        line = line.split('\n')[0].split()
        content = ' '.join(k for k in line)
    return content
 
def get_pic():
  files = sorted(os.listdir(alignments_path))
  attention_files = []
  for file in files:
    if file.endswith('_alignment.png'):
       attention_files.append(file)
  return attention_files[-1]
 
 
def main():
    graph = facebook.GraphAPI(token)
    # profile = graph.get_object(
    #    'me', fields='first_name,location,link,email,groups')
    group = graph.get_object(id=group_id)
    id = group['id']
 
    content = get_progress(log_file)
    pic = get_pic()
    pic = alignments_path + '/' + pic
    graph.put_photo(album_path=id + '/photos', image=open(pic, 'rb'), message=expt_name + ' : ' + content)
 
    print(group)
 
 
if __name__ == '__main__':
    main()
 

