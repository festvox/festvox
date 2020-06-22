import arxiv
import random
import os, sys
import facebook
import json
import facebook
import os
import sys
import random
from time import sleep

token = os.environ.get('FACEBOOK_TOKEN_JUDITH')
group_id = str(os.environ.get('DeEntanglement_GROUP_ID'))
research_dir = os.environ.get('research_dir')
papers_file = research_dir + '/papers_GANDisentanglement.csv'
print("The group is ", group_id)

f = open(papers_file)
title2summary ={}
for line in f:
   print(line)
   line = line.split('\n')[0]
   print("Line is ", line)
   title, summary = line.split(',')[1], line.split(',')[0]
   title2summary[title] = summary

def get_content():
    title = random.choice(list(title2summary.keys()))
    summary = title2summary[title]
    content = title + '\n' + summary
    return content


def main():
    graph = facebook.GraphAPI(token)
    group = graph.get_object(id=group_id)
    id = group['id']

    for i in range(30):
       content = get_content()
       graph.put_object(id, 'feed', message=content)
       sleep(120)

    print(group)


main()
