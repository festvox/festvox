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
quotes_file = research_dir + '/quotes.all'
print("The group is ", group_id)
query = sys.argv[1]

from heapq import nlargest

max_papers = 500

# https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
def sample_from_iterable(it, k):
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


# Get an interator over query results
result = arxiv.query(
  query=query,
  max_chunk_results=10,
  max_results=max_papers,
  iterative=True
)

#papers = sample_from_iterable(result, max_papers)
papers = []
for paper in result():
   papers += [paper]

paper = random.choice(papers)


#print(paper)
print(paper['title'])
print(paper['summary'])
print
#for l in paper:
#   print(l)
   #print('\n')


token = os.environ.get('FACEBOOK_TOKEN_JUDITH')
group_id = str(os.environ.get('DeEntanglement_GROUP_ID'))
research_dir = os.environ.get('research_dir')
quotes_file = research_dir + '/quotes.all'
print("The group is ", group_id)


def get_content(papers):

    paper = random.choice(papers)
    title = paper['title']
    summary = paper['summary']
    summary = summary.replace('\n',' ')
    url = paper['arxiv_url']
    content = title + '(' + url + ')' +  '\n\n' + summary
    return content


def main():
    graph = facebook.GraphAPI(token)
    group = graph.get_object(id=group_id)
    id = group['id']

    for i in range(10):
       content = get_content(papers)
       graph.put_object(id, 'feed', message=content)
       sleep(3600)

    print(group)


main()
