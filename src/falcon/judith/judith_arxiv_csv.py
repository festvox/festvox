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
query = sys.argv[1]
#filter_term = sys.argv[2]
query = ' '.join(k for k in query.split('_'))
querylog_file = research_dir + '/researchlogs_' + query + '.csv'
print("The group is ", group_id)

from heapq import nlargest

max_papers = 50000

# https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
def sample_from_iterable(it, k):
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


# Get an interator over query results
result = arxiv.query(
  query=query,
  max_chunk_results=100,
  max_results=max_papers,
  iterative=True
)

#papers = sample_from_iterable(result, max_papers)
papers = []
for paper in result():
  #if filter_term in paper['summary']:
  papers += [paper]
  #else:
  #   print("Ignoring ", paper['title'])  


def get_content(paper):

    title = paper['title']
    summary = paper['summary']
    summary = summary.replace('\n',' ').replace(',', ' ').rstrip('\r\n')
    title = title.replace('\n',' ').replace(',', ' ').rstrip('\r\n') 
    url = paper['arxiv_url']
    content = title + ',' + summary
    return content

g = open(querylog_file, 'w')
for paper in papers:
	content = get_content(paper)
	g.write(content + '\n')
g.close()	