import arxiv

# Get an interator over query results
result = arxiv.query(
  query="GAN",
  max_chunk_results=10,
  max_results=1,
  iterative=True
)

for paper in result():
   #print(paper)
   pass

#print(paper)
print(paper['title'])
print(paper['summary'])
print(paper['arxiv_url'])
for l in paper:
   print(l)
   #print('\n')
