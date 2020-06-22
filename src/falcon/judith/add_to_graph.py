from py2neo import Graph, Node, Relationship  # , authenticate
import os
import sys
from add_py2neo import *

password = str(os.environ.get('NEO4J_PASSWORD'))
user = 'neo4j'
# authenticate("localhost:7474", user, password)
g = Graph("http://localhost:7474/db/data/", password=password)
# print(g)
# sys.exit()

research_dir = os.environ.get('research_dir')
queries = ['graph']


#g.delete_all()
#g = add_concepts(g)
tx = g.begin()

graph_stuff = g.nodes.match("Concepts", name="Graph Processing").first()
if graph_stuff is None:
   graph_stuff = Node("Concepts", name="Graph Processing")


list_of_terms = ['graph']

term2framework = {}
term2framework['graph'] = graph_stuff


def return_terms(summary):
  list_of_terms = ['variational', 'adversarial', 'GAN', 'entangle', 'information', 'bottleneck', 'representation', 'understanding', 'graph']
  terms = []
  for term in list_of_terms:
      if term in summary:
         terms += [term]
  return ','.join(k for k in terms)


def get_terms(summary, paper, list_of_terms=list_of_terms):

    relation = None
    previous_frameworks = []
    lines = summary.split('.')
    #terms = []
    for line in lines:
        #print(line)
        for term in list_of_terms:
            if term in line:
               #terms += [term]
               framework = term2framework[term]
               if framework in previous_frameworks:
               	  continue
               previous_frameworks += [framework]
               print("  Term: ", term)
               print("  Framework: ", framework)
               if relation is None:
               	  relation = Relationship(framework, "AS", paper)
               	  if framework2paradigm[framework] is not None:
                     paradigm = framework2paradigm[framework]
                     relation += Relationship(paradigm, "AS", paper)
                     tx.create(relation)	
               else:	  
                  relation = Relationship(framework, "AS", paper)
                  tx.create(relation)
                  paradigm = framework2paradigm[framework]
                  if paradigm is not None:
                     relation = Relationship(paradigm, "AS", paper)
                  tx.create(relation)
    #terms = list(set(terms))
    #print(terms)
    #paper.update(terms = ','.join(k for k in terms)) 
    #if relation is not None:
    #   tx.create(relation)
    #print("  Relation is ", relation)       
    return paper

def populate_query_stats(query):
    querylog_file = research_dir + '/researchlogs_' + query + '.csv'
 
    f = open(querylog_file)
    for line in f:
       line = line.split('\n')[0].split(',')
       try:
         title, summary = line[0], line[1]
         #print(title)
         terms = return_terms(summary)
         paper = Node("Papers", name=title, summary=summary, terms = terms)
         # paper2summary = Relationship(paper, "summary", summary)
         #tx.create(paper2summary)
         get_terms(summary, paper)
         tx.create(paper)
         print("Added to graph ", title)
       except IndexError:
         print("Failed on this ", line)
         pass  
    f.close()

for query in queries:
    populate_query_stats(query)    


tx.commit()

'''
# Dissertation
deentanglement = Dissertation('De-Entanglement')
g.push(deentanglement)

# Challenges
scalability = Challenge('Scalability')
flexibility = Challenge('Flexibility')
explainability = Challenge('Explainability')


g.push(deentanglement)
g.push(scalability)
g.push(flexibility)
g.push(explainability)


tx = g.begin()

sc = Relationship(deentanglement.node, "INCLUDES", scalability)
tx.create(sc)

fl = Relationship(deentanglement.node, "INCLUDES", flexibility)
tx.create(fl)

ex = Relationship(deentanglement.node, "INCLUDES", explainability)
tx.create(ex)

content = Information("Content")
g.push(content)

style = Information("Style")
g.push(style)

tx.create(Relationship(deentanglement, "OF", content))
tx.create(Relationship(deentanglement, "OF", style))

tx.commit()
'''


'''
g = add_stuff_graphobjects(g)
print(g)
sys.exit()

tx = g.begin()

challenges = g.find("Challenges")
scalability, flexibility, explainability = [c for c in challenges]

search = Node("Scenarios", name="Search_")
search_scalability = Relationship(explainability, "IN", search)
search_explainability = Relationship(scalability, "IN", search)

search = search_scalability + search_explainability
tx.create(search)

tx.commit()


'''
