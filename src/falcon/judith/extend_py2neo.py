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

phseq_file = 'txt.phseq.data'

tx = g.begin()
phones_dict = defaultdict(lambda: len(phones_dict))
f = open(phseq_file)
for line in f:
	line = line.split('\n')[0].split()
	fname = line[0]
	phones = list(set(line[1:]))
	print(fname, phones)
	utt = Node("Utterance", name=fname)
	for phone in phones:
          if phone in phones_dict.keys():
             phone = g.nodes.match("Phones", name=phone).first()
       	     relation = Relationship(utterance, "HAS", phone)
       	     tx.create(relation)
          else:
             phone = Node("Phones", name=phone)	  
             relation = Relationship(utterance, "HAS", phone)
             tx.create(relation)
             phones_dict[phone]

tx.commit()
f.close()

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
