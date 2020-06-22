from py2neo import Graph, Node, Relationship
from py2neo.ogm import GraphObject, Property, RelatedTo
import os, sys

password = str(os.environ.get('NEO4J_PASSWORD'))

class Dissertation(GraphObject):

    __primarykey__ = 'title'

    title = Property()
    consists = RelatedTo("Dissertation")

    def __init__(self, title):
        self.title = title

    def find(self):
        dissertation = self.match(graph, self.title).first()
        return dissertation

    def register(self):
        if not self.find():
            dissertation = Node('Dissertation', title=self.title)
            self.node = dissertation
            graph.create(dissertation)
            print("Created ", self.title)
            return True
        else:
            return False


class Challenge(GraphObject):

    __primarykey__ = 'name'

    name = Property()

    def __init__(self, name):
        self.name = name

    def find(self):
        challenge = self.match(graph, self.challenge).first()
        return challenge

    def register(self):
        if not self.find():
            challenge = Node('Challenge', name=self.name)
            graph.create(challenge)
            return True
        else:
            return False

class Information(GraphObject):

    __primarykey__ = 'name'

    name = Property()

    def __init__(self, name):
        self.name = name

    def find(self):
        info = self.match(graph, self.name).first()
        return info

    def register(self):
        if not self.find():
            info = Node('Information', name=self.name)
            graph.create(info)
            return True
        else:
            return False            


def add_concepts(g):

   tx = g.begin()

   deentanglement = Node("Dissertation", name='DeEntanglement')
   tx.create(deentanglement)

   scalability = Node("Challenges", name="Scalability")
   tx.create(scalability)

   flexibility = Node("Challenges", name="Flexibility")
   tx.create(flexibility)

   explainability = Node("Challenges", name="Explainability")
   tx.create(explainability)

   sc = Relationship(deentanglement, "INCLUDES", scalability)
   tx.create(sc)

   fl = Relationship(deentanglement, "INCLUDES", flexibility)
   tx.create(fl)

   ex = Relationship(deentanglement, "INCLUDES", explainability)
   tx.create(ex)

   content = Node("Information", name="Content")
   tx.create(content)
   tx.create(Relationship(deentanglement, "OF", content))

   style = Node("Information", name="Style")
   tx.create(style)
   tx.create(Relationship(deentanglement, "OF", style))

   misc = Node("Information", name="misc")
   tx.create(misc)
   tx.create(Relationship(deentanglement, "OF", misc))

   priors = Node("Paradigms", name="Priors")
   tx.create(priors)
   tx.create(Relationship(deentanglement, "BY", priors))

   divergences = Node("Paradigms", name="Divergences")
   tx.create(divergences)
   tx.create(Relationship(deentanglement, "BY", divergences))

   optimization = Node("Paradigms", name="Optimization")
   tx.create(optimization)
   tx.create(Relationship(deentanglement, "BY", optimization))

   density = Node("Paradigms", name="Density Estimation")
   tx.create(density)
   tx.create(Relationship(deentanglement, "BY", density))

   vae = Node("Frameworks", name="VAE")
   tx.create(vae)
   tx.create(Relationship(priors, "BY", vae))

   gan = Node("Frameworks", name="GAN")
   tx.create(gan)
   tx.create(Relationship(divergences, "BY", gan))

   entanglement = Node("Concepts", name="Entanglement")
   tx.create(entanglement)
   tx.create(Relationship(entanglement, "BY", gan))
   tx.create(Relationship(entanglement, "BY", vae))
   tx.create(Relationship(entanglement, "BY", priors))
   tx.create(Relationship(entanglement, "BY", divergences))
   tx.create(Relationship(entanglement, "BY", optimization))
   tx.create(Relationship(entanglement, "BY", density))



   tx.commit()

   return g


def add_stuff_graphobjects(g):

   deentanglement = Dissertation()
   g.push(deentanglement)

   return g



