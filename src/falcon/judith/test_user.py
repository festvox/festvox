from py2neo import Graph, Node
import os, sys
from py2neo.ogm import GraphObject, Property


password = str(os.environ.get('NEO4J_PASSWORD'))
user = 'neo4j'
#authenticate("localhost:7474", user, password)
graph = Graph("http://localhost:7474/db/data/", password=password)


class User(GraphObject):
    __primarykey__ = 'username'

    username = Property()

    def __init__(self, username):
        self.username = username

    def find(self):
        user = self.match(graph, self.username).first()
        return user

    def register(self, password):
        if not self.find():
            user = Node('User', username=self.username)
            graph.create(user)
            return True
        else:
            return False



lukas = User('lukasscot')
graph.push(lukas)
