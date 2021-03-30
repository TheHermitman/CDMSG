import networkx as nx
import sys


if __name__=='__main__':
    # g = nx.karate_club_graph()
    g = nx.read_gml("network_nx.gml")
    print(g.nodes)
    # nx.write_gml(g, 'karate.gml')
