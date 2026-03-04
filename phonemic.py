from pgmpy.models import BayesianNetwork
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import os
import pandas as pd




def get_expert_graph():
    nodes = ['age_of_acquisition','avg_cluster_size','num_switches', 'num_clusters',
         'speech_rate','word_frequency_mean','word_length_mean', 'total_words', 'cognitive_impairment']
    edges = [('cognitive_impairment', 'avg_cluster_size'), 
             ('cognitive_impairment', 'num_switches'), 
             ('cognitive_impairment', 'speech_rate'), 
            ('cognitive_impairment', 'num_clusters'), 
             ('cognitive_impairment', 'word_length_mean'), 
             ('cognitive_impairment', 'age_of_acquisition'), 
             ('cognitive_impairment', 'word_frequency_mean'),
             ('avg_cluster_size', 'num_switches'),
             ('avg_cluster_size', 'speech_rate'),
             ('age_of_acquisition', 'word_length_mean'),
             ('age_of_acquisition', 'word_frequency_mean'),
             ('word_length_mean', 'speech_rate'),
             ('total_words', 'speech_rate'),
            ('num_clusters', 'avg_cluster_size'),
            ('cognitive_impairment', 'total_words')
             ]
    bn = BayesianNetwork()
    bn.add_nodes_from(nodes)
    for edge in edges:
        bn.add_edge(edge[0], edge[1])

    print(f"Nodes: {bn.nodes()}")
    print(f"Edges: ")
    for edge in bn.edges():
        print(f"{edge[0]} -> {edge[1]}")
    
    return bn
     
    



if __name__ == '__main__':
    bn = get_expert_graph()