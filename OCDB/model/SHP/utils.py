import networkx as nx
import numpy as np

def check_DAG(edge_mat):
    c_g = nx.from_numpy_array(edge_mat - np.diag(np.diag(edge_mat)), create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(c_g)