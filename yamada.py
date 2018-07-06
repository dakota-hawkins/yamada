import networkx as nx
import numpy as np

example = {1: {2: {'weight': 2},
               3: {'weight': 1}},
           2: {1: {'weight': 2},
               3: {'weight': 3},
               4: {'weight': 1}},
           3: {1: {'weight': 1},
               2: {'weight': 3},
               4: {'weight': 2},
               5: {'weight': 2}},
           4: {2: {'weight': 1},
               3: {'weight': 2},
               5: {'weight': 1},
               6: {'weight': 3}},
           5: {3: {'weight': 2},
               4: {'weight': 1},
               6: {'weight': 3}},
           6: {4: {'weight': 3},
               5: {'weight': 3}}}
example_graph = nx.Graph(example)


class yamada(object):

    def __init__(self, graph):
        self.__instantiate_graph(graph)
        return None

    def __instantiate_graph(self, graph):
        """Ensure graph is a cyclic, weighted, and complete."""
        self.graph = graph
        return None

    def __is_admissible(tree, fixed_edges, restricted_edges):
        """
        Test whether a spanning tree is FR-admissible.

        As defined by Yamada et al., A spanning tree, T, is FR-admissible if
        and only if all edges in F are in T, and R and T are disjoint.

        Parameters:
            tree (nx.Graph): minimum spanning tree.
            fixed_edges(list-like): container of fixed edges.
            restricted_edges(list-like): container of restricted edges.
        Return:
            (Boolean): whether `tree` is FR-admissible
        """
        # Test F is subset of T
        for edge in fixed_edges:
            if edge not in tree.edges:
                return(False)
        # Test T and R disjoint
        if len(set(restricted_edges).intersection(set(tree.edges))) != 0:
            return(False)
        return(True)

    def random_node(self):
        r_idx = np.random.randint(0, high=self.graph.number_of_nodes())
        return(self.graph.nodes[r_idx])

    def substitute(self, tree, fixed_edges, restricted_edges):
        """
        Find the substitute-set for a given edge in an minimal-spanning tree.
        """

        return(None)
    
