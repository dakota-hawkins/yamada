"""
Unit tests for yamada.py

@author: Dakota Hawkins
@date: July 11, 2018
"""

import unittest  # unit tests

import networkx as nx  # graph module

import sys  # get to yamada module
sys.path.insert(0, '../')
import yamada


# Helper functions
def unique_undirected_edge(edge1, edge2):
    """
    Test whether two edges link to the same node assuming an undirected graph.
    """
    if edge1 == edge2:
        return False
    elif edge1 == edge2[::-1]:
        return False
    return True

def unique_trees(msts):
    """
    Test whether a list of minimum spanning trees are unique.

    Args:
        msts (list, nx.Graph): list of minimum spanning trees.

    Returns:
        (boolean): whether all trees in the list are unique.
    """
    for i, mst1 in enumerate(msts):
        for j, mst2 in enumerate(msts):
            if i != j:
                for edge1, edge2 in zip(mst1.edges, mst2.edges):
                    if not unique_undirected_edge(edge1, edge2):
                        return False
    return True

def instantiate_k_graph(k):
    """
    Instantiate a unit-weight, complete, and undirected graph with `k` nodes.

    Args:
        k (int): number of nodes in the graph.
    Returns:
        (nx.Graph): a networkx graph object of `k` graph.
    """
    k_graph = nx.Graph()
    nodes = range(1, k + 1)
    k_graph.add_nodes_from(nodes)
    for i in nodes:
        for j in nodes:
            if i != j:
                k_graph.add_edge(i, j, weight=1)
    return k_graph

#TODO: implement this I guess. 
# class HelperTests(unittest.TestCase):


class YamadaTest(unittest.TestCase):

    def setUp(self):
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
        graph = nx.Graph(example)
        tree = {1: {2: {'weight': 2},
                    3: {'weight': 1}},
                2: {4: {'weight': 1}},
                4: {5: {'weight': 1},
                    6: {'weight': 3}}}
        self.tree = nx.Graph(tree)
        self.yamada = yamada.Yamada(graph)

    def test_edge_replacement_presence(self):
        tree = self.yamada.replace_edge(self.tree, (4, 5), (3, 5))
        self.assertTrue((3, 5) in tree.edges)

    def test_edge_replacement_weight(self):
        tree = self.yamada.replace_edge(self.tree, (4, 5), (3, 5))
        self.assertTrue(tree[3][5]['weight'] == 2)

    def test_edge_replacement_removal(self):
        tree = self.yamada.replace_edge(self.tree, (4, 5), (3, 5))
        self.assertTrue((4, 5) not in tree.edges)

    def test_edge_replacement_new_edge_set(self):
        tree = self.yamada.replace_edge(self.tree, (4, 5), (3, 5))
        new_edges = set(tree.edges)
        old_edges = set(self.tree.edges) 
        self.assertTrue(new_edges.difference(old_edges) == set([(3, 5)]))

    def test_edge_replacement_old_edge_set(self):
        tree = self.yamada.replace_edge(self.tree, (4, 5), (3, 5))
        new_edges = set(tree.edges)
        old_edges = set(self.tree.edges) 
        self.assertTrue(old_edges.difference(new_edges) == set([(4, 5)]))


class K3Test(unittest.TestCase):
    """
    Test the minimum spanning trees returned from K3.
    
    K3 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
    """

    def setUp(self):
        k3 = instantiate_k_graph(3)
        k3_yamada = yamada.Yamada(k3)
        self.msts = k3_yamada.spanning_trees()
    
    def test_number_of_msts(self):
        self.assertTrue(len(self.msts) == 3)

    def test_unique_msts(self):
        self.assertTrue(unique_trees(self.msts))


class K4Test(unittest.TestCase):
    """
    Test the minimum spanning trees returned from K4.
    
    K4 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
    """

    def setUp(self):
        k4 = instantiate_k_graph(4)
        k4_yamada = yamada.Yamada(k4)
        self.msts = k4_yamada.spanning_trees()
    
    def test_number_of_msts(self):
        self.assertTrue(len(self.msts) == 16)

    def test_unique_msts(self):
        self.assertTrue(unique_trees(self.msts))

# Currently takes too long on broken implementation
# class K5Test(unittest.TestCase):
#     """
#     Test the minimum spanning trees returned from K5.
    
#     K5 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
#     """

#     def setUp(self):
#         k5 = instantiate_k_graph(5)
#         k5_yamada = yamada.Yamada(k5)
#         self.msts = k5_yamada.spanning_trees()
    
#     def test_number_of_msts(self):
#         self.assertTrue(len(self.msts) == 125)

#     def test_unique_msts(self):
#         self.assertTrue(unique_trees(self.msts))

# Takes too long on currently broken implementation
# class K6Test(unittest.TestCase):
#     """
#     Test the minimum spanning trees returned from K5.
    
#     K5 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
#     """

#     def setUp(self):
#         k6 = instantiate_k_graph(6)
#         k6_yamada = yamada.Yamada(k6)
#         self.msts = k6_yamada.spanning_trees()
    
#     def test_number_of_msts(self):
#         self.assertTrue(len(self.msts) == 1296)

#     def test_unique_msts(self):
#         self.assertTrue(unique_trees(self.msts))


class SubstituteTest(unittest.TestCase):
    """Test Substitute class in yamada.py"""

    def setUp(self):
        sub_example = {1: {2: {'weight': 3},
                           3: {'weight': 12},
                          10: {'weight': 12}},
                       2: {1: {'weight': 3},
                           8: {'weight': 12},
                           10: {'weight': 12}},
                       3: {1: {'weight': 12},
                           4: {'weight': 7},
                           5: {'weight': 10},
                           6: {'weight': 10}},
                       4: {3: {'weight': 7},
                           7: {'weight': 1},
                           10: {'weight': 10}},
                       5: {3: {'weight': 10},
                           6: {'weight': 3},
                           7: {'weight': 13},
                           8: {'weight': 10}},
                       6: {3: {'weight': 10},
                           5: {'weight': 3},
                           7: {'weight': 10}},
                       7: {4: {'weight': 1},
                           5: {'weight': 13},
                           6: {'weight': 10},
                           9: {'weight': 10}},
                       8: {2: {'weight': 12},
                           9: {'weight': 6},
                           5: {'weight': 10}},
                       9: {7: {'weight': 10},
                           8: {'weight': 6},
                           10: {'weight': 7}},
                       10: {1: {'weight': 12},
                            2: {'weight': 12},
                            4: {'weight': 10},
                            9: {'weight': 7}}}
        sub_tree_example = {1: {2: {'weight': 3}},
                            2: {10: {'weight': 12}},
                            10: {9: {'weight': 7}},
                            9: {8: {'weight': 6},
                                7: {'weight': 10}},
                            7: {4: {'weight': 1},
                                6: {'weight': 10}},
                            4: {3: {'weight': 7}},
                            6: {5: {'weight': 3}}}
        self.graph = nx.Graph(sub_example)
        self.tree = nx.Graph(sub_tree_example)

    def test_substitute_edges(self):
        sub = yamada.Substitute(graph=self.graph, tree=self.tree,
                                fixed_edges=set(), restricted_edges=set(),
                                ordered=True)
        sub_edges = sub.substitute()
        self.assertTrue(sub_edges[(1, 2)] is None)
        self.assertTrue(sub_edges[(2, 10)] == (1, 3))
        self.assertTrue(sub_edges[(3, 4)] is None)
        self.assertTrue(sub_edges[(4, 7)] is None)
        self.assertTrue(sub_edges[(5, 6)] is None)
        self.assertTrue(sub_edges[(6, 7)] == (5, 3))
        self.assertTrue(sub_edges[(7, 9)] == (4, 10))
        self.assertTrue(sub_edges[(8, 9)] is None)
        self.assertTrue(sub_edges[(9, 10)] is None)

    def test_no_substitute_edges(self):
        sub = yamada.Substitute(graph=self.tree, tree=self.tree,
                                fixed_edges=set(), restricted_edges=set(),
                                ordered=True)
        sub_edges = sub.substitute()
        self.assertTrue(sub_edges is None)


if __name__ == "__main__":
    unittest.main()