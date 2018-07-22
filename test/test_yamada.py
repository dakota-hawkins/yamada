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
                mst1_edges = set(mst1.edges)
                mst2_edges = set(mst2.edges)
                if len(mst1_edges.difference(mst2_edges)) == 0:
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

def tree_from_edge_list(edge_list, parent_graph):
    """
    Extract tree from a parent graph given a set of edges.

    Args:
        edge_list (list, tuple): list of edges where edges are tuples (u, v)
            such that u and v are nodes in `parent_graph`.
        parent_graph (nx.Graph): parent graph to extract edge data from.
    Returns:
        (nx.Graph): tree of provided nodes with data extracted from parent
            graph. 
    """
    tree = nx.Graph()
    tree.add_nodes_from(parent_graph.nodes)
    for edge in edge_list:
        weight = parent_graph[edge[0]][edge[1]]['weight']
        tree.add_edge(*edge, weight=weight)

    return tree

#TODO: implement this I guess. 
# class HelperTests(unittest.TestCase):

class YamadaHelperTest(unittest.TestCase):

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


class YamadaK3Test(unittest.TestCase):
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


class YamadaK4Test(unittest.TestCase):
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


class YamadaK5Test(unittest.TestCase):
    """
    Test the minimum spanning trees returned from K5.
    
    K5 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
    """

    def setUp(self):
        k5 = instantiate_k_graph(5)
        k5_yamada = yamada.Yamada(k5)
        self.msts = k5_yamada.spanning_trees()
    
    def test_number_of_msts(self):
        self.assertTrue(len(self.msts) == 125)

    def test_unique_msts(self):
        self.assertTrue(unique_trees(self.msts))

class YamadaK6Test(unittest.TestCase):
    """
    Test the minimum spanning trees returned from K5.
    
    K5 is a complete graph of three nodes with fixed weights, w(e_i) = 1.
    """

    def setUp(self):
        k6 = instantiate_k_graph(6)
        k6_yamada = yamada.Yamada(k6)
        self.msts = k6_yamada.spanning_trees()
    
    def test_number_of_msts(self):
        self.assertTrue(len(self.msts) == 1296)

    def test_unique_msts(self):
        self.assertTrue(unique_trees(self.msts))

class YamadaKnownMstTest(unittest.TestCase):
    """
    Test discovered minimum spanning trees from Figure 3 in Yamada et al.
    """

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
        self.graph = nx.Graph(example)
        graph_yamada = yamada.Yamada(self.graph)
        self.msts = graph_yamada.spanning_trees()
    
    def test_number_of_msts(self):
        self.assertTrue(len(self.msts) == 6)

    def test_mst_weights(self):
        weight_test = []
        for each in self.msts:
            weights = sum([each[u][v]['weight'] for u, v in each.edges])
            weight_test.append(weights == 8)
        self.assertTrue(all(weight_test))

    def test_mst1_membership(self):
        mst1_edges = [(1, 2), (1, 3), (2, 4), (4, 5), (4, 6)]
        mst1 = tree_from_edge_list(mst1_edges, self.graph)
        # mst1 should be in discovered msts -> expect non-unique list of msts
        # when mst1 is added
        self.assertTrue(not unique_trees([mst1] + self.msts))

    def test_mst2_membership(self):
        mst2_edges = [(1, 3), (3, 4), (2, 4), (4, 5), (4, 6)]
        mst2 = tree_from_edge_list(mst2_edges, self.graph)
        self.assertTrue(not unique_trees([mst2] + self.msts))

    def test_mst3_membership(self):
        mst3_edges = [(1, 3), (1, 2), (2, 4), (4, 5), (5, 6)]
        mst3 = tree_from_edge_list(mst3_edges, self.graph)
        self.assertTrue(not unique_trees([mst3] + self.msts))

    def test_mst4_membership(self):
        mst4_edges = [(1, 3), (3, 5), (2, 4), (4, 5), (4, 6)]
        mst4 = tree_from_edge_list(mst4_edges, self.graph)
        self.assertTrue(not unique_trees([mst4] + self.msts))

    def test_mst5_membership(self):
        mst5_edges = [(1, 3), (3, 4), (2, 4), (4, 5), (5, 6)]
        mst5 = tree_from_edge_list(mst5_edges, self.graph)
        self.assertTrue(not unique_trees([mst5] + self.msts))

    def test_mst6_membership(self):
        mst6_edges = [(1, 3), (2, 4), (4, 5), (3, 5), (5, 6)]
        mst6 = tree_from_edge_list(mst6_edges, self.graph)
        self.assertTrue(not unique_trees([mst6] + self.msts))

class YamadaEarlyTerminationTest(unittest.TestCase):
    """Test early termination in Yamada class."""

    def test_k3_termination(self):
        k3 = instantiate_k_graph(3)
        k3_yamada = yamada.Yamada(k3, n_trees=1)
        msts = k3_yamada.spanning_trees()
        self.assertTrue(len(msts) == 1)

    def test_k4_termination(self):
        k4 = instantiate_k_graph(4)
        k4_yamada = yamada.Yamada(k4, n_trees=9)
        msts = k4_yamada.spanning_trees()
        self.assertTrue(len(msts) == 9)
    
    def test_k5_termination(self):
        k5 = instantiate_k_graph(5)
        k5_yamada = yamada.Yamada(k5, 53)
        msts = k5_yamada.spanning_trees()
        self.assertTrue(len(msts) == 53)
    
    def test_k6_termination(self):
        k6 = instantiate_k_graph(6)
        k6_yamada = yamada.Yamada(k6, 312)
        msts = k6_yamada.spanning_trees()
        self.assertTrue(len(msts) == 312)

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
                                fixed_edges=set(), restricted_edges=set())
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
                                fixed_edges=set(), restricted_edges=set())
        sub_edges = sub.substitute()
        self.assertTrue(sub_edges is None)


if __name__ == "__main__":
    unittest.main()