import networkx as nx
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt

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
sub_graph = nx.Graph(sub_example)
sub_tree = nx.Graph(sub_tree_example)


class Yamada(object):

    def __init__(self, graph):
        self.instantiate_graph(graph)
        return None

    def instantiate_graph(self, graph):
        """Ensure graph is acyclic, weighted, and complete."""
        self.graph = graph
        return None

    def is_admissible(self, tree, fixed_edges, restricted_edges):
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

    def random_node(self, seed=None):
        """Select random node from graph."""
        if seed is not None:
            np.random.seed(seed)
        r_idx = np.random.randint(0, high=self.graph.number_of_nodes())
        return(list(self.graph.nodes)[r_idx])

    def find_incident_edges(self, tree, node, restricted_edges):
        """
        Find all incident edges for a given node not in either the current
        minimum spanning tree or a set of restricted edges.

        Arguments:
            tree (nx.Graph): a minimum spanning tree.
            node (int): node of interest.
            restricted_set (set): set of edges not to consider.
        
        Return:
            (set): set of incident edges to `node` not contained in
                `restricted_edges`.
        """
        incident_set = set()
        for neighbor in nx.neighbors(self.graph, node):
            edge = (node, neighbor)
            if edge not in restricted_edges and edge not in tree.edges():
                incident_set.add(edge)

        return incident_set

    def postorder_descendants(self, tree):
        """
        Retrieve postorder descendants for all nodes within a tree given a
        random source node.

        Reorders a tree in a postorder fashion to retrieve descendants and order
        mappings for all nodes within a tree.

        Parameters:
            tree (nx.graph): minimum spanning tree of an undirected graph.
        
        Return:
            (dict, dict): tuple of dictionaries. First dictionary maps each node
                in the graph to its postorder position. The second dictionary
                serves as a look up for postorder descendants for each node.
        """
        source_node = self.random_node()
        # convert to directed graph for descendant identification
        directed = tree.to_directed()
        postorder_dict = OrderedDict()

        # map nodes to their postorder position
        for i, node in enumerate(nx.dfs_postorder_nodes(directed, source_node)):
            postorder_dict[node] = i
            # remove directed edges not already logged in dictionary
            # --> higher postorder, won't be descendant
            parent_nodes = []
            for neighbor in nx.neighbors(directed, node):
                if neighbor not in postorder_dict:
                    parent_nodes.append((node, neighbor))
            directed.remove_edges_from(parent_nodes)

        # map nodes to their postordered descendants
        descendant_dict = {}
        for each in postorder_dict:
            descendant_dict[each] = [postorder_dict[each]]
            for child in nx.descendants(directed, each):
                descendant_dict[each].append(postorder_dict[child])
            descendant_dict[each].sort()

        return(postorder_dict, descendant_dict)

    def get_postordered_edge(self, node, postorder_dict, tree):
        possible_edges = [(node, x) for x in nx.neighbors(tree, node)]
        lowest_edge = possible_edges[0]
        for edge in possible_edges:
            node_po = postorder_dict[node]
            edge_po = postorder_dict[edge[1]]
            if edge_po < postorder_dict[lowest_edge[1]] and edge_po > node_po:
                lowest_edge = edge
        return lowest_edge

    def edge_exists(self, node, quasi_cut_set):
        for weighted_edge in quasi_cut_set:
            if weighted_edge[1] == node:
                return True
        return False

    def equal_weight_descendant(self, weighted_edge, quasi_cuts, postorder_dict, descendant_dict):
        weight, node = weighted_edge[0:2]
        new_edge = None
        for cut_edge in quasi_cuts:
            if postorder_dict[cut_edge[1]] in descendant_dict[node]\
            and cut_edge[0] == weight:
                new_edge = cut_edge
        return(new_edge)


    def substitute(self, tree, fixed_edges, restricted_edges):
        """
        Find the substitute-set for a given edge in an minimal-spanning tree.

        Parameters:

        """
        substitute_dict = dict()
        for e in tree.edges:
            substitute_dict[e] = None
            substitute_dict[e[::-1]]= None
        # step 1
        quasi_cuts = set()  # set Q in original paper
        # TODO: ensure lexographical order with respect to w, v
        postorder_dict, descendant_dict = self.postorder_descendants(tree)
        # step 2
        for node in list(postorder_dict)[:-1]:
            incident_edges = self.find_incident_edges(tree, node,
                                                      restricted_edges)
            # step 2.1
            for edge in incident_edges:
                weighted_edge = (self.graph.get_edge_data(*edge)['weight'],
                                 *edge)
                reversed_edge = (weighted_edge[0], *weighted_edge[1:][::-1])

                # step 2.1.a 
                if postorder_dict[edge[1]] < descendant_dict[edge[0]][0]:
                    if reversed_edge in quasi_cuts:
                        quasi_cuts.remove(reversed_edge)
                        quasi_cuts.add(weighted_edge)

                # step 2.1.b
                if postorder_dict[edge[1]] in descendant_dict[edge[0]]:
                    if reversed_edge in quasi_cuts:
                        quasi_cuts.remove(reversed_edge)
                
                # step 2.1.c
                if postorder_dict[edge[1]] > descendant_dict[edge[0]][-1]:
                    quasi_cuts.add(weighted_edge)
            
            #step 2.2
            node_edge = self.get_postordered_edge(node, postorder_dict, tree)
            weight = tree.get_edge_data(*node_edge)['weight']

            if node_edge not in fixed_edges:
                while substitute_dict[node_edge] is None and len(quasi_cuts) > 0:
                    print(quasi_cuts)
                    # step 2.2.a
                    cut_edge = self.equal_weight_descendant((weight, *node_edge),
                                    quasi_cuts, postorder_dict, descendant_dict)
                    # step 2.2.b
                    if cut_edge is not None:
                        if postorder_dict[cut_edge[2]] in descendant_dict[node]:
                            quasi_cuts.remove(cut_edge)
                        # step 2.2.c
                        else:
                            substitute_dict[node_edge] = cut_edge[1:]
                    else:
                        break
            # edge_exists(node, quasi_cuts):
        return(substitute_dict)
    
