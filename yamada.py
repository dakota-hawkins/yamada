import networkx as nx
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from sortedcontainers import SortedSet

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
    

class YamadaSubstitute(object):
    """Class for the substitute algorithm from Yamada et al. 2010."""

    def __init__(self, graph, tree, fixed_edges, restricted_edges,
                 ordered=False):
        self.graph = graph
        self.tree = tree
        self.fixed_edges = fixed_edges
        self.restricted_edges = restricted_edges
        self.ordered = ordered


    def instantiate_substitute_variables(self):
        self.directed = self.tree.to_directed()  # directed graph for postorder
        self.postorder_nodes, self.descendants = self.postorder_tree()
        # set Q in original paper
        self.quasi_cuts = SortedSet(key=lambda x: (x[0], x[1], x[2])) 

    def random_node(self, seed=None):
        """Select random node from graph."""
        if seed is not None:
            np.random.seed(seed)
        r_idx = np.random.randint(0, high=self.graph.number_of_nodes())
        return(list(self.graph.nodes)[r_idx])

    def find_incident_edges(self, node):
        """
        Find all incident edges for a given node not in the current minimum
        spanning tree nor a set of restricted edges.

        Arguments:
            node (int): node of interest.
        
        Return:
            (set): set of weighted incident edges to `node` not contained in
                `restricted_edges`.
        """
        incident_set = set()
        for neighbor in nx.neighbors(self.graph, node):
            edge = (node, neighbor)
            if edge not in self.restricted_edges and edge not in self.tree.edges():
                w_edge = (self.graph.get_edge_data(*edge)['weight'], *edge)
                incident_set.add(w_edge)

        return incident_set

    def postorder_tree(self):
        """
        Invoke postorder ordering on all nodes and edges within the tree given a
        random source node.

        Reorders a tree in a postorder fashion to retrieve descendants and order
        mappings for all nodes within a tree.

        Parameters:
            ordered (boolean): whether the nodes in `self.tree` are already in
                postorder.
        
        Return:
            (dict, dict): tuple of dictionaries. First dictionary maps each node
                in the graph to its postorder position. The second dictionary
                serves as a look up for postorder descendants for each node.
        """
        nodes = self.tree.nodes
        if not self.ordered:
            nodes = nx.dfs_postorder_nodes(self.directed, self.random_node())

        postorder_nodes = OrderedDict()

        # map nodes to their postorder position
        for i, node in enumerate(nodes):
        # for i, node in enumerate(list(self.graph.nodes())):
            postorder_nodes[node] = i + 1
            # remove directed edges not already logged in dictionary
            # --> higher postorder, won't be descendant
            parent_nodes = []
            for neighbor in nx.neighbors(self.directed, node):
                if neighbor not in postorder_nodes:
                    parent_nodes.append((node, neighbor))
            self.directed.remove_edges_from(parent_nodes)

        # map nodes to their postordered descendants
        descendants = {}
        for each in postorder_nodes:
            descendants[each] = [postorder_nodes[each]]
            for child in nx.descendants(self.directed, each):
                descendants[each].append(postorder_nodes[child])
            descendants[each].sort()

        return(postorder_nodes, descendants)

    def postordered_edges(self):
        """Return postorded, weighted edges."""
        edges = []
        for u, v in self.directed.edges():
            w = self.graph.get_edge_data(*(u,v))['weight']
            edges.append((w, v, u)) 
        edges = sorted(edges, key=lambda x: (self.postorder_nodes[x[1]],
                                             self.postorder_nodes[x[2]]))
        return edges


    def equal_weight_descendant(self, weighted_edge):
        weight, node = weighted_edge[0:2]
        for cut_edge in self.quasi_cuts:
            related = self.postorder_nodes[cut_edge[1]] in self.descendants[node]
            if related and cut_edge[0] == weight:
                return(cut_edge)
        return(None)


    def substitute(self):
        """
        Find the substitute-set for a given edge in an minimal-spanning tree.

        Parameters:

        """
        # step 1
        self.instantiate_substitute_variables()
        ordered_edges = self.postordered_edges()
        substitute_dict = {e[1:]: [] for e in ordered_edges}
        print(substitute_dict)

        
        # step 2
        for n_edge in ordered_edges:
            node = n_edge[1]
            incident_edges = self.find_incident_edges(node)
            # step 2.1
            for i_edge in incident_edges:
                reversed_edge = (i_edge[0], *i_edge[1:][::-1])

                # step 2.1.a 
                if self.postorder_nodes[i_edge[2]] < self.descendants[i_edge[1]][0]:
                    if reversed_edge in self.quasi_cuts:
                        self.quasi_cuts.remove(reversed_edge)
                    self.quasi_cuts.add(i_edge)

                # step 2.1.b
                if self.postorder_nodes[i_edge[2]] in self.descendants[i_edge[1]]:
                    if reversed_edge in self.quasi_cuts:
                        self.quasi_cuts.remove(reversed_edge)
                
                # step 2.1.c
                if self.postorder_nodes[i_edge[2]] > self.descendants[i_edge[1]][-1]:
                    self.quasi_cuts.add(i_edge)
            
            #step 2.2
            if n_edge not in self.fixed_edges:
                print(n_edge[1], [x for x in self.quasi_cuts])
                # step 2.2.a
                cut_edge = self.equal_weight_descendant(n_edge)

                while cut_edge is not None:
                    # step 2.2.b
                    if self.postorder_nodes[cut_edge[2]] in self.descendants[node]:
                        self.quasi_cuts.remove(cut_edge)
                        # back to step 2.2.a
                        cut_edge = self.equal_weight_descendant(n_edge)
                    # step 2.2.c                        substitute_dict[n_edge[1:][::-1]].append(cut_edge[1:])
                    else:
                        substitute_dict[n_edge[1:]].append(cut_edge[1:])
                        cut_edge = None

        return(substitute_dict)
    

