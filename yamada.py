import networkx as nx
from collections import OrderedDict
from sortedcontainers import SortedSet
from numpy import random, inf

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

# TODO: Same source node for postordering, necessary?


def is_weighted(graph):
    """
    Determine if graph has a 'weight' attribute.
    
    Args:
        graph (nx.Graph): graph to test.
    Returns:
        (boolean): whether the graph has a 'weight' attribute associated with
        each edge in the graph. 
    """
    for edge in graph.edges():
        edge_data = graph.get_edge_data(*edge)
        try:
            edge_data['weight']
        except KeyError:
            return False
    return True


def has_self_cycles(graph):
    """
    Determine if nodes in a graph contain self-cycles.

    Args:
        graph (nx.Graph): graph to test.
    Returns:
        (boolean): whether any node in the graph has an edge connecting to
            itself.
    """
    edges = graph.edges()
    for node in graph.nodes():
        if (node, node) in edges:
            return True
    return False


def check_input_graph(graph):
    """
    Ensure a graph is weighted, has no self-cycles, and is connected.
    """
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be a connected.")
    if has_self_cycles(graph):
        raise ValueError("Input graph must have no self-cycles.")
    if not is_weighted(graph):
        raise ValueError("Input graph must have weighted edges.")


def is_tree_of_graph(child, parent):
    """
    Determine if a potential child graph is a tree of a parent graph.

    Args:
        child (nx.Graph): potential child graph of `parent`.
        parent (nx.Graph): proposed parent graph of `child`.

    Returns:
        (boolean): whether `child` is a tree with all of its edges found in
            `parent`.
    """
    parent_edges = parent.edges()
    for child_edge in child.edges():
        if child_edge not in parent_edges:
            return False
    return nx.is_tree(child)


def check_input_tree(tree, parent_graph):
    """
    Ensure a proposed tree is a child of the parent graph, that the tree is
    weighted, has no self-cycles, and is connected.
    """
    check_input_graph(tree)
    if not is_tree_of_graph(tree, parent_graph):
        raise ValueError("Input tree is not a spanning tree.")


class Yamada(object):

    def __init__(self, graph, n_trees=inf):
        self.instantiate_graph(graph)
        self.trees = []  # minimum spanning trees of graph
        self.n_trees = n_trees
    

    def instantiate_graph(self, graph):
        """Ensure graph has no self-cycles, is weighted, and connected."""
        check_input_graph(graph)
        self.graph = graph

    def is_admissible(self, tree, fixed_edges, restricted_edges):
        """
        Test whether a spanning tree is FR-admissible.

        As defined by Yamada et al., A spanning tree, T, is FR-admissible if
        and only if all edges in F are in T, and R and T are disjoint.

        Args:
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

    def replace_edge(self, tree, old_edge, new_edge):
        """
        Replace an edge in a tree with a substitute edge.

        Args:
            tree (nx.Graph): minumum spanning tree.
            old_edge (tuple (int, int)): edge in `tree` to be replaced.
            new_edge (tuple (int, int)): substitute edge for `old_edge` that
                creates a new minimum spanning tree.
            weight (float): weight of replacement edge.
        Returns:
            (nx.Graph): new minimum spanning tree following edge replacement.
        """
        if new_edge in self.graph.edges():
            tree.remove_edge(*old_edge)
            weight = self.graph[new_edge[0]][new_edge[1]]['weight']
            tree.add_edge(*new_edge, weight=weight)
        else:
            raise ValueError("{} is not contained in parent graph"\
                             .format(new_edge))
        return tree

    def spanning_trees(self):
        """
        Find all minimum spanning trees contained in `self.graph`

        Returns:
            (list, nx.Graph): list of all discovered minimum spanning trees. 
        """
        tree = nx.minimum_spanning_tree(self.graph)
        self.trees.append(tree)
        mst_edge_sets = self.new_spanning_trees(tree, set(), set())
        while len(mst_edge_sets) > 0 and len(self.trees) < self.n_trees:
            # container for generated edge sets
            new_edge_sets = []
            i = 0
            for each in mst_edge_sets:
                # ensure number of trees does not exceed threshold
                if len(self.trees) < self.n_trees:
                    # generate new spanning trees and their associated edge sets
                    print(i, each)
                    edge_set = self.new_spanning_trees(each['tree'],
                                                       each['fixed'],
                                                       each['restricted'])
                    # append every newly discovered tree
                    for every in edge_set:
                        new_edge_sets.append(every)
                    i += 1
            # re-assign edge sets for looping
            mst_edge_sets = new_edge_sets

        return self.trees
        
    def new_spanning_trees(self, tree, fixed_edges, restricted_edges):
        """
        All_MST2 algorithm from Yamada et al. 2010 to find all minimum spanning
        trees.

        The algorithm is modified for a breadth-first search in lieu of a depth
        first search. This difference is motivated by the possibility of capping
        the number of spanning trees returned. It was reasoned a capped
        depth-first search could return less variable tree structures.
        Therefore, a breadth-first approach was preferred.

        Args:
            tree (nx.Graph): current minimum spanning tree for `self.graph`. 
            fixed_edges (set): set of fixed edges as defined in Yamada et al.
                2010.
            restricted_edges (set): set of restricted edges as defined in
                Yamada 2010.
        Returns:
            (list, dict): list of dictionaries containing newly discovered
                minimum spanning trees and their associated fixed and
                restricted edge sets. Dictionary keys are 'tree', 'fixed', and
                'restricted', respectively.
        """
        # find substitute edges -> step 1 in All_MST2 from Yamada et al. 2010
        step_1 = Substitute(self.graph, tree, fixed_edges, restricted_edges)
        s_edges = step_1.substitute()
        edge_sets = []
        if s_edges is not None:
            for i, edge in enumerate(s_edges):
                if s_edges[edge] is not None:
                    # create new minimum spanning tree with substitute edge
                    new_edge = s_edges[edge]
                    tree_i = self.replace_edge(tree, edge, new_edge)

                    # add new tree to list of minimum spanning trees
                    self.trees.append(tree_i)

                    # update F and R edge sets 
                    fixed_i = fixed_edges.union(list(s_edges.keys())[0:i])
                    restricted_i = restricted_edges.union([edge])
                    edge_sets.append({'tree': tree_i,
                                      'fixed': fixed_i,
                                      'restricted': restricted_i})
                    
                    # break tree generation if the number of MSTs exceeds limit
                    if len(self.trees) == self.n_trees:
                        break

        return edge_sets

class Substitute(object):
    """
    Substitute algorithm from Yamada et al. 2010.
    
    Attributes:
        graph (nx.Graph): undirected graph.
        tree (nx.Graph): minimum spanning tree of `graph`
        fixed_edges (set): set of fixed edges as described in Yamata et al. 2010.
        restricted_edges (set): set of restricted edges as described in Yamata
            et al. 2010.
        ordered (boolean): whether indices in `tree` are already postordered.
        postorder_nodes (dict, int:int): dictionary mapping original nodes to
            their postorder index. Instantiated during `substitute()` call.
        descendants (dict, int:list[int]): dictionary mapping nodes to their
            postordered descendants. Descendants referenced by their postorder
            index. Instantiated during `subsitute()` call. 
        directed (nx.Graph): directed graph of `tree` respecting postordered
            nodes. Instantiated during `substitute()` call.
        quasi_cutes (SortedSet, (w, u, v)): ordered sets of possible substitute
            edges. Sorted by w, u, and then v. Instantiated during
            `substitute()` call. 
    """

    def __init__(self, graph, tree, fixed_edges, restricted_edges,
                 ordered=False):
        """
        Substitute algorithm from Yamada et al. 2010.

        Args:
            graph (nx.Graph): undirected graph.
            tree (nx.Graph): minimum spanning tree of `graph`
            fixed_edges (set): set of fixed edges as described in Yamata et al.
                2010.
            restricted_edges (set): set of restricted edges as described in
                Yamata et al. 2010.
            ordered (boolean): whether indices in `tree` are already postordered.
        """
        check_input_graph(graph)
        self.graph = graph
        check_input_tree(tree, graph)
        self.tree = tree
        self.fixed_edges = fixed_edges
        self.restricted_edges = restricted_edges
        self.ordered = ordered

    def instantiate_substitute_variables(self):
        """Instantiate variables for postordering nodes and quasi cuts."""
        self.directed = self.tree.to_directed()  # directed graph for postorder
        self.postorder_nodes, self.descendants = self.postorder_tree()
        # set Q in original paper
        self.quasi_cuts = SortedSet(key=lambda x: (x[0], x[1], x[2])) 

    def random_node(self, seed=None):
        """
        Select random node from graph.
        
        Args:
            seed (int, optional): random seed to use. Default is None. 
        Returns:
            (int): randomly selected node from `self.graph`.
        """
        if seed is not None:
            random.seed(seed)
        r_idx = random.randint(0, high=self.graph.number_of_nodes())
        return(list(self.graph.nodes)[r_idx])

    def find_incident_edges(self, node):
        """
        Find all incident edges for a given node not in the current minimum
        spanning tree nor a set of restricted edges.

        Args:
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

        Returns:
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
        """
        Find an equal-weight descendant of the origin node for a provided edge.

        Finds a edge (x, y, z) in `self.quasi_cuts` such that the starting node,
        `y`, is a postorder descendant of a the starting node, `u`, in the
        provided edge (w, u, v) and x == w.

        Args:
            weighted_edge (tuple, (int, int, int)): tuple representation of a
                weighted edge with the form (w, u, v): `w` is the weight of the
                edge, `u` is the starting node of the edge, and `v` is the final
                node of the edge.
        Returns:
            tuple (int, int, int): returns tuple representation of the first
                discovered equal-weighted descendant edge. Returns None if no
                such edge exists.
        """
        weight, node = weighted_edge[0:2]
        for cut_edge in self.quasi_cuts:
            related = self.postorder_nodes[cut_edge[1]] in self.descendants[node]
            if related and cut_edge[0] == weight:
                return(cut_edge)
        return(None)

    def _create_substitute_dict(self, ordered_edges):
        """
        Create dictionary linking edges to their substitutes.

        Args:
            ordered_edges (list, tuple (u, v)): list of postordered edges.
        Returns:
            (OrderedDict): dictionary linking edges to their substitutes.
        """
        substitute_dict = OrderedDict()
        for e in ordered_edges:
            substitute_dict[e[1:]] = None
        return substitute_dict
        

    def substitute(self):
        """
        Finds all substitute edges for a minimum spanning tree.

        Returns:
            (dict, (tuple(u, v): [(x, y)]): dictionary mapping edges in `tree`
                to list of possible substitute edges. 
        """
        # step 1
        self.instantiate_substitute_variables()
        ordered_edges = self.postordered_edges()
        substitute_dict = None
        
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

                # step 2.2.a
                cut_edge = self.equal_weight_descendant(n_edge)
                while cut_edge is not None:
                    
                    # step 2.2.b
                    if self.postorder_nodes[cut_edge[2]] in self.descendants[node]:
                        self.quasi_cuts.remove(cut_edge)

                        # back to step 2.2.a
                        cut_edge = self.equal_weight_descendant(n_edge)

                    # step 2.2.c
                    else:
                        if substitute_dict is None:
                            substitute_dict = self._create_substitute_dict(
                                                                  ordered_edges)

                        substitute_dict[n_edge[1:]] = cut_edge[1:]
                        cut_edge = None

        return(substitute_dict)
    

