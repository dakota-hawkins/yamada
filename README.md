# yamada
Python implementation of the Yamada-Kataoka-Watanabe algorithm to find all minimum spanning trees in an undirected graph.

Implementation mostly follows the `ALL_MST2` algorithm outlined in the original paper. The implementation differs slightly by performing a breadth-first search in liue of a depth-first search. This modification was made so that more variable spanning trees were returned when capping the total number of trees returned.

[**Original Paper**](http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf)

 Yamada, T. Kataoka, S. Watanabe, K. "Listing all the minimum spanning trees in an undirected graph". *International Journal of Computer Mathematics*. Vol 87, No. 14. pp. 3175 - 3185. November 2010.
 
 
 ## Example
 ```Python
import yamada
import networkx as nx
 
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

# retrieve all minimum spanning trees 
graph_yamada = yamada.Yamada(graph)
all_msts = graph_yamada.spanning_trees()
print(len(all_msts))

# retrieve fixed number of minimum spanning trees
graph_yamada = yamada.Yamada(graph, n_trees=3)
msts = graph_yamada.spanning_trees()
print(len(msts))
 ```

