# yamada
Python implementation of the Yamada-Kataoka-Watanabe algorithm to find all minimum spanning trees in an undirected graph.

Implementation mostly follows the `ALL_MST2` algorithm outlined in the original paper. The implementation differs slightly by performing a breadth-first search in liue of a depth-first search. This modification was made so that more variable spanning trees were returned when capping the total number of trees returned.

[**Original Paper**](http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf)

Yamada, T. Kataoka, S. Watanabe, K. "Listing all the minimum spanning trees in an undirected graph". *International Journal of Computer Mathematics*. Vol 87, No. 14. pp. 3175 - 3185. November 2010.
 
## Installation
This module is not currently hosted through popular package managing tools such as `pip` or `conda`. Therefore, simply cloning, downloading, or forking this repository is the best way to install the package. To ensure global access to the module, you'll want to add the repository location to your `PYTHONPATH`.
 
## Tests
Proper implementation was tested using the examples found in the original paper, and implementation of those tests can be found in the `test` subdirectory. The graph structure used in Figure 3 of the original paper, is used to explicitly test for exact minimum spanning tree membership. Meanwhile, the unit-weight, complete graphs k<sub>i</sub> are tested for unique membership and expected length for i in {3, 4, 5, 6}. The `Substitute()` algorithm is tested using the example found in table 3 of the original paper.
 
 To run the tests simply execute the following command:
 
 ```
 python tests/test_yamada.py
 ```
 
## Dependencies
 
This module depends on the `numpy`, `networkx`, `collections`, `sortedcontainers`, `sys`, and `unittest` packages, and was written in Python 3.6. The exact requirements can be found in the `requirements.txt` file. A `yamada.yaml` file is also provided for `conda` environment creation.
 
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

