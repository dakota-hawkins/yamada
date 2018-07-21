# yamada
Python implementation<sup>*</sup> of the Yamada-Kataoka-Watanabe algorithm to find all minimum spanning trees in an undirected graph.

Implementation mostly follows the `ALL_MST2` algorithm outlined in the original paper. The implementation differs slightly by performing a breadth-first search in liue of a depth-first search. This modification was made so that more variable spanning trees were returned when capping the total number of trees returned.

**Original Paper**

 Yamada, T. Kataoka, S. Watanabe, K. "Listing all the minimum spanning trees in an undirected graph". *International Journal of Computer Mathematics*. Vol 87, No. 14. pp. 3175 - 3185. November 2010.
 
 [Pdf](http://www.nda.ac.jp/~yamada/paper/enum-mst.pdf)
 
 \**Implementation still in progress. Currently does not behave properly.*

