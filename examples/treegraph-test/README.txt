Examples of 'taxonomic normalization' and merging of phylogenies with
a common taxonomic hierarchy, using the ivy.treegraph module.

Note: graph_tool (http://graph-tool.skewed.de) is required in addition
to other ivy dependencies (matplotlib, etc.).

To run the examples, first create the NCBI taxonomy graph:

$ . fetch-taxonomy.sh            # downloads and unpacks the NCBI taxonomy dump
                                 # into subdirectory 'ncbi'

$ python make-taxonomy-graph.py  # creates compressed GML file ncbi/ncbi.xml.gz


Mapping a single tree
=====================

To normalize a single tree, and visualize how it aligns with the NCBI
taxonomy, run map-single-tree.py:

$ python map-single-tree.py

This will fetch a tree from Tank and Donoghue 2010
(http://reelab.net/phylografter/stree/svgView.html/2) in Newick format
from the Phylografter server, determine the taxonomic representation
of its nodes, and display it as a graph, where:

  * green edges are branches of the tree that trace paths in the
    taxonomic hierarchy,

  * blue edges are branches of the tree that are compatible with the
    taxonomic hierarchy,

  * red edges are taxonomic relationships that are contradicted by the
    phylogeny.

The graph window is interactive, and can be zoomed, panned, etc.: see
http://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.GraphWidget


Mapping and merging multiple trees
==================================

First, download a few trees from Phylografter to 'strees.newick':

$ python fetch-strees.py

Next, map them and merge them into a common graph:

$ python merge-trees.py

