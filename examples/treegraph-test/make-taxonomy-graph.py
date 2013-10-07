from ivy import treegraph as tg

# this may take a few minutes
g = tg.create_ncbi_taxonomy_graph(basepath='ncbi')

# save the graph to a GraphML file
g.save('ncbi/ncbi.xml.gz')

