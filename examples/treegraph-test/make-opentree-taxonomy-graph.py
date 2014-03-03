from ivy import treegraph as tg

# this may take a few minutes
g = tg.create_opentree_taxonomy_graph(basepath='ott2.2')

# save the graph to a GraphML file
g.save('ott2.2/ott2.2.xml.gz')

