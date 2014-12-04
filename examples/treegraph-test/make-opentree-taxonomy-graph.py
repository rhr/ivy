from ivy import treegraph as tg

# this may take a few minutes
g = tg.create_opentree_taxonomy_graph(basepath='ott')

# save the graph to a GraphML file
g.save('ott/ott2.8.gt.gz')

