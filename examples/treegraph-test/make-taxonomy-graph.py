from ivy import treegraph as tg
import datetime
t = datetime.date.today()

# this may take a few minutes
g = tg.create_ncbi_taxonomy_graph(basepath='ncbi')

# save the graph to a file in graph-tool's binary format, gzipped
g.save('ncbi/ncbi.{}.gt.gz'.format(t.strftime('%Y%m%d')))

