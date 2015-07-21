from rpy2.robjects.packages import importr
import numpy as np
from ivy.interactive import *

treefile = ("/home/rree/Dropbox/seminars/botany-2015/"
            "ultrametric.tse.STATE_2537000.scaled100.newick")

# minimal R->python data transfer
ape = importr('ape')
bamm = importr('BAMMtools')
tree = ape.read_tree(treefile)
edata = bamm.getEventData(tree, eventdata="bamm_tse_event_data.txt", burnin=0.2)
dtrates = bamm.dtRates(edata, 0.01, tmat=True).rx2('dtrates')
nodeidx = np.array(dtrates.rx2('tmat').rx(True, 1), dtype=int)
rates = np.array(dtrates.rx2('rates'))
netdiv = rates[0]-rates[1]

r = ivy.tree.read(treefile)
i = 1
for lf in r.leaves():
    lf.apeidx = i
    i += 1
for n in r.clades():
    n.apeidx = i
    i += 1

f = treefig(r)

# collect segment and value data for a matplotlib LineCollection
segments = []
values = []
for n in r.descendants():
    n.rates = netdiv[nodeidx==n.apeidx]
    c = f.detail.n2c[n]
    pc = f.detail.n2c[n.parent]
    seglen = (c.x-pc.x)/len(n.rates)
    for i, rate in enumerate(n.rates):
        x0 = pc.x+i*seglen
        x1 = x0+seglen
        segments.append(((x0, c.y), (x1, c.y)))
        values.append(rate)
    segments.append(((pc.x, pc.y), (pc.x, c.y)))
    values.append(n.rates[0])

from matplotlib.cm import coolwarm
from matplotlib.collections import LineCollection
lc = LineCollection(segments, cmap=coolwarm, lw=2)
lc.set_array(np.array(values))
f.detail.add_collection(lc)

# update the figure
f.figure.canvas.draw_idle()



















