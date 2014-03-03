import ivy
from ivy.vis.symbols import leafspace_triangles

d = {}
with open('EBnonpassfamiliessppSR.txt') as f:
    for line in f:
        v = line.strip().split('\t')
        d[v[0]] = float(v[1])

r = ivy.tree.read('EBnonpassfamiliesSR.newick')
for lf in r.leaves():
    lf.leafspace = (d[lf.label]-1)/10.0
    # uncomment to strip off the genus names
    #lf.label = lf.label.split('_')[-1]

f = ivy.vis.hardcopy.TreeFigure(r, relwidth=0.75)
p = f.axes
outf = '/tmp/EBnonpassfamiliesSR.pdf'
f.savefig(outf)
f.figure.set_size_inches(20,50)
leafspace_triangles(p, color='green', rca=0.5)

# this needs to be called a few times to adjust spacing
p.home()
f.savefig(outf)
p.home()
f.savefig(outf)
p.home()
f.savefig(outf)
