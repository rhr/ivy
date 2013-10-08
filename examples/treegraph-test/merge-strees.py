import ivy, requests
from ivy import treegraph as tg
from collections import defaultdict
import graph_tool.all as gt

g = tg.load_taxonomy_graph('ncbi/ncbi.xml.gz')

strees = []
i = 1
with open('strees.newick') as f:
    for line in f:
        r = ivy.tree.read(line)
        ivy.tree.index(r)
        for n in r:
            if n.isleaf:
                v = n.label.split('_')
                n.snode_id = int(v[0])
                n.taxid = int(v[1]) if (len(v)>1 and
                                        v[1] and v[1] != 'None') else None
            else:
                n.snode_id = int(n.label)
        r.stree = i
        strees.append(r)
        i += 1

stree2color = {}
for i, r in enumerate(strees):
    stree2color[r.stree] = tg.color20[i % 20]

taxids = set()
for r in strees:
    tg.map_stree(g, r)
    for lf in r.leaves(): taxids.update(lf.taxid_rootpath)

taxg = tg.taxid_new_subgraph(g, taxids)

verts = taxg.new_vertex_property('bool')
edges = taxg.new_edge_property('bool')
for r in strees:
    tg.merge_stree(taxg, r, r.stree, verts, edges)
 
root = taxg.root

gv = tg.graph_view(taxg, vfilt=verts, efilt=edges)

for x in gv.vertices():
    if x.in_degree()==0 and int(x)!=int(taxg.root):
        v = taxg.vertex(int(x))
        while 1:
            e = v.in_edges().next()
            edges[e] = 1
            p = e.source()
            if verts[p]: break
            else:
                verts[p] = 1
                v = p

# the following code sets up the visualization
ecolor = gv.new_edge_property('string')
for e in gv.edges():
    est = taxg.edge_strees[e]
    if len(est)==1: ecolor[e] = stree2color[est[0]]
    else: ecolor[e] = 'gray'

ewidth = gv.new_edge_property('int')
for e in gv.edges():
    est = taxg.edge_strees[e]
    if len(est): ewidth[e] = 1+len(est)
    else: ewidth[e] = 1

vcolor = gv.new_vertex_property('string')
for v in gv.vertices(): vcolor[v] = 'gray'
for v in gv.vertices():
    if taxg.vertex_in_taxonomy[v]: vcolor[v] = 'green'
    else:
        vcolor[v] = stree2color[taxg.vertex_strees[v][0]]
        if len(taxg.vertex_strees[v])>1:
            vcolor[v] = 'yellow'
    s = (taxg.vertex_name[v] or '').lower()

vsize = gv.new_vertex_property('int')
for v in gv.vertices():
    n = len(taxg.vertex_strees[v])
    if n: vsize[v] = n+2
    else: vsize[v] = 1

vtext = gv.new_vertex_property('string')
for v in gv.vertices():
    if v.out_degree():
        s = taxg.vertex_name[v]
        vtext[v] = s

dprops = gv.new_vertex_property('string')
for v in gv.vertices():
    st = str(list(taxg.vertex_strees[v]))
    s = taxg.vertex_name[v]
    if not s:
        s = str([ taxg.taxid_name(x) for x in taxg.vertex_stem_cdef[v] ])
    s = '%s: %s' % (st, s)
    dprops[v] = s
    
pos, pin = tg.layout(taxg, gv, root, sfdp=True, deg0=190.0,
                     degspan=160.0, radius=1000)

gt.interactive_window(
    gv, pos=pos, pin=True,
    vertex_fill_color=vcolor,
    vertex_text_position=3.1415,
    vertex_text=vtext,
    vertex_size=vsize,
    vertex_pen_width=0,
    edge_color=ecolor,
    edge_pen_width=ewidth,
    display_props=dprops,
    update_layout=False
    )
