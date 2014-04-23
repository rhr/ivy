import requests
import ivy
from ivy import treegraph as tg
import graph_tool.all as gt

g = tg.load_taxonomy_graph('ncbi/ncbi.xml.gz')

stree = 2 # Tank & Donoghue 2010 - Campanulidae
u = 'http://reelab.net/phylografter/stree/newick.txt/%s' % stree
lfmt='snode.id,ottol_name.ncbi_taxid,otu.label'
p = dict(lfmt=lfmt, ifmt='snode.id')
resp = requests.get(u, params=p)
r = ivy.tree.read(resp.content)
r.ladderize()
ivy.tree.index(r)
for n in r:
    if n.isleaf:
        v = n.label.split('_')
        n.snode_id = int(v[0])
        n.taxid = int(v[1]) if (len(v)>1 and
                                v[1] and v[1] != 'None') else None
    else:
        n.snode_id = int(n.label)
r.stree = stree

tg.map_stree(g, r)
taxids = set()
for lf in r.leaves():
    taxids.update(lf.taxid_rootpath)
taxg = tg.taxid_new_subgraph(g, taxids)
# taxg is a new graph containing only the taxids in stree

# these properties will store the vertices and edges that are traced
# by r
verts = taxg.new_vertex_property('bool')
edges = taxg.new_edge_property('bool')

# add stree's nodes and branches into taxonomy graph
tg.merge_stree(taxg, r, stree, verts, edges)
# verts and edges now filter the paths traced by r in taxg

# next, add taxonomy edges to taxg connecting 'incertae sedis'
# leaves in stree to their containing taxa
for lf in r.leaves():
    if lf.taxid and lf.taxid in taxg.taxid_vertex and lf.incertae_sedis:
        taxv = taxg.taxid_vertex[lf.taxid]
        ev = taxg.edge(taxv, lf.v, True)
        if ev:
            assert len(ev)==1
            e = ev[0]
        else:
            e = taxg.add_edge(taxv, lf.v)
        taxg.edge_in_taxonomy[e] = 1

# make a view of taxg that keeps only the vertices and edges traced by
# the source tree
gv = tg.graph_view(taxg, vfilt=verts, efilt=edges)
gv.vertex_strees = taxg.vertex_strees
gv.edge_strees = taxg.edge_strees

# the following code sets up the visualization
ecolor = taxg.new_edge_property('string')
for e in taxg.edges():
    est = taxg.edge_strees[e]
    eit = taxg.edge_in_taxonomy[e]
    if len(est) and not eit: ecolor[e] = 'blue'
    elif len(est) and eit: ecolor[e] = 'green'
    else: ecolor[e] = 'yellow'

ewidth = taxg.new_edge_property('int')
for e in taxg.edges():
    est = taxg.edge_strees[e]
    if len(est): ewidth[e] = 3
    else: ewidth[e] = 1

vcolor = taxg.new_vertex_property('string')
for v in taxg.vertices():
    if not taxg.vertex_in_taxonomy[v]: vcolor[v] = 'blue'
    else: vcolor[v] = 'green'

vsize = taxg.new_vertex_property('int')
for v in taxg.vertices():
    if taxg.vertex_in_taxonomy[v] or v.out_degree()==0:
        vsize[v] = 4
    else: vsize[v] = 2

pos, pin = tg.layout(taxg, gv, gv.root, sfdp=True, deg0=195.0,
                     degspan=150.0, radius=400)

for v in gv.vertices(): pin[v] = 1

for e in taxg.edges():
    src = e.source()
    tgt = e.target()
    if not verts[src]:
        verts[src] = 1
        pos[src] = [0.0, 0.0]
        vcolor[src] = 'red'
    if not verts[tgt]:
        verts[tgt] = 1
        pos[tgt] = [0.0, 0.0]
        vcolor[tgt] = 'red'
    if not edges[e]:
        edges[e] = 1
        ecolor[e] = 'red'
        ewidth[e] = 1.0
        gv.wt[e] = 1.0

pos = gt.sfdp_layout(gv, pos=pos, pin=pin, eweight=gv.wt, multilevel=False)

gt.interactive_window(
    gv, pos=pos, pin=True,
    vertex_fill_color=vcolor,
    vertex_text_position=3.1415,
    vertex_text=taxg.vertex_name,
    vertex_size=vsize,
    edge_color=ecolor,
    edge_pen_width=ewidth,
    update_layout=False
    )
