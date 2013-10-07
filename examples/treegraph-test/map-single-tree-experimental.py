import requests
import ivy
from ivy import treegraph_experimental as tg
import graph_tool.all as gt

# instantiate the taxonomic hierarchy
g = tg.TaxonomyGraph.load_from_GML('ncbi/ncbi.xml.gz')

# fetch source tree (id=2) from server as newick and instantiate it as
# an ivy tree (pointer to the root node)
stree = 2
u = 'http://reelab.net/phylografter/stree/newick.txt/%s' % stree
# fields to include in the leaf labels of the newick string: here the
# NCBI taxid is requested
lfmt='snode.id,ottol_name.ncbi_taxid,otu.label'
p = dict(lfmt=lfmt, ifmt='snode.id')
resp = requests.get(u, params=p)

# r is the root node of the tree
r = ivy.tree.read(resp.content)
r.ladderize()
ivy.tree.index(r)

# parse the labels attached to leaf and internal nodes
for n in r:
    if n.isleaf:
        v = n.label.split('_')
        n.snode_id = int(v[0])
        n.taxid = int(v[1]) if (len(v)>1 and
                                v[1] and v[1] != 'None') else None
    else:
        n.snode_id = int(n.label)
r.stree = stree

# determine the taxonomic representation of nodes in r given the
# taxonomy g
g.map_stree(r)

# collect all taxids represented in r and instatiate a new taxonomy
# graph ``taxg`` containing just those taxa
taxids = set()
for lf in r.leaves():
    taxids.update(lf.taxid_rootpath)
taxg = g.taxid_new_subgraph(taxids)

# these store the vertices and edges that trace ``r`` in ``taxg``
verts = taxg.new_vertex_property('bool')
edges = taxg.new_edge_property('bool')

# add r's nodes and branches into taxonomy graph
taxg.merge_stree(r, stree, verts, edges)
# verts and edges now filter the paths traced by r in taxg

# next, add taxonomy edges to taxg connecting 'incertae sedis'
# leaves in r to their containing taxa
for lf in r.leaves():
    if lf.taxid and lf.incertae_sedis:
        taxv = taxg.taxid_vertex[lf.taxid]
        ev = taxg.edge(taxv, lf.v, True)
        if ev:
            assert len(ev)==1
            e = ev[0]
        else:
            e = taxg.add_edge(taxv, lf.v)
        taxg.edge_in_taxonomy[e] = 1

gv = taxg.view(vfilt=verts, efilt=edges)

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

vtext = taxg.new_vertex_property('string')
for v in taxg.vertices():
    if v.out_degree():
        vtext[v] = taxg.vertex_name[v]

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
        vtext[src] = taxg.vertex_name[src]
    if not verts[tgt]:
        verts[tgt] = 1
        pos[tgt] = [0.0, 0.0]
        vcolor[tgt] = 'red'
        vtext[tgt] = taxg.vertex_name[tgt]
    ## if tgt.out_degree()==0:
    ##     taxid = taxg.vertex_taxid[tgt]
    ##     if taxid in r.conflicts:
    ##         for n in r.conflicts[taxid]:
    ##             ne = gv.add_edge(tgt, n.v)
    ##             ecolor[ne] = 'red'
    ##             ewidth[ne] = 1.0
    ##             gv.wt[ne] = 1.0
    ##             edges[ne] = 1
    if not edges[e]:
        edges[e] = 1
        ecolor[e] = 'red'
        ewidth[e] = 1.0
        gv.wt[e] = 1.0

pos = gt.sfdp_layout(gv.g, pos=pos, pin=pin,
                     ## C=10, p=3,# theta=2,
                     ## K=0.1,
                     eweight=gv.wt, 
                     ## mu=0.0,
                     multilevel=False)


gt.interactive_window(
    gv.g, pos=pos, pin=True,
    vertex_fill_color=vcolor,
    vertex_text_position=3.1415,
    ## vertex_text=vtext,
    vertex_text=taxg.vertex_name,
    vertex_size=vsize,
    edge_color=ecolor,
    edge_pen_width=ewidth,
    update_layout=False
    )
