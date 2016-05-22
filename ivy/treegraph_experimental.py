from __future__ import absolute_import, division, print_function, unicode_literals

import os, requests, math, pickle, logging
from collections import defaultdict, Counter, namedtuple
from functools import cmp_to_key

from functools import cmp_to_key
import graph_tool.all as gt
from . import newick
from . import tree
newick.add_label_chars('/#&-')

color20 = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
           "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
           "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
           "#17becf", "#9edae5"]

def get_or_create_vp(g, name, ptype):
    p = g.vp.get(name)
    if not p:
        p = g.new_vertex_property(ptype)
        g.vp[name] = p
    return p

def get_or_create_ep(g, name, ptype):
    p = g.ep.get(name)
    if not p:
        p = g.new_edge_property(ptype)
        g.ep[name] = p
    return p

class TaxonomyGraph(object):
    """
    g is a graph_tool.Graph of vertices (taxa) and edges
    (parent->child relationships).

    vertex properties:
       name (string)
       rank (string)
       taxid (int),
       istaxon (bool)
       dubious (bool)
       incertae sedis (bool),
       collapsed (bool)
       snode (int)
       stree (vector<int>)
       stem_cdef (vector<int)

    edge properties:
       istaxon (bool)
       stree (vector<int>)
    """
    def __init__(self):
        self.g = None

    def vertices(self):
        return self.g.vertices()

    def vertex(self, x):
        return self.g.vertex(x)

    def edges(self):
        return self.g.edges()

    def edge(self, s, t, all_edges=False):
        return self.g.edge(s, t, all_edges)

    def add_edge(self, a, b):
        return self.g.add_edge(a, b)
        
    def new_vertex_property(self, x):
        return self.g.new_vertex_property(x)

    def new_edge_property(self, x):
        return self.g.new_edge_property(x)

    def taxid_name(self, taxid):
        v = self.taxid_vertex.get(taxid)
        if v: return self.vertex_name[v]

    def taxid_dubious(self, taxid):
        v = self.taxid_vertex.get(taxid)
        if v: return self.dubious[v]

    def taxid_hindex(self, taxid):
        v = self.taxid_vertex.get(taxid)
        if v: return self.hindex[v]
        else: print('no hindex:', taxid)

    @staticmethod
    def new_from_NCBI_taxdump(basepath):
        '''
        create a TaxonomyGraph containing the NCBI taxonomic hierarchy
        using files from:

        ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz

        which are assumed to be unpacked in the directory ``basepath``
        '''
        self = TaxonomyGraph()
        G = gt.Graph()
        self.g = G
        self.vertex_name = get_or_create_vp(G, 'name', 'string')
        self.vertex_rank = get_or_create_vp(G, 'rank', 'string')
        self.vertex_taxid = get_or_create_vp(G, 'taxid', 'int')
        self.edge_in_taxonomy = get_or_create_ep(G, 'istaxon', 'bool')
        self.vertex_in_taxonomy = get_or_create_vp(G, 'istaxon', 'bool')
        self.dubious = get_or_create_vp(G, 'dubious', 'bool')
        self.incertae_sedis = get_or_create_vp(G, 'incertae_sedis', 'bool')
        self.collapsed = get_or_create_vp(G, 'collapsed', 'bool')
        self.taxid_vertex = {}
        self.edge_strees = get_or_create_ep(G, 'stree', 'vector<int>')
        self.vertex_snode = get_or_create_vp(G, 'snode', 'int')
        self.vertex_strees = get_or_create_vp(G, 'stree', 'vector<int>')
        self.vertex_stem_cdef = get_or_create_vp(G, 'stem_cdef', 'vector<int>')
        self.stem_cdef_vertex = defaultdict(lambda: G.add_vertex())

        node_fields = ["taxid", "parent_taxid", "rank", "embl_code",
                       "division_id", "inherited_div_flag", "genetic_code_id",
                       "inherited_gc_flag", "mt_gc_id", "inherited_mgc_flag",
                       "genbank_hidden_flag", "hidden_subtree_root_flag",
                       "comments"]
        name_fields = ["taxid", "name", "unique_name", "name_class"]
        def process_nodes(f):
            logging.info('...processing nodes')
            n = {}
            c = defaultdict(list)
            i = 0
            for line in f:
                v = [ x.strip() or None for x in line.split("|") ]
                s = dict(list(zip(node_fields, v[:-1])))
                for k, v in list(s.items()):
                    if k.endswith('_flag'):
                        s[k] = bool(int(v))
                    else:
                        try: s[k] = int(v)
                        except: pass
                tid = s['taxid']
                n[tid] = s
                if tid > 1: c[s['parent_taxid']].append(tid)
                i += 1
                print('%s           \r' % i, end=' ')
            print()
            return n, c

        def process_names(f):
            logging.info('...processing names')
            seen = set()
            synonyms = defaultdict(list)
            accepted = {}
            i = 0
            for line in f:
                v = [ x.strip() or None for x in line.split("|") ]
                v[0] = int(v[0])
                s = dict(list(zip(name_fields, v[:-1])))
                name = s['name']; uname = s['unique_name']; taxid = s['taxid']
                s['type'] = 'taxonomic_name'
                s['source'] = 'ncbi'
                if uname or name in seen: s['homonym_flag'] = True
                if (s['name_class'] == 'scientific name' and
                    (uname or name) not in seen):
                    s['primary'] = True
                    accepted[taxid] = s
                else:
                    s['primary'] = False
                    synonyms[taxid].append(s)
                seen.add(uname or name)
                i += 1
                print('%s           \r' % i, end=' ')
            print()
            return accepted, synonyms

        with open(os.path.join(basepath, 'nodes.dmp')) as f:
            nodes, ptid2ctid = process_nodes(f)
        with open(os.path.join(basepath, 'names.dmp')) as f:
            accepted, synonyms = process_names(f)

        nnodes = len(nodes)
        viter = G.add_vertex(nnodes)

        logging.info('...creating graph vertices')

        i = 0
        for tid, d in nodes.items():
            v = G.vertex(i)
            self.vertex_in_taxonomy[v] = 1
            self.taxid_vertex[tid] = v
            self.vertex_taxid[v] = tid
            self.vertex_rank[v] = d['rank']
            try:
                name = accepted[tid]['name'] # !! should deal with unique_name
                self.vertex_name[v] = name
            except KeyError:
                print(tid)
            i += 1
            print('%s           \r' % (nnodes-i), end=' ')
        print()

        logging.info('...creating graph vertices')
        i = 0; n = len(ptid2ctid)
        for tid, child_tids in ptid2ctid.items():
            pv = self.taxid_vertex[tid]
            for ctid in child_tids:
                cv = self.taxid_vertex[ctid]
                e = G.add_edge(pv, cv)
                self.edge_in_taxonomy[e] = 1
            i += 1
            print('%s           \r' % (n-i), end=' ')
        print()

        self._filter()
        self._index_graph(G)
        self.root = G.vertex(0)
        return self

    @staticmethod
    def new_from_graph(g):
        self = TaxonomyGraph()
        self.g = g
        self.vertex_name = g.vp['name']
        self.vertex_taxid = g.vp['taxid']
        self.edge_in_taxonomy = g.ep['istaxon']
        self.vertex_in_taxonomy = g.vp['istaxon']
        self.dubious = g.vp['dubious']
        self.incertae_sedis = g.vp['incertae_sedis']
        self.hindex = g.vp['hindex']
        self.taxid_vertex = {}
        vfilt, vinv = g.get_vertex_filter()
        g.set_vertex_filter(self.vertex_in_taxonomy)
        for v in g.vertices():
            taxid = self.vertex_taxid[v]
            self.taxid_vertex[taxid] = v
        g.set_vertex_filter(vfilt, vinv)
        self.edge_strees = get_or_create_ep(g, 'stree', 'vector<int>')
        self.vertex_snode = get_or_create_vp(g, 'snode', 'int')
        self.vertex_strees = get_or_create_vp(g, 'stree', 'vector<int>')
        self.vertex_stem_cdef = get_or_create_vp(g, 'stem_cdef', 'vector<int>')
        self.stem_cdef_vertex = defaultdict(lambda: g.add_vertex())
        g.set_vertex_filter(self.vertex_in_taxonomy, inverted=True)
        for v in g.vertices():
            cdef = self.vertex_stem_cdef[v]
            if cdef: self.stem_cdef_vertex[tuple(cdef)] = v
        g.set_vertex_filter(vfilt, vinv)
        return self

    @staticmethod
    def load_from_GML(source):
        self = TaxonomyGraph()
        g = gt.load_graph(source)
        self.g = g
        self.vertex_name = g.vp['name']
        self.vertex_taxid = g.vp['taxid']
        self.edge_in_taxonomy = g.ep['istaxon']
        self.vertex_in_taxonomy = g.vp['istaxon']
        self.dubious = g.vp['dubious']
        self.incertae_sedis = g.vp['incertae_sedis']
        self.hindex = g.vp['hindex']
        self.taxid_vertex = {}
        g.set_vertex_filter(self.vertex_in_taxonomy)
        for v in g.vertices():
            tid = self.vertex_taxid[v]
            self.taxid_vertex[tid] = v
        g.set_vertex_filter(None)

        self.edge_strees = get_or_create_ep(g, 'stree', 'vector<int>')
        self.vertex_snode = get_or_create_vp(g, 'snode', 'int')
        self.vertex_strees = get_or_create_vp(g, 'stree', 'vector<int>')
        self.vertex_stem_cdef = get_or_create_vp(g, 'stem_cdef', 'vector<int>')
        self.stem_cdef_vertex = defaultdict(lambda: g.add_vertex())
        g.set_vertex_filter(self.vertex_in_taxonomy, inverted=True)
        for v in g.vertices():
            cdef = self.vertex_stem_cdef[v]
            if cdef: self.stem_cdef_vertex[tuple(cdef)] = v
        g.set_vertex_filter(None)

        self.root = g.vertex(0)
        return self

    def _filter(self):
        # higher taxa that should be removed (nodes collapsed), and their
        # immmediate children flagged incertae sedis and linked back to
        # the parent of collapsed node
        incertae_keywords = [
            'endophyte','scgc','libraries','samples',
            'metagenome','unclassified',
            'other','unidentified','mitosporic','uncultured','incertae',
            'environmental']

        # taxa that are not clades, and should be removed (collapsed) -
        # children linked to parent of collapsed node
        collapse_keywords = ['basal ','stem ','early diverging ']

        # higher taxa that should be removed along with all of their children
        remove_keywords = ['viroids','virus','viruses','viral','artificial']

        logging.info('removing vertices that are not real taxa (clades)')
        rm = self.collapsed
        g = self.g
        def f(x): rm[x] = 1
        T = Traverser(post=f)
        for v in filter(lambda x:x.out_degree(), g.vertices()):
            name = self.vertex_name[v].lower()
            s = name.split()
            for kw in remove_keywords:
                if kw in s:
                    gt.dfs_search(g, v, T)
                    break

            for kw in incertae_keywords:
                if kw in s:
                    rm[v] = 1
                    for c in v.out_neighbours():
                        self.incertae_sedis[c] = 1
                    break

            s = name.replace('-', ' ')
            for w in collapse_keywords:
                if s.startswith(w):
                    rm[v] = 1
                    break

        g.set_vertex_filter(rm, inverted=True)
        # assume root == vertex 0
        outer = [ v for v in g.vertices()
                  if int(v) and v.in_degree()==0 ]
        g.set_vertex_filter(None)

        for v in outer:
            p = next(v.in_neighbours())
            while rm[p]:
                p = next(p.in_neighbours())
            self.edge_in_taxonomy[g.add_edge(p, v)] = 1
        print('done')

        g.set_vertex_filter(rm, inverted=True)

        for v in g.vertices():
            if int(v): assert v.in_degree()==1
    
    def _index_graph(self):
        '''
        create a vertex property map with hierarchical (left, right)
        indices
        '''
        logging.info('indexing graph (left-right and depth values)')
        g = self.g
        v = g.vertex(0) # root
        hindex = get_or_create_vp(g, 'hindex', 'vector<int>')
        depth = get_or_create_vp(g, 'depth', 'int')
        n = [g.num_vertices()]
        def traverse(p, left, dep):
            depth[p] = dep
            if p.out_degree():
                l0 = left
                for c in p.out_neighbours():
                    l, r = traverse(c, left+1, dep+1)
                    left = r
                lr = (l0, r+1)
            else:
                lr = (left, left+1)
            hindex[p] = lr
            ## print g.vertex_name[p], lr
            print(n[0], '\r', end=' ')
            n[0] -= 1
            return lr
        self.hindex = hindex
        traverse(v, 1, 0)
        logging.info('...done')

    def map_stree(self, root):
        """
        Traverse the tree `root` and assign taxon ids (taxids) to its
        nodes, as well as other information, e.g. whether a leaf is
        'incertae sedis' within a higher taxon, and how nodes conflict
        with the taxonomy.
        """
        print('mapping', root.stree)
        for n in root:
            if n.children: n.taxid = None
            n.taxids = set() # taxa represented by the node
            n.stem_cdef = set()
            n.crown_cdef = set()
            n.vertex = None # vertex in G
            n.taxid_conflicts = set() # taxa in descendants that are not
                                      # monophyletic
            n.taxid_mrcas = set() # taxa whose mrca in the tree maps to the node
            n.leaf_rootpath = {}
            n.nleaves = 0
            n.leaf_is_higher_taxon = False
            n.incertae_sedis = False

        lvs = root.leaves()
        leafcounts = Counter()
        for lf in lvs:
            if not lf.taxid:
                print('!!! [%s] no taxid:' % root.stree, lf.snode_id, lf.label)
                lf.incertae_sedis = True
                lf.taxid_rootpath = []
            elif lf.taxid not in self.taxid_vertex:
                print('!!! [%s] taxid not in taxonomy:' % \
                      root.stree, lf.snode_id, lf.label)
                lf.incertae_sedis = True
                lf.taxid_rootpath = []
            else:
                leafcounts[lf.taxid] += 1
                lf.v = self.taxid_vertex[lf.taxid]
                lf.incertae_sedis = self.incertae_sedis[lf.v]
                lf.taxid_rootpath = taxid_rootpath(self, lf.taxid)

            for n in lf.rootpath(): n.nleaves += 1

        multileaves = [ taxid for taxid, count in list(leafcounts.items())
                        if count > 1 ]
        root_mrca = rootpath_mrca([ x.taxid_rootpath for x in lvs
                                    if x.taxid_rootpath ])
        root.taxid = root_mrca
        print('root is', self.taxid_name(root_mrca))

        for lf in lvs:
            rp = lf.taxid_rootpath
            if rp: break
        i = rp.index(root_mrca) - len(rp) + 1

        for lf in lvs:
            if lf.taxid_rootpath:
                if i: lf.taxid_rootpath = lf.taxid_rootpath[:i]
                if lf.taxid in multileaves:
                    print('!!! multiple taxid: %s at lf %s' % (lf.taxid, lf))
                    lf.incertae_sedis = True

        all_taxids = set()
        for x in lvs: all_taxids.update(x.taxid_rootpath)
        subgraph = self.taxid_subgraph(all_taxids)
        all_taxids.remove(root.taxid)

        for lf in lvs:
            if lf.taxid and not lf.incertae_sedis:
                vtx = subgraph.vertex(int(self.taxid_vertex[lf.taxid]))
                if vtx.out_degree()>0:
                    # higher taxon at leaf
                    print('!!! higher taxon %s at lf %s' % (lf.taxid, lf.label))
                    lf.leaf_is_higher_taxon = True

        def taxid_cmp(t1, t2):
            """
            taxids: if t1 is nested in t2, return t1 before t2
            """
            if t1 == t2: return 0
            v1 = self.taxid_vertex[t1]
            v2 = self.taxid_vertex[t2]
            l1, r1 = self.hindex[v1]
            l2, r2 = self.hindex[v2]
            if (l1 > l2) and (r1 < r2): return -1 # t1 nested in t2
            elif (l1 < l2) and (r1 > r2): return 1 # t2 nested in t1
            else: return 0

        taxid_key = cmp_to_key(taxid_cmp)

        conflicts = {}
        con2mrca = {}
        for taxid in all_taxids:
            '''
            find the mrca node of each taxid in the tree, and determine if
            it is monophyletic, taking into account tips that are incertae
            sedis and/or unmapped
            '''
            nxt, bck = self.taxid_hindex(taxid)
            v = [ lf for lf in lvs if taxid in lf.taxid_rootpath ]
            n = root.mrca(v)
            n.taxid_mrcas.add(taxid)
            ismono = True
            for lf in n.leaves():
                if (lf not in v) and lf.taxid:
                    if lf.taxid in self.taxid_vertex:
                        if not lf.incertae_sedis:
                            if lf.leaf_is_higher_taxon:
                                # is taxid nested within lf.taxid?
                                vnext, vback = self.taxid_hindex(lf.taxid)
                                if not (nxt > vnext and bck < vback): # no
                                    conflicts[taxid] = v
                                    n.taxid_conflicts.add(taxid)
                                    con2mrca[taxid] = n
                                    ismono = False
                                    break
                            else:
                                conflicts[taxid] = v
                                n.taxid_conflicts.add(taxid)
                                con2mrca[taxid] = n
                                ismono = False
                                break

                        else:
                            # is the parent taxon of lf.taxid an ancestor of taxid?
                            try:
                                p = next(lf.v.in_neighbours())
                            except StopIteration: # lf is mapped to root!
                                continue
                            ptax = self.vertex_taxid[p]
                            if taxid_cmp(ptax, taxid) > 0: # taxid is nested in ptax
                                continue
                            conflicts[taxid] = v
                            n.taxid_conflicts.add(taxid)
                            con2mrca[taxid] = n
                            ismono = False
                            break
                    else:
                        lf.taxid = None
                        lf.incertae_sedis = True

            if (ismono and n.children) or len(v)==1: n.taxids.add(taxid)

        for n in root:
            if len(n.taxids)>1:
                n.taxids = sorted(n.taxids, key=taxid_key)
            else:
                n.taxids = list(n.taxids)

        for lf in lvs:
            if lf.taxid in conflicts:
                lf.incertae_sedis = True
                mrca = con2mrca[lf.taxid]
                for n in lf.rootpath():
                    if n == mrca: break
                    n.incertae_sedis = True

        for n in root.postiter():
            if n.taxids:
                n.stem_cdef = (n.taxids[-1],)
                n.crown_cdef = (n.taxids[0],)
            elif n.taxid:
                if n.taxid not in conflicts:
                    n.stem_cdef = (n.taxid,)
                    n.crown_cdef = (n.taxid,)
                else:
                    n.stem_cdef = tuple()
                    n.crown_cdef = tuple()
            elif n.isleaf:
                n.stem_cdef = tuple()
                n.crown_cdef = tuple()
            else:
                v = set(n.taxid_mrcas)
                for c in n.children: v.update(c.stem_cdef)
                cdef = set(v)
                stem_discard = set()
                crown_discard = set()
                while v:
                    t1 = v.pop()
                    for t2 in v:
                        try: i = taxid_cmp(t1, t2)
                        except: continue
                        if i == -1:
                            stem_discard.add(t1)
                            crown_discard.add(t2)
                        elif i == 1:
                            stem_discard.add(t2)
                            crown_discard.add(t1)
                n.stem_cdef = set(cdef)
                for x in stem_discard: n.stem_cdef.remove(x)
                n.crown_cdef = set(cdef)
                for x in crown_discard: n.crown_cdef.remove(x)

        for n in root:
            n.stem_cdef = tuple(sorted(n.stem_cdef))
            n.crown_cdef = tuple(sorted(n.crown_cdef))

        for n in root.postiter():
            if n.parent and n.stem_cdef and n.stem_cdef==n.parent.stem_cdef:
                n.stem_cdef = tuple()

        root.conflicts = conflicts
        root.con2mrca = con2mrca
        return root

    def taxid_subgraph(self, taxids):
        """
        create a new TaxonomyGraph based on a filtered view of self.g
        """
        g = self.g
        vfilt = g.new_vertex_property('bool')
        for x in taxids:
            v = self.taxid_vertex[x]
            vfilt[v] = 1
        gv = gt.GraphView(g, vfilt=vfilt, efilt=self.edge_in_taxonomy)
        gv.vfilt = vfilt
        for v in gv.vertices():
            assert self.vertex_taxid[v] in taxids, self.vertex_taxid[v]
        r = [ x for x in gv.vertices() if x.in_degree()==0 ]
        if len(r)>1:
            for x in r:
                print('!!! root? vertex', int(x), g.vertex_name[x])
        assert len(r)==1

        return self.new_from_graph(gv)

    def taxid_new_subgraph(self, taxids):
        """
        create a new TaxonomyGraph based on a copy of self.g
        """
        newg = gt.Graph()
        g = self.g
        for pname, p in list(g.vp.items()):
            newp = newg.new_vertex_property(p.value_type())
            newg.vp[pname] = newp
        for pname, p in list(g.ep.items()):
            newp = newg.new_edge_property(p.value_type())
            newg.ep[pname] = newp
        ovi2nvi = {}
        for x in taxids:
            v = self.taxid_vertex[x]
            newv = newg.add_vertex()
            ovi2nvi[int(v)] = int(newv)
            for pname, p in list(g.vp.items()):
                newg.vp[pname][newv] = p[v]
        for x in taxids:
            v = self.taxid_vertex[x]
            newv = newg.vertex(ovi2nvi[int(v)])
            for e in v.in_edges():
                if g.ep['istaxon'][e] and int(e.source()) in ovi2nvi:
                    src = newg.vertex(ovi2nvi[int(e.source())])
                    newe = newg.add_edge(src, newv)
                    newg.ep['istaxon'][newe] = 1
                    break
        r = [ x for x in newg.vertices() if x.in_degree()==0 ]
        if len(r)>1:
            for x in r:
                tid = newg.vp['istaxon'][x]
                print('!!! root? vertex', int(x), tid, newg.vertex_name[x])
        #assert len(r)==1, r
        sg = self.new_from_graph(newg)
        sg.root = r[0]
        return sg

    def merge_stree(self, root, stree, verts=None, edges=None):
        G = self.g
        if verts is None: verts = G.new_vertex_property('bool')
        if edges is None: edges = G.new_edge_property('bool')
        for node in root:
            if node.taxids and not node.incertae_sedis:
                # node represents 1+ taxa
                it = iter(node.taxids)
                t = next(it)
                v = self.taxid_vertex[t]
                node.v = v
                verts[v] = 1
                strees = self.vertex_strees[v]
                if stree not in strees: strees.append(stree)
                p = v
                while 1:
                    try: t = next(it)
                    except StopIteration: break
                    v = self.taxid_vertex[t]
                    verts[v] = 1
                    strees = self.vertex_strees[v]
                    if stree not in strees: strees.append(stree)
                    assert v != p
                    e = G.edge(v, p) or G.add_edge(v, p)
                    ## if v == p:
                    ##     print '!! merge_stree', node.snode_id, v
                    ##     loops.append((root, e))
                    edges[e] = 1
                    strees = self.edge_strees[e]
                    if stree not in strees: strees.append(stree)
                    ## G.edge_stree[e] = stree
                    p = v

            elif node.taxid and not node.incertae_sedis:
                v = self.taxid_vertex[node.taxid]
                node.v = v
                verts[v] = 1
                strees = self.vertex_strees[v]
                if stree not in strees: strees.append(stree)

            elif node.stem_cdef and not node.incertae_sedis: # node is a clade
                v = self.stem_cdef_vertex[node.stem_cdef]
                node.v = v
                verts[v] = 1
                self.vertex_stem_cdef[v] = node.stem_cdef
                strees = self.vertex_strees[v]
                if stree not in strees: strees.append(stree)

            else:
                v = G.add_vertex()
                node.v = v
                verts[v] = 1
                strees = self.vertex_strees[v]
                if stree not in strees: strees.append(stree)

            if node.parent:
                v = node.v
                if node.taxids and not node.incertae_sedis:
                    v = self.taxid_vertex[node.taxids[-1]]
                p = node.parent
                e = G.edge(p.v, v) or G.add_edge(p.v, v)
                if p.v == v:
                    print('!! LOOP:', node.snode_id, v)
                    ## loops.append((node, e))
                edges[e] = 1
                strees = self.edge_strees[e]
                if stree not in strees: strees.append(stree)

        for lf in root.leaves():
            if lf.incertae_sedis and lf.v:
                self.vertex_name[lf.v] = lf.label

        return verts, edges

    def view(self, vfilt=None, efilt=None):
        g = self.g
        view = self.new_from_graph(gt.GraphView(g, vfilt=vfilt, efilt=efilt))
        r = [ x for x in view.vertices() if x.in_degree()==0 ]
        ## assert len(r)==1
        assert r
        if len(r) > 1: print('!!! disconnected view')
        view.root = r[0]
        return view

class Traverser(gt.DFSVisitor):
    def __init__(self, pre=None, post=None):
        # function to call on each vertex, preorder
        if not pre: pre = lambda x:None
        self.pre = pre
        if not post: post = lambda x:None
        self.post = post # postorder

    def discover_vertex(self, v):
        self.pre(v)

    def finish_vertex(self, v):
        self.post(v)

def taxid_cmp(g, t1, t2):
    """
    taxids: if t1 is nested in t2, return t1 before t2
    """
    if t1 == t2: return 0
    v1 = g.taxid_vertex[t1]
    v2 = g.taxid_vertex[t2]
    l1, r1 = g.hindex[v1]
    l2, r2 = g.hindex[v2]
    if (l1 > l2) and (r1 < r2): return -1 # t1 nested in t2
    elif (l1 < l2) and (r1 > r2): return 1 # t2 nested in t1
    else: return 0

taxid_key = cmp_to_key(taxid_cmp)

def index_graph(g, reindex=False):
    '''
    create a vertex property map with hierarchical (left, right)
    indices
    '''
    logging.info('indexing graph (left-right and depth values)')
    if 'hindex' in g.vp and 'depth' in g.vp and not reindex:
        return g
    v = g.vertex(0) # root
    hindex = get_or_create_vp(g, 'hindex', 'vector<int>')
    depth = get_or_create_vp(g, 'depth', 'int')
    n = [g.num_vertices()]
    def traverse(p, left, dep):
        depth[p] = dep
        if p.out_degree():
            l0 = left
            for c in p.out_neighbours():
                l, r = traverse(c, left+1, dep+1)
                left = r
            lr = (l0, r+1)
        else:
            lr = (left, left+1)
        hindex[p] = lr
        ## print g.vertex_name[p], lr
        print(n[0], '\r', end=' ')
        n[0] -= 1
        return lr
    g.hindex = hindex
    traverse(v, 1, 0)
    logging.info('...done')

def create_ncbi_taxonomy_graph(basepath='ncbi'):
    '''
    create a graph containing the NCBI taxonomic hierarchy using files from:

    ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz

    which are assumed to be unpacked in the directory ``basepath``
    '''
    node_fields = ["taxid", "parent_taxid", "rank", "embl_code", "division_id",
                   "inherited_div_flag", "genetic_code_id", "inherited_gc_flag",
                   "mt_gc_id", "inherited_mgc_flag", "genbank_hidden_flag",
                   "hidden_subtree_root_flag", "comments"]
    name_fields = ["taxid", "name", "unique_name", "name_class"]
    def process_nodes(f):
        logging.info('...processing nodes')
        n = {}
        c = defaultdict(list)
        i = 0
        for line in f:
            v = [ x.strip() or None for x in line.split("|") ]
            s = dict(list(zip(node_fields, v[:-1])))
            for k, v in list(s.items()):
                if k.endswith('_flag'):
                    s[k] = bool(int(v))
                else:
                    try: s[k] = int(v)
                    except: pass
            tid = s['taxid']
            n[tid] = s
            if tid > 1: c[s['parent_taxid']].append(tid)
            i += 1
            print('%s           \r' % i, end=' ')
        print()
        return n, c

    def process_names(f):
        logging.info('...processing names')
        seen = set()
        synonyms = defaultdict(list)
        accepted = {}
        i = 0
        for line in f:
            v = [ x.strip() or None for x in line.split("|") ]
            v[0] = int(v[0])
            s = dict(list(zip(name_fields, v[:-1])))
            name = s['name']; uname = s['unique_name']; taxid = s['taxid']
            s['type'] = 'taxonomic_name'
            s['source'] = 'ncbi'
            if uname or name in seen: s['homonym_flag'] = True
            if s['name_class'] == 'scientific name' and (uname or name) not in seen:
                s['primary'] = True
                accepted[taxid] = s
            else:
                s['primary'] = False
                synonyms[taxid].append(s)
            seen.add(uname or name)
            i += 1
            print('%s           \r' % i, end=' ')
        print()
        return accepted, synonyms

    with open(os.path.join(basepath, 'nodes.dmp')) as f:
        nodes, ptid2ctid = process_nodes(f)
    with open(os.path.join(basepath, 'names.dmp')) as f:
        accepted, synonyms = process_names(f)

    G = gt.Graph()
    G.vertex_name = get_or_create_vp(G, 'name', 'string')
    G.vertex_rank = get_or_create_vp(G, 'rank', 'string')
    G.vertex_taxid = get_or_create_vp(G, 'taxid', 'int')
    G.edge_in_taxonomy = get_or_create_ep(G, 'istaxon', 'bool')
    G.vertex_in_taxonomy = get_or_create_vp(G, 'istaxon', 'bool')
    G.dubious = get_or_create_vp(G, 'dubious', 'bool')
    G.incertae_sedis = get_or_create_vp(G, 'incertae_sedis', 'bool')
    G.collapsed = get_or_create_vp(G, 'collapsed', 'bool')
    G.taxid_vertex = {}

    nnodes = len(nodes)
    viter = G.add_vertex(nnodes)

    logging.info('...creating graph vertices')

    i = 0
    for tid, d in nodes.items():
        v = G.vertex(i)
        G.vertex_in_taxonomy[v] = 1
        G.taxid_vertex[tid] = v
        G.vertex_taxid[v] = tid
        G.vertex_rank[v] = d['rank']
        try:
            name = accepted[tid]['name'] # !! should deal with unique_name
            G.vertex_name[v] = name
        except KeyError:
            print(tid)
        i += 1
        print('%s           \r' % (nnodes-i), end=' ')
    print()

    logging.info('...creating graph vertices')
    i = 0; n = len(ptid2ctid)
    for tid, child_tids in ptid2ctid.items():
        pv = G.taxid_vertex[tid]
        for ctid in child_tids:
            cv = G.taxid_vertex[ctid]
            e = G.add_edge(pv, cv)
            G.edge_in_taxonomy[e] = 1
        i += 1
        print('%s           \r' % (n-i), end=' ')
    print()

    G.edge_strees = get_or_create_ep(G, 'stree', 'vector<int>')
    G.vertex_snode = get_or_create_vp(G, 'snode', 'int')
    G.vertex_strees = get_or_create_vp(G, 'stree', 'vector<int>')
    G.vertex_stem_cdef = get_or_create_vp(G, 'stem_cdef', 'vector<int>')
    G.stem_cdef_vertex = defaultdict(lambda: G.add_vertex())

    _filter(G)
    index_graph(G)
    _attach_funcs(G)

    G.root = G.vertex(0)

    return G

## def _create_ott_taxonomy_graph(version='2.1'):
##     g = gt.Graph()
##     g.vertex_name = get_or_create_vp(g, 'name', 'string')
##     g.vertex_taxid = get_or_create_vp(g, 'taxid', 'int')
##     g.edge_in_taxonomy = get_or_create_ep(g, 'istaxon', 'bool')
##     g.vertex_in_taxonomy = get_or_create_vp(g, 'istaxon', 'bool')
##     g.dubious = get_or_create_vp(g, 'dubious', 'bool')
##     g.incertae_sedis = get_or_create_vp(g, 'incertae_sedis', 'bool')
##     g.collapsed = get_or_create_vp(g, 'collapsed', 'bool')
##     g.taxid_vertex = {}

##     #N = 2644684
##     taxid2vid = {}
##     data = []
##     n = 0
##     split = lambda s: (
##         [ x.strip() or None for x in s.split('|')][:-1] if s[-2]=='\t'
##         else [ x.strip() or None for x in s.split('|')]
##     )
##     with open('ott-%s/taxonomy' % version) as f:
##         f.readline()
##         for v in map(split, f):
##             for i in 0,1: v[i] = int(v[i] or 0)
##             taxid = v[0]
##             taxid2vid[taxid] = n
##             data.append(v)
##             print n, '\r',
##             n += 1
##         print 'done'

##     g.add_vertex(n)
##     for i, row in enumerate(data):
##         taxid = row[0]
##         parent = row[1]
##         name = row[2]
##         v = g.vertex(i)
##         g.vertex_in_taxonomy[v] = 1
##         g.vertex_taxid[v] = taxid
##         g.vertex_name[v] = name
##         g.taxid_vertex[taxid] = v

##         if row[-1] and 'D' in row[-1]: g.dubious[v] = 1

##         if parent:
##             pv = g.vertex(taxid2vid[parent])
##             e = g.add_edge(pv, v)
##             g.edge_in_taxonomy[e] = 1
##         print i, '\r',
##     print 'done'

##     _filter(g)
##     index_graph(g)
##     _attach_funcs(g)

##     g.root = g.vertex(0)
##     return g

def graph_json(g, dist=None, pos=None, ecolor=None, ewidth=None,
               vcolor=None, vsize=None, vtext=None, fp=None):
    import simplejson
    nodes = []
    links = []
    idx = {}

    if pos:
        xmin = min([ pos[x][0] for x in g.vertices() ])
        ymin = min([ pos[x][1] for x in g.vertices() ])
        for x in g.vertices(): pos[x] = [pos[x][0]-xmin, pos[x][1]-ymin]

    for i,v in enumerate(g.vertices()):
        idx[int(v)] = i
        taxid = g.vertex_taxid[v]
        name = g.taxid_name(taxid) if taxid else 'node%s' % int(v)
        isleaf = v.out_degree()==0
        d = dict(label=name, isleaf=isleaf, strees=list(g.vertex_strees[v]))
        if taxid: d['taxid'] = taxid
        if dist: d['dist'] = dist[v]
        if pos and pos[v]:
            x, y = pos[v]
            d['x'] = x; d['y'] = y
            d['fixed'] = True
        if vcolor: d['color'] = vcolor[v]
        if vsize: d['size'] = vsize[v]
        if vtext: d['label'] = vtext[v]
        d['altlabel'] = g.vertex_name[v]
        nodes.append(d)
    for e in g.edges():
        source = idx[int(e.source())]
        target = idx[int(e.target())]
        strees = g.edge_strees[e]
        d = dict(source=source, target=target, strees = list(strees),
                 taxedge=bool(g.edge_in_taxonomy[e]))
        if ecolor: d['color'] = ecolor[e]
        if ewidth: d['width'] = ewidth[e]
        links.append(d)
    if fp:
        simplejson.dump(dict(nodes=nodes, links=links), fp)
    else:
        return simplejson.dumps(dict(nodes=nodes, links=links))


def graph_view(g, vfilt=None, efilt=None):
    view = gt.GraphView(g, vfilt=vfilt, efilt=efilt)
    view.vertex_name = g.vertex_name
    view.vertex_taxid = g.vertex_taxid
    view.edge_in_taxonomy = g.edge_in_taxonomy
    view.vertex_in_taxonomy = g.vertex_in_taxonomy
    view.dubious = g.dubious
    view.incertae_sedis = g.incertae_sedis
    view.taxid_vertex = g.taxid_vertex
    view.hindex = g.hindex
    view.edge_strees = g.edge_strees
    view.vertex_snode = g.vertex_snode
    view.vertex_strees = g.vertex_strees
    view.vertex_stem_cdef = g.vertex_stem_cdef
    view.stem_cdef_vertex = g.stem_cdef_vertex
    _attach_funcs(view)
    r = [ x for x in view.vertices() if x.in_degree()==0 ]
    ## assert len(r)==1
    assert r
    if len(r) > 1: print('!!! disconnected view')
    view.root = r[0]
    return view

def _attach_funcs(g):
    def taxid_name(taxid):
        v = g.taxid_vertex.get(taxid)
        if v: return g.vertex_name[v]
    g.taxid_name = taxid_name

    def taxid_dubious(taxid):
        v = g.taxid_vertex.get(taxid)
        if v: return g.dubious[v]
    g.taxid_dubious = taxid_dubious

    def taxid_hindex(taxid):
        v = g.taxid_vertex.get(taxid)
        if v: return g.hindex[v]
        else: print('no hindex:', taxid)
    g.taxid_hindex = taxid_hindex


def load_taxonomy_graph(source):
    g = gt.load_graph(source)
    g.vertex_name = g.vp['name']
    g.vertex_taxid = g.vp['taxid']
    g.edge_in_taxonomy = g.ep['istaxon']
    g.vertex_in_taxonomy = g.vp['istaxon']
    g.dubious = g.vp['dubious']
    g.incertae_sedis = g.vp['incertae_sedis']
    g.hindex = g.vp['hindex']
    g.taxid_vertex = {}
    g.set_vertex_filter(g.vertex_in_taxonomy)
    for v in g.vertices():
        tid = g.vertex_taxid[v]
        g.taxid_vertex[tid] = v
    g.set_vertex_filter(None)

    g.edge_strees = get_or_create_ep(g, 'stree', 'vector<int>')
    g.vertex_snode = get_or_create_vp(g, 'snode', 'int')
    g.vertex_strees = get_or_create_vp(g, 'stree', 'vector<int>')
    g.vertex_stem_cdef = get_or_create_vp(g, 'stem_cdef', 'vector<int>')
    g.stem_cdef_vertex = defaultdict(lambda: g.add_vertex())
    g.set_vertex_filter(g.vertex_in_taxonomy, inverted=True)
    for v in g.vertices():
        cdef = g.vertex_stem_cdef[v]
        if cdef: g.stem_cdef_vertex[tuple(cdef)] = v
    g.set_vertex_filter(None)

    _attach_funcs(g)

    g.root = g.vertex(0)
    ## r = [ x for x in g.vertices() if x.in_degree()==0 ]
    ## assert len(r)==1
    ## g.root = r[0]
    ## p = g.vp.get('collapsed')
    ## if p and sum(p.a):
    ##     g = graph_view(g, vfilt=p)
    return g

def taxid_subgraph(g, taxids):
    vfilt = g.new_vertex_property('bool')
    for x in taxids:
        v = g.taxid_vertex[x]
        vfilt[v] = 1
    sg = gt.GraphView(g, vfilt=vfilt, efilt=g.edge_in_taxonomy)
    sg.vfilt = vfilt
    for v in sg.vertices():
        assert g.vertex_taxid[v] in taxids, g.vertex_taxid[v]
    r = [ x for x in sg.vertices() if x.in_degree()==0 ]
    if len(r)>1:
        for x in r:
            print('!!! root? vertex', int(x), g.vertex_name[x])
    assert len(r)==1
    sg.root = r[0]
    sg.taxid_vertex = {}
    for v in sg.vertices():
        taxid = g.vertex_taxid[v]
        sg.taxid_vertex[v] = taxid
    _attach_funcs(sg)
    return sg

def taxid_new_subgraph(g, taxids):
    newg = gt.Graph()
    for pname, p in list(g.vp.items()):
        newp = newg.new_vertex_property(p.value_type())
        newg.vp[pname] = newp
    for pname, p in list(g.ep.items()):
        newp = newg.new_edge_property(p.value_type())
        newg.ep[pname] = newp
    newg.vertex_name = newg.vp['name']
    newg.vertex_taxid = newg.vp['taxid']
    newg.edge_in_taxonomy = newg.ep['istaxon']
    newg.vertex_in_taxonomy = newg.vp['istaxon']
    newg.dubious = newg.vp['dubious']
    newg.incertae_sedis = newg.vp['incertae_sedis']
    newg.hindex = newg.vp['hindex']
    newg.edge_strees = newg.ep['stree']
    newg.vertex_snode = newg.vp['snode']
    newg.vertex_strees = newg.vp['stree']
    newg.vertex_stem_cdef = newg.vp['stem_cdef']
    newg.stem_cdef_vertex = defaultdict(lambda: newg.add_vertex())
    newg.taxid_vertex = {}
    ovi2nvi = {}
    for x in taxids:
        v = g.taxid_vertex[x]
        newv = newg.add_vertex()
        ovi2nvi[int(v)] = int(newv)
        for pname, p in list(g.vp.items()):
            newg.vp[pname][newv] = p[v]
        newg.taxid_vertex[x] = newv
    for x in taxids:
        v = g.taxid_vertex[x]
        newv = newg.vertex(ovi2nvi[int(v)])
        for e in v.in_edges():
            if g.ep['istaxon'][e] and int(e.source()) in ovi2nvi:
                src = newg.vertex(ovi2nvi[int(e.source())])
                newe = newg.add_edge(src, newv)
                newg.ep['istaxon'][newe] = 1
                break
    r = [ x for x in newg.vertices() if x.in_degree()==0 ]
    if len(r)>1:
        for x in r:
            tid = newg.vertex_taxid[x]
            print('!!! root? vertex', int(x), tid, newg.vertex_name[x])
    #assert len(r)==1, r
    newg.root = r[0]
    _attach_funcs(newg)
    return newg

def taxonomy_subtree(G, v):
    g = gt.Graph()
    g.vertex_in_taxonomy = get_or_create_vp(g, 'istaxon', 'bool')
    g.edge_in_taxonomy = get_or_create_ep(g, 'istaxon', 'bool')
    g.vertex_taxid = get_or_create_vp(g, 'taxid', 'int')
    g.vertex_name = get_or_create_vp(g, 'name', 'string')
    g.dubious = get_or_create_vp(g, 'dubious','bool')
    g.incertae_sedis = get_or_create_vp(g, 'incertae_sedis','bool')
    g.taxid_vertex = {}

    GP = [ x for x in list(G.vp.items()) if x[0] in g.vp ]
    def copy_vertex_properties(Gv, gv):
        for pname, pmap in GP:
            g.vp[pname][gv] = pmap[Gv]

        taxid = G.vertex_taxid[Gv]
        g.taxid_vertex[taxid] = gv

    def traverse(x, p):
        nx = g.add_vertex()
        copy_vertex_properties(x, nx)
        g.edge_in_taxonomy[g.add_edge(p, nx)] = 1
        for c in x.out_neighbours():
            traverse(c, nx)

    p = g.add_vertex()
    copy_vertex_properties(v, p)
    for c in v.out_neighbours():
        traverse(c, p)

    index_graph(g)
    _attach_funcs(g)

    g.root = g.vertex(0)
    return g
    
def graph2sqlite(g, fname):
    import sqlite3 as db
    from os import path
    create = not path.isfile(fname)
    con = db.connect(fname)
    cur = con.cursor()

    if create:
        cur.execute('create table if not exists name ( '
                    'uid integer primary key, '
                    'parent_uid integer, '
                    'name text not null, '
                    'depth integer not null, '
                    'rank text, '
                    'next integer not null, '
                    'back integer not null '
                    ')')

    vertex_taxid = g.vp['taxid']
    vertex_name = g.vp['name']
    vertex_rank = g.vp['rank']
    vertex_depth = g.vp['depth']
    hindex = g.vp['hindex']
    for v in g.vertices():
        taxid = vertex_taxid[v]
        if int(v)>0:
            parent_taxid = vertex_taxid[next(v.in_neighbours())]
        else:
            parent_taxid = None
        name = vertex_name[v].decode('utf8')
        rank = vertex_rank[v]
        depth = vertex_depth[v]
        nxt, bck = g.hindex[v]
        data = (taxid, parent_taxid, name, nxt, bck)
        cur.execute('insert into name '
                    '(uid, parent_uid, name, depth, rank, next, back) '
                    'values (?, ?, ?, ?, ?, ?, ?);', data)
        print(int(v), '\r', end=' ')
    print('done')
    
    if create:
        cur.execute('create index nextback on name (next, back)')
        cur.execute('create index parent on name (parent_uid)')
        cur.execute('create index nameidx on name (name)')

    con.commit()
    cur.close()
    con.close()

def taxid_rootpath(G, taxid, stop=None):
    v = G.taxid_vertex[taxid]
    x = [taxid]
    while v.in_degree():
        p = next(v.in_neighbours())
        tid = G.vertex_taxid[p]
        x.append(tid)
        v = p
        if stop and tid == stop: break
    return x

def rootpath_mrca(rootpaths, i=-1):
    s = set([ x[i] for x in rootpaths ])
    assert len(s)==1, s
    while len(s)==1:
        p = s
        i -= 1
        try: s = set([ x[i] for x in rootpaths ])
        except IndexError: break
    return p.pop()

def map_stree(G, root):
    print('mapping', root.stree)
    ## root = fetch_stree(stree_id, cache=cache,
    ##                    prune_to_ingroup=prune_to_ingroup)
    ## print '  tree built'
    for n in root:
        if n.children: n.taxid = None
        n.taxids = set() # taxa represented by the node
        n.stem_cdef = set()
        n.crown_cdef = set()
        n.vertex = None # vertex in G
        n.taxid_conflicts = set() # taxa in descendants that are not
                                  # monophyletic
        n.taxid_mrcas = set() # taxa whose mrca in the tree maps to the node
        n.leaf_rootpath = {}
        n.nleaves = 0
        n.leaf_is_higher_taxon = False
        n.incertae_sedis = False

    lvs = root.leaves()
    leafcounts = Counter()
    for lf in lvs:
        if not lf.taxid:
            print('!!! [%s] no taxid:' % root.stree, lf.snode_id, lf.label)
            lf.incertae_sedis = True
            lf.taxid_rootpath = []
        elif lf.taxid not in G.taxid_vertex:
            print('!!! [%s] taxid not in taxonomy:' % \
                  root.stree, lf.snode_id, lf.label)
            lf.incertae_sedis = True
            lf.taxid_rootpath = []
        else:
            leafcounts[lf.taxid] += 1
            lf.v = G.taxid_vertex[lf.taxid]
            lf.incertae_sedis = G.incertae_sedis[lf.v]
            lf.taxid_rootpath = taxid_rootpath(G, lf.taxid)

        for n in lf.rootpath(): n.nleaves += 1

    multileaves = [ taxid for taxid, count in list(leafcounts.items()) if count > 1 ]
    root_mrca = rootpath_mrca([ x.taxid_rootpath for x in lvs
                                if x.taxid_rootpath ])
    root.taxid = root_mrca
    print('root is', G.taxid_name(root_mrca))
    
    for lf in lvs:
        rp = lf.taxid_rootpath
        if rp: break
    i = rp.index(root_mrca) - len(rp) + 1

    for lf in lvs:
        if lf.taxid_rootpath:
            if i: lf.taxid_rootpath = lf.taxid_rootpath[:i]
            if lf.taxid in multileaves:
                print('!!! multiple taxid: %s at lf %s' % (lf.taxid, lf))
                lf.incertae_sedis = True
                
    all_taxids = set()
    for x in lvs: all_taxids.update(x.taxid_rootpath)
    subgraph = taxid_subgraph(G, all_taxids)
    all_taxids.remove(root.taxid)

    for lf in lvs:
        if lf.taxid and not lf.incertae_sedis:
            vtx = subgraph.vertex(int(G.taxid_vertex[lf.taxid]))
            if vtx.out_degree()>0:
                # higher taxon at leaf
                print('!!! higher taxon %s at lf %s' % (lf.taxid, lf.label))
                lf.leaf_is_higher_taxon = True

    def taxid_cmp(t1, t2):
        """
        taxids: if t1 is nested in t2, return t1 before t2
        """
        if t1 == t2: return 0
        v1 = G.taxid_vertex[t1]
        v2 = G.taxid_vertex[t2]
        l1, r1 = G.hindex[v1]
        l2, r2 = G.hindex[v2]
        if (l1 > l2) and (r1 < r2): return -1 # t1 nested in t2
        elif (l1 < l2) and (r1 > r2): return 1 # t2 nested in t1
        else: return 0

    taxid_key = cmp_to_key(taxid_cmp)
        
    conflicts = {}
    con2mrca = {}
    for taxid in all_taxids:
        '''
        find the mrca node of each taxid in the tree, and determine if
        it is monophyletic, taking into account tips that are incertae
        sedis and/or unmapped
        '''
        nxt, bck = G.taxid_hindex(taxid)
        v = [ lf for lf in lvs if taxid in lf.taxid_rootpath ]
        n = root.mrca(v)
        n.taxid_mrcas.add(taxid)
        ismono = True
        for lf in n.leaves():
            if (lf not in v) and lf.taxid:
                if lf.taxid in G.taxid_vertex:
                    if not lf.incertae_sedis:
                        if lf.leaf_is_higher_taxon:
                            # is taxid nested within lf.taxid?
                            vnext, vback = G.taxid_hindex(lf.taxid)
                            if not (nxt > vnext and bck < vback): # no
                                conflicts[taxid] = v
                                n.taxid_conflicts.add(taxid)
                                con2mrca[taxid] = n
                                ismono = False
                                break
                        else:
                            conflicts[taxid] = v
                            n.taxid_conflicts.add(taxid)
                            con2mrca[taxid] = n
                            ismono = False
                            break

                    else:
                        # is the parent taxon of lf.taxid an ancestor of taxid?
                        try:
                            p = next(lf.v.in_neighbours())
                        except StopIteration: # lf is mapped to root!
                            continue
                        ptax = G.vertex_taxid[p]
                        if taxid_cmp(ptax, taxid) > 0: # taxid is nested in ptax
                            continue
                        conflicts[taxid] = v
                        n.taxid_conflicts.add(taxid)
                        con2mrca[taxid] = n
                        ismono = False
                        break
                else:
                    lf.taxid = None
                    lf.incertae_sedis = True
                    
        if (ismono and n.children) or len(v)==1: n.taxids.add(taxid)

    for n in root:
        if len(n.taxids)>1:
            n.taxids = sorted(n.taxids, key=taxid_key)
        else:
            n.taxids = list(n.taxids)

    for lf in lvs:
        if lf.taxid in conflicts:
            lf.incertae_sedis = True
            mrca = con2mrca[lf.taxid]
            for n in lf.rootpath():
                if n == mrca: break
                n.incertae_sedis = True

    for n in root.postiter():
        if n.taxids:
            n.stem_cdef = (n.taxids[-1],)
            n.crown_cdef = (n.taxids[0],)
        elif n.taxid:
            if n.taxid not in conflicts:
                n.stem_cdef = (n.taxid,)
                n.crown_cdef = (n.taxid,)
            else:
                n.stem_cdef = tuple()
                n.crown_cdef = tuple()
        elif n.isleaf:
            n.stem_cdef = tuple()
            n.crown_cdef = tuple()
        else:
            v = set(n.taxid_mrcas)
            for c in n.children: v.update(c.stem_cdef)
            cdef = set(v)
            stem_discard = set()
            crown_discard = set()
            while v:
                t1 = v.pop()
                for t2 in v:
                    try: i = taxid_cmp(t1, t2)
                    except: continue
                    if i == -1:
                        stem_discard.add(t1)
                        crown_discard.add(t2)
                    elif i == 1:
                        stem_discard.add(t2)
                        crown_discard.add(t1)
            n.stem_cdef = set(cdef)
            for x in stem_discard: n.stem_cdef.remove(x)
            n.crown_cdef = set(cdef)
            for x in crown_discard: n.crown_cdef.remove(x)

    for n in root:
        n.stem_cdef = tuple(sorted(n.stem_cdef))
        n.crown_cdef = tuple(sorted(n.crown_cdef))

    for n in root.postiter():
        if n.parent and n.stem_cdef and n.stem_cdef==n.parent.stem_cdef:
            n.stem_cdef = tuple()

    root.conflicts = conflicts
    root.con2mrca = con2mrca
    return root


def merge_stree(G, root, stree, verts=None, edges=None):
    if verts is None: verts = G.new_vertex_property('bool')
    if edges is None: edges = G.new_edge_property('bool')
    ## G.edge_strees = get_or_create_ep(G, 'stree', 'vector<int>')
    ## G.vertex_snode = get_or_create_ep(G, 'snode', 'int')
    ## G.vertex_strees = get_or_create_vp(G, 'stree', 'vector<int>')
    ## G.vertex_stem_cdef = get_or_create_vp(G, 'stem_cdef', 'vector<int>')
    ## G.stem_cdef_vertex = defaultdict(lambda: G.add_vertex())
    ## stree = root.rec.tree
    for node in root:
        if node.taxids and not node.incertae_sedis: # node represents 1+ taxa
            it = iter(node.taxids)
            t = next(it)
            v = G.taxid_vertex[t]
            node.v = v
            verts[v] = 1
            strees = G.vertex_strees[v]
            if stree not in strees: strees.append(stree)
            p = v
            while 1:
                try: t = next(it)
                except StopIteration: break
                v = G.taxid_vertex[t]
                verts[v] = 1
                strees = G.vertex_strees[v]
                if stree not in strees: strees.append(stree)
                assert v != p
                e = G.edge(v, p) or G.add_edge(v, p)
                ## if v == p:
                ##     print '!! merge_stree', node.snode_id, v
                ##     loops.append((root, e))
                edges[e] = 1
                strees = G.edge_strees[e]
                if stree not in strees: strees.append(stree)
                ## G.edge_stree[e] = stree
                p = v

        elif node.taxid and not node.incertae_sedis:
            v = G.taxid_vertex[node.taxid]
            node.v = v
            verts[v] = 1
            strees = G.vertex_strees[v]
            if stree not in strees: strees.append(stree)

        elif node.stem_cdef and not node.incertae_sedis: # node is a clade
            v = G.stem_cdef_vertex[node.stem_cdef]
            node.v = v
            verts[v] = 1
            G.vertex_stem_cdef[v] = node.stem_cdef
            strees = G.vertex_strees[v]
            if stree not in strees: strees.append(stree)

        else:
            v = G.add_vertex()
            node.v = v
            verts[v] = 1
            strees = G.vertex_strees[v]
            if stree not in strees: strees.append(stree)

        if node.parent:
            v = node.v
            if node.taxids and not node.incertae_sedis:
                v = G.taxid_vertex[node.taxids[-1]]
            p = node.parent
            e = G.edge(p.v, v) or G.add_edge(p.v, v)
            if p.v == v:
                print('!! LOOP:', node.snode_id, v)
                ## loops.append((node, e))
            edges[e] = 1
            strees = G.edge_strees[e]
            if stree not in strees: strees.append(stree)

    for lf in root.leaves():
        if lf.incertae_sedis and lf.v:
            G.vertex_name[lf.v] = lf.label

    return verts, edges


def _filter(g):
    # higher taxa that should be removed (nodes collapsed), and their
    # immmediate children flagged incertae sedis and linked back to
    # the parent of collapsed node
    incertae_keywords = [
        'endophyte','scgc','libraries','samples','metagenome','unclassified',
        'other','unidentified','mitosporic','uncultured','incertae',
        'environmental']

    # taxa that are not clades, and should be removed (collapsed) -
    # children linked to parent of collapsed node
    collapse_keywords = ['basal ','stem ','early diverging ']

    # higher taxa that should be removed along with all of their children
    remove_keywords = ['viroids','virus','viruses','viral','artificial']

    logging.info('removing vertices that are not real taxa (clades)')
    rm = g.collapsed
    def f(x): rm[x] = 1
    T = Traverser(post=f)
    for v in filter(lambda x:x.out_degree(), g.vertices()):
        name = g.vertex_name[v].lower()
        s = name.split()
        for kw in remove_keywords:
            if kw in s:
                gt.dfs_search(g, v, T)
                break

        for kw in incertae_keywords:
            if kw in s:
                rm[v] = 1
                for c in v.out_neighbours():
                    g.incertae_sedis[c] = 1
                break

        s = name.replace('-', ' ')
        for w in collapse_keywords:
            if s.startswith(w):
                rm[v] = 1
                break

    g.set_vertex_filter(rm, inverted=True)
    # assume root == vertex 0
    outer = [ v for v in g.vertices()
              if int(v) and v.in_degree()==0 ]
    g.set_vertex_filter(None)

    for v in outer:
        p = next(v.in_neighbours())
        while rm[p]:
            p = next(p.in_neighbours())
        g.edge_in_taxonomy[g.add_edge(p, v)] = 1
    print('done')

    g.set_vertex_filter(rm, inverted=True)

    for v in g.vertices():
        if int(v): assert v.in_degree()==1
    
def layout(G, g, rootv, sfdp=True, deg0=-45.0, degspan=90.0, radius=100):
    isouter = lambda x: not bool([ v for v in x.out_neighbours() if v != x ])
    ## isouter = lambda x: x.out_degree()==0
    outer_vertices = [ int(x) for x in g.vertices() if isouter(x) ]
    nouter = len(outer_vertices)
    angle = [float(deg0)]
    unit = float(degspan)/(nouter-1)
    iv2angle = {}
    outer_seen = set()
    iv2dist = defaultdict(lambda:0)
    all_seen = set()

    pos = g.new_vertex_property('vector<float>')
    pin = g.new_vertex_property('bool')
    pin[rootv] = 1
    for v in g.vertices():
        pos[v] = [0.0, 0.0]

    radius = float(radius)

    wt = g.new_edge_property('float')
    for e in g.edges():
        strees = G.edge_strees[e]
        if len(strees) > 0: wt[e] = 1.0
        else: wt[e] = 0.01
    for iv in outer_vertices:
        pv, pe = gt.shortest_path(g.g, rootv, g.vertex(iv))
        for e in reversed(pe):
            wt[e] += 0.5
    g.wt = wt

    def traverse(v, dist=0, angle=angle):
        iv = int(v)
        if iv in outer_vertices:
            iv2dist[iv] = max(iv2dist[iv], dist)
            if iv not in outer_seen:
                ang = angle[0]
                iv2angle[iv] = ang
                angle[0] += unit
                rad = math.radians(ang)
                x = math.cos(rad) * radius
                y = math.sin(rad) * radius
                pos[v] = [x, y]
                pin[v] = 1
                outer_seen.add(iv)
        all_seen.add(iv)
        for oe in v.out_edges():
            ov = oe.target()
            ## if len(G.edge_strees[oe])>0 and (int(ov) not in all_seen):
            if int(ov) not in all_seen:
                traverse(ov, dist+1, angle)
    traverse(rootv)

    if sfdp:
        pos = gt.sfdp_layout(g.g, pos=pos, pin=pin, C=10, p=3,# theta=2,
                             K=0.1,
                             eweight=wt,
                             mu=0.0, multilevel=False)

    return pos, pin


def articulation_bfs(g, root, arts):
    isouter = lambda x: not [ v for v in x.out_neighbours() if v != x ]
    seen = set()
    seen.add(int(root))
    leaves = []
    def traverse(v):
        for n in v.out_neighbours():
            i = int(n)
            if n == v:
                seen.add(i)
                continue
            if isouter(n) or (arts[n] and n.out_degree()>1
                              and len(g.vertex_strees[n])>1):
                if i not in leaves: leaves.append(i)
                seen.add(i)
            else:
                if i not in seen: traverse(n)
                seen.add(i)
    traverse(root)
    return [ g.vertex(i) for i in leaves ]

def bicomp_tree(g, rootv, arts, traversed):
    "generate tree of biconnected components of DAG"
    r = tree.Node()
    leaf_verts = articulation_bfs(g, rootv, arts)
    verts = [ x for x in leaf_verts if x not in traversed and x != rootv ]
    traversed.update(verts)
    r.v = rootv
    r.leaf_verts = leaf_verts
    if not leaf_verts: r.isleaf = True
    print(list(map(int, leaf_verts)))
    for v in verts:
        r.add_child(bicomp_tree2(g, v, arts, traversed))
    return r
