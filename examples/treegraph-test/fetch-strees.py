"""
download trees from www.phylografter.net in Newick format
"""
import ivy, requests

#taxid_field = 'ncbi_taxid' # NCBI taxon id
taxid_field = 'id' # OTT taxon id

def fetch_stree(stree_id):
    print 'fetching stree', stree_id, '...',
    u = 'http://reelab.net/phylografter/stree/newick.txt/%s' % stree_id
    lfmt='snode.id,ott_node.{},otu.label'.format(taxid_field)
    p = dict(lfmt=lfmt, ifmt='snode.id')
    resp = requests.get(u, params=p)
    r = ivy.tree.read(resp.content)
    r.stree = stree_id
    r.ladderize()
    print 'done'
    return r

## Source tree 2: Tank & Donoghue 2010 - Campanulidae
## Source tree 3: Jansen et al 2007 - 81 cp genes, 64 taxa - angiosperms
## Source tree 4: Wang et al 2009 - rosids
## Source tree 5: Wurdack et al 2009 - Malpighiales

with open('strees.newick','w') as f:
    for stree_id in (2,3,4,5):
        r = fetch_stree(stree_id)
        f.write('%s\n' % r.write())
        
