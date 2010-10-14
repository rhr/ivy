"""
For drawing big trees.  Calculate which clades can be 'collapsed' and
displayed with a placeholder.

TODO: test and develop this module further
"""
from storage import Storage

def autocollapse_info(node, collapsed, visible=True, info={}):
    """
    gather information to determine if a node should be collapsed

    *collapsed* is a set containing nodes that are already collapsed
    """
    if node not in info:
        s = Storage()
        info[node] = s
    else:
        s = info[node]
        
    if visible and (node in collapsed):
        visible = False
        
    nnodes = 1 # total number of nodes, including node
    # number of visible leaves
    nvisible = int((visible and node.isleaf) or (node in collapsed))
    ntips = int(node.isleaf)
    ntips_visible = int(node.isleaf and visible)
    s.has_labeled_descendant = False
    s.depth = 1

    for child in node.children:
        autocollapse_info(child, collapsed, visible, info)
        cs = info[child]
        nnodes += cs.nnodes
        nvisible += cs.nvisible
        ntips += cs.ntips
        ntips_visible += cs.ntips_visible
        if (child.label and (not child.isleaf)) \
           or (cs.has_labeled_descendant):
            s.has_labeled_descendant = True
        if cs.depth >= s.depth:
            s.depth = cs.depth+1
    s.nnodes = nnodes
    s.nvisible = nvisible
    s.ntips = ntips
    s.ntips_visible = ntips_visible
    return info

def autocollapse(root, collapsed=None, keep_visible=None, max_visible=1000):
    """
    traverse a tree and find nodes that should be collapsed in order
    to satify *max_visible*

    *collapsed* is a set object for storing collapsed nodes

    *keep_visible* is a set object of nodes that should not be placed
    in *collapsed*
    """
    collapsed = collapsed or set()
    keep_visible = keep_visible or set()
    ntries = 0
    while True:
        if ntries > 10:
            return
        info = autocollapse_info(root, collapsed)
        nvisible = info[root].nvisible
        if nvisible <= max_visible:
            return
        
        v = []
        for node in root.iternodes():
            s = info[node]
            if (node.label and (not node.isleaf) and node.parent and
                (node not in keep_visible)):
                w = s.nvisible/float(s.depth)
                if s.has_labeled_descendant:
                    w *= 0.25
                v.append((w, node, s))
        v.sort(); v.reverse()
        for w, node, s in v:
            if node not in keep_visible and s.nvisible < (nvisible-1):
                print node
                collapsed.add(node)
                nvisible -= s.nvisible
            if nvisible <= max_visible:
                break
        ntries += 1
    return collapsed
