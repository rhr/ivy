from array import array
from layout import depth_length_preorder_traversal

class AsciiBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._b = [ array('c', ' '*width) for line in range(height) ]

    def putstr(self, r, c, s):
        assert r < self.height
        assert c+len(s) <= self.width, "%s %s %s '%s'" % (self.width, r, c, s)
        self._b[r][c:c+len(s)] = array('c', s)

    def __str__(self):
        return "\n".join([ b.tostring() for b in self._b ])

def sum_to_root(node, internodes=True, length=False):
    i = 0
    n = node
    while 1:
        if not n.parent:
            break
        else:
            n = n.parent
            i += 1
    return i

## def depth_length_preorder_traversal(node):
##     if not node.parent:
##         node.depth = 0
##         node.length_to_root = 0.0
##     else:
##         p = node.parent
##         node.depth = p.depth + 1
##         node.length_to_root = p.length_to_root + (node.length or 0.0)
##     for ch in node.children:
##         depth_length_preorder_traversal(ch)

def smooth_cpos(node, n2c):
    for ch in node.children:
        smooth_cpos(ch, n2c)
        
    if node.parent and not node.isleaf:
        px = n2c[node.parent].c
        cx = min([ n2c[ch].c for ch in node.children ])
        dxp = n2c[node].c - px
        cxp = cx - n2c[node].c
        node.c = int(px + (cx - px)*0.5)

def scale_cpos(node, n2c, scalef, root_offset):
    if node.parent:
        n2c[node].c = n2c[node.parent].c + int(node.length * scalef)
    else:
        n2c[node].c = root_offset

    for ch in node.children:
        scale_cpos(ch, n2c, scalef, root_offset)

def set_rpos(node, n2c):
    for child in node.children:
        set_rpos(child, n2c)
        nc = n2c[node]
        if node.children:
            children = node.children
            c0 = n2c[children[0]]
            c1 = n2c[children[-1]]
            rmin = c0.r; rmax = c1.r
            nc.r = int(rmin + (rmax-rmin)/2.0)

def render(root, unitlen=3, minwidth=50, maxwidth=None, scaled=False,
           show_internal_labels=True):
    n2c = depth_length_preorder_traversal(root)
    leaves = root.leaves(); nleaves = len(leaves)
    maxdepth = max([ n2c[lf].depth for lf in leaves ])
    max_labelwidth = max([ len(lf.label) for lf in leaves ]) + 1

    root_offset = 0
    if root.label and show_internal_labels:
        root_offset = len(root.label)
        
    width = maxdepth*unitlen + max_labelwidth + 2 + root_offset
    height = 2*nleaves - 1

    if width < minwidth:
        unitlen = (minwidth - max_labelwidth - 2 - root_offset)/maxdepth
        width = maxdepth*unitlen + max_labelwidth + 2 + root_offset

    buf = AsciiBuffer(width, height)

    for i, lf in enumerate(leaves):
        c = n2c[lf]
        c.c = width - max_labelwidth - 2
        c.r = i*2

    for node in root.postiter():
        nc = n2c[node]
        if node.children:
            children = node.children
            c0 = n2c[children[0]]
            c1 = n2c[children[-1]]
            rmin = c0.r; rmax = c1.r
            nc.r = int(rmin + (rmax-rmin)/2.0)
            nc.c = min([ n2c[ch].c for ch in children ]) - unitlen

    if not scaled:
        smooth_cpos(root, n2c)
    else:
        maxlen = max([ n2c[lf].length_to_root for lf in leaves ])
        scalef = (n2c[leaves[0]].c + 1 - root_offset)/maxlen
        scale_cpos(root, n2c, scalef, root_offset)

    for node in root.postiter():
        nc = n2c[node]
        if node.parent:
            pc = n2c[node.parent]
            for r in range(min([nc.r, pc.r]),
                           max([nc.r, pc.r])):
                buf.putstr(r, pc.c, ":")

            sym = getattr(nc, "hchar", "-")
            vbar = sym*(nc.c-pc.c)
            buf.putstr(nc.r, pc.c, vbar)

        if node.isleaf:
            buf.putstr(nc.r, nc.c+1, " "+node.label)
        else:
            if node.label and show_internal_labels:
                buf.putstr(nc.r, nc.c-len(node.label), node.label)

        buf.putstr(nc.r, nc.c, "+")
        
    return str(buf)

if __name__ == "__main__":
    import random, tree
    rand = random.Random()
    
    t = tree.read(
        "(foo,((bar,(dog,cat)dc)dcb,(shoe,(fly,(cow, bowwow)cowb)cbf)X)Y)Z;"
        )

    #t = tree.read("(((foo:4.6):5.6, (bar:6.5, baz:2.3):3.0):3.0);")
    #t = tree.read("(foo:4.6, (bar:6.5, baz:2.3)X:3.0)Y:3.0;")

    i = 1
    print render(t, scaled=0, show_internal_labels=1)
    r = t.get("cat").parent
    tree.reroot(t, r)
    tp = t.parent
    tp.remove_child(t)
    c = t.children[0]
    t.remove_child(c)
    tp.add_child(c)
    print render(r, scaled=0, show_internal_labels=1)
