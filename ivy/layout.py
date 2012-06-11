"""
layout nodes in 2d space

The function of interest is `calc_node_positions` (aka nodepos)
"""
import numpy

class Coordinates:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Coordinates(%g, %g)" % (self.x, self.y)

    def point(self):
        return (self.x, self.y)

def smooth_xpos(node, n2coords):
    if not node.isleaf:
        children = node.children
        for ch in children:
            smooth_xpos(ch, n2coords)
        
        if node.parent:
            px = n2coords[node.parent].x
            cx = min([ n2coords[ch].x for ch in children ])
            n2coords[node].x = (px + cx)/2.0

def depth_length_preorder_traversal(node, n2coords=None, isroot=False):
    "calculate node depth (root = depth 0) and length to root"
    if n2coords is None:
        n2coords = {}
    coords = n2coords.get(node) or Coordinates()
    coords.node = node
    if (not node.parent) or isroot:
        coords.depth = 0
        coords.length_to_root = 0.0
    else:
        try:
            p = n2coords[node.parent]
            coords.depth = p.depth + 1
            coords.length_to_root = p.length_to_root + (node.length or 0.0)
        except KeyError:
            print node.label, node.parent.label
        except AttributeError:
            coords.depth = 0
            coords.length_to_root = 0
    n2coords[node] = coords

    for ch in node.children:
        depth_length_preorder_traversal(ch, n2coords, False)

    return n2coords

def calc_node_positions(node, width, height,
                        lpad=0, rpad=0, tpad=0, bpad=0,
                        scaled=True, smooth=True, n2coords=None):
    "origin is at upper left"
    width -= (lpad + rpad)
    height -= (tpad + bpad)

    if n2coords is None:
        n2coords = {}
    depth_length_preorder_traversal(node, n2coords=n2coords)
    leaves = node.leaves()
    nleaves = len(leaves)
    maxdepth = max([ n2coords[lf].depth for lf in leaves ])
    unitwidth = width/float(maxdepth)
    unitheight = height/(nleaves-1.0)

    xoff = (unitwidth * 0.5)
    yoff = (unitheight * 0.5)

    if scaled:
        maxlen = max([ n2coords[lf].length_to_root for lf in leaves ])
        scale = width/maxlen

    for i, lf in enumerate(leaves):
        c = n2coords[lf]
        c.y = i * unitheight
        if not scaled:
            c.x = width
        else:
            c.x = c.length_to_root * scale

    for n in node.postiter():
        c = n2coords[n]
        if (not n.isleaf) and n.children:
            children = n.children
            ymax = n2coords[children[0]].y
            ymin = n2coords[children[-1]].y
            c.y = (ymax + ymin)/2.0
            if not scaled:
                c.x = min([ n2coords[ch].x for ch in children ]) - unitwidth
            else:
                c.x = c.length_to_root * scale

    if (not scaled) and smooth:
        for i in range(10):
            smooth_xpos(node, n2coords)

    for coords in n2coords.values():
        coords.x += lpad
        coords.y += tpad

    for n, coords in n2coords.items():
        if n.parent:
            p = n2coords[n.parent]
            coords.px = p.x; coords.py = p.y
        else:
            coords.px = None; coords.py = None

    return n2coords

nodepos = calc_node_positions

def cartesian(node, xscale=1.0, leafspace=None, scaled=True, n2coords=None,
              smooth=0, array=numpy.array, ones=numpy.ones, yunit=None):
    
    if n2coords is None:
        n2coords = {}

    depth_length_preorder_traversal(node, n2coords, True)
    leaves = node.leaves()
    nleaves = len(leaves)

    # leafspace is a vector that should sum to nleaves
    if leafspace is None:
        leafspace = ones((nleaves,))
    else:
        assert len(leafspace) == nleaves
        leafspace = array(leafspace)/(sum(leafspace)/nleaves)
    
    maxdepth = max([ n2coords[lf].depth for lf in leaves ])
    depth = maxdepth * xscale
    if not yunit: yunit = 1.0/nleaves
    ## yunit = 1.0

    if scaled:
        maxlen = max([ n2coords[lf].length_to_root for lf in leaves ])
        depth = maxlen

    y = 0
    for i, lf in enumerate(leaves):
        c = n2coords[lf]
        yoff = leafspace[i] * yunit
        c.y = y + yoff*0.5
        y += yoff
        if not scaled:
            c.x = depth
        else:
            c.x = c.length_to_root

    for n in node.postiter():
        c = n2coords[n]
        if not n.isleaf:
            children = n.children
            v = [n2coords[children[0]].y, n2coords[children[-1]].y]
            v.sort()
            ymin, ymax = v
            c.y = (ymax + ymin)/2.0
            if not scaled:
                c.x = min([ n2coords[ch].x for ch in children ]) - 1.0
            else:
                c.x = c.length_to_root

    if not scaled:
        for i in range(smooth):
            smooth_xpos(node, n2coords)

    return n2coords

if __name__ == "__main__":
    import tree
    node = tree.read("(a:3,(b:2,(c:4,d:5):1,(e:3,(f:1,g:1):2):2):2);")
    for i, n in enumerate(node.iternodes()):
        if not n.isleaf:
            n.label = "node%s" % i
    node.label = "root"
    n2c = calc_node_positions(node, width=10, height=10, scaled=True)

    from pprint import pprint
    pprint(n2c)
