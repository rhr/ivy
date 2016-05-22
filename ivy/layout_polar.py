from __future__ import absolute_import, division, print_function, unicode_literals

import math
from .storage import Storage

CLOCKWISE = -1
COUNTERCLOCKWISE = 1

class Coordinates:
    def __init__(self):
        pass

def smooth_xpos(node, n2coords):
    if not node.isleaf:
        children = node.children
        for ch in children:
            smooth_xpos(ch, n2coords)

        if node.parent:
            px = n2coords[node.parent].x
            cx = min([ n2coords[ch].x for ch in children ])
            n2coords[node].x = (px + cx)/2.0

    #print "scaled", node.label, node.x, node.y

def depth_length_preorder_traversal(node, n2coords=None):
    """
    Calculate node depth (root = depth 0) and length to root

    Args:
        node (Node): A node object.
    Returns:
        dict: Mapping of nodes to coordinates instances. Coordinate
        instances have attributes "depth" and "length_to_root"
    """
    if n2coords is None:
        n2coords = {}
    coords = n2coords.get(node) or Coordinates()
    coords.node = node
    if not node.parent:
        coords.depth = 0
        coords.length_to_root = 0.0
    else:
        #print node.parent, node.parent.length
        try:
            p = n2coords[node.parent]
            coords.depth = p.depth + 1
            coords.length_to_root = p.length_to_root + (node.length or 0.0)
        except KeyError:
            print(node.label, node.parent.label)
        except AttributeError:
            coords.depth = 0
            coords.length_to_root = 0
    n2coords[node] = coords

    for ch in node.children:
        depth_length_preorder_traversal(ch, n2coords)

    return n2coords

def calc_node_positions(node, radius=1.0, pole=None,
                        start=0, end=None,
                        direction=COUNTERCLOCKWISE,
                        scaled=False, n2coords=None):
    """
    Calculate where nodes should be positioned in 2d space for drawing a
    polar tree

    Args:
        node (Node): A (root) node
        radius (float): The radius of the tree. Optional, defaults to 1
        pole (tuple): Tuple of floats. The cartesian coordinate of the pole.
          Optional, defaults to None.
        end (float): Where the tree ends. For best results, between 0 and 360.
          Optional, defaults to None.
        direction: CLOCKWISE or COUNTERCLOCKWISE. The direction the tree is
          drawn. Optional, defaults to COUNTERCLOCKWISE
        scaled (bool): Whether or not the tree is scaled. Optional, defaults
          to False.
    Returns:
        dict: Mapping of nodes to Coordinates object
    """
    leaves = node.leaves()
    nleaves = len(leaves)

    if pole is None:
        pole = (0,0) # Cartesian coordinate of pole
    if end is None:
        end = 360.0

    unitwidth = float(end)/nleaves

    if n2coords is None:
        n2coords = {}

    depth_length_preorder_traversal(node, n2coords)

    c = n2coords[node]
    c.x = pole[0]
    c.y = pole[1]
    maxdepth = max([ n2coords[lf].depth for lf in leaves ])
    unitdepth = radius/float(maxdepth)
    #unitangle = (2*math.pi)/nleaves
    totalarc = end - start
    if direction == CLOCKWISE:
        totalarc = 360.0 - totalarc

    if scaled:
        maxlen = max([ n2coords[lf].length_to_root for lf in leaves ])
        scale = radius/maxlen

    for i, lf in enumerate(leaves):
        i += 1
        c = n2coords[lf]
        c.angle = start + i*unitwidth*direction
        #print lf.label, c.angle
        if not scaled:
            c.depth = radius
        else:
            c.depth = c.length_to_root * scale

    for n in node.postiter():
        c = n2coords[n]
        if not n.isleaf:
            children = n.children
            min_angle = n2coords[children[0]].angle
            max_angle = n2coords[children[-1]].angle
            c.angle = (max_angle + min_angle)/2.0
            #print min_angle, max_angle, c.angle
            if not scaled:
                c.depth = min([ n2coords[ch].depth for ch in children ]) \
                          - unitdepth
            else:
                c.depth = c.length_to_root * scale

        if n.parent:
            c.x = math.cos(math.radians(c.angle))*c.depth + pole[0]
            c.y = math.sin(math.radians(c.angle))*c.depth + pole[1]


    ## if not scaled:
    ##     for i in range(10):
    ##         smooth_xpos(node, n2coords)

    return n2coords

def test():
    from . import newick
    node = newick.parse("(a:3,(b:2,(c:4,d:5):1,(e:3,(f:1,g:1):2):2):2);")
    for i, n in enumerate(node.iternodes()):
        if not n.isleaf:
            n.label = "node%s" % i
    node.label = "root"
    n2c = calc_node_positions(node, radius=100, scaled=False)

    from matplotlib.patches import Arc, PathPatch
    from matplotlib.collections import PatchCollection, LineCollection
    import matplotlib.pyplot as P
    f = P.figure()
    a = f.add_subplot(111)
    arcs = []; lines = []
    for n in node.iternodes():
        c = n2c[n]
        if n.parent and n.children:
            theta1 = n2c[n.children[0]].angle
            theta2 = n2c[n.children[-1]].angle
            arc = Arc((0,0), c.depth*2, c.depth*2, theta1=theta1, theta2=theta2)
            arcs.append(arc)
        if n.parent:
            p = n2c[n.parent]
            px = math.cos(math.radians(c.angle))*p.depth
            py = math.sin(math.radians(c.angle))*p.depth
            lines.append(((c.x,c.y),(px, py)))

        if n.label:
            txt = a.annotate(
                n.label,
                xy=(c.x, c.y),
                xytext=(0, 0),
                textcoords="offset points"
                )


    arcs = PatchCollection(arcs, match_original=True)
    a.add_collection(arcs)
    lines = LineCollection(lines)
    a.add_collection(lines)
    a.set_xlim((-100,100))
    a.set_ylim((-100,100))
    f.show()
