"""
Layer functions to add to a tree plot with the addlayer method
"""
import sys, time, bisect, math, types, os, operator, functools
from collections import defaultdict
from itertools import chain
from pprint import pprint
from ivy import tree, bipart
from ivy.layout import cartesian
from ivy.storage import Storage
from ivy import pyperclip as clipboard
#from ..nodecache import NodeCache
import matplotlib, numpy
import matplotlib.pyplot as pyplot
from matplotlib.figure import SubplotParams, Figure
from matplotlib.axes import Axes, subplot_class_factory
from matplotlib.patches import PathPatch, Rectangle, Arc
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
from matplotlib.transforms import Bbox, offset_copy, IdentityTransform, \
     Affine2D
from matplotlib import cm as mpl_colormap
from matplotlib import colors as mpl_colors
from matplotlib.colorbar import Colorbar
from matplotlib.collections import RegularPolyCollection, LineCollection, \
     PatchCollection
from matplotlib.lines import Line2D
from matplotlib.cm import coolwarm
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except ImportError:
    pass
from matplotlib._png import read_png
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ivy.vis import symbols, colors
from ivy.vis import hardcopy as HC
from ivy.vis import events
try:
    import Image
except ImportError:
    from PIL import Image
    
_tango = colors.tango()

def add_label(treeplot, labeltype, vis=True, leaf_offset=4, leaf_valign="center",
             leaf_halign="left", leaf_fontsize=10, branch_offset=-5,
             branch_valign="center", branch_halign="right", 
             branch_fontsize="10"):
    """
    Add text labels to tree
    
    Args:
        treeplot: treeplot (fig.tree)
        labeltype (str): "leaf" or "branch"
        
    """
    assert labeltype in ["leaf", "branch"], "invalid label type: %s" % labeltype
    n2c = treeplot.n2c
    for node, coords in n2c.items():
        x = coords.x; y = coords.y
        if node.isleaf and node.label and labeltype == "leaf":
            txt = treeplot.annotate(
                node.label,
                xy=(x, y),
                xytext=(leaf_offset, 0),
                textcoords="offset points",
                verticalalignment=leaf_valign,
                horizontalalignment=leaf_halign,
                fontsize=leaf_fontsize,
                clip_on=True,
                picker=True,
                visible=False
            )
            txt.node = node
            #print "Setting node label to", str(txt), str(id(txt))
            treeplot.node2label[node]=txt

        if (not node.isleaf) and node.label and labeltype == "branch":
            txt = treeplot.annotate(
                node.label,
                xy=(x, y),
                xytext=(branch_offset,0),
                textcoords="offset points",
                verticalalignment=branch_valign,
                horizontalalignment=branch_halign,
                fontsize=branch_fontsize,
                bbox=dict(fc="lightyellow", ec="none", alpha=0.8),
                clip_on=True,
                picker=True,
                visible=vis
            )
            txt.node = node
            treeplot.node2label[node]=txt
            
    # Drawing the leaves so that only as many labels as will fit get rendered
    if labeltype == "leaf":        
        leaves = list(filter(lambda x:x[0].isleaf,
                             treeplot.get_visible_nodes(labeled_only=True)))
        psep = treeplot.leaf_pixelsep()
        fontsize = min(leaf_fontsize, max(psep, 8))
        n2l = treeplot.node2label
        transform = treeplot.transData.transform
        sub = operator.sub

        for n in leaves:
            n2l[n[0]].set_visible(False)

        # draw leaves
        leaves_drawn = []
        for n, x, y in leaves:
            txt = treeplot.node2label[n]
            if not leaves_drawn:
                txt.set_visible(vis)
                leaves_drawn.append(txt)
                treeplot.figure.canvas.draw_idle()
                matplotlib.pyplot.show()
                continue

            txt2 = leaves_drawn[-1]
            y0 = y; y1 = txt2.xy[1]
            sep = sub(*transform(([0,y0],[0,y1]))[:,1])
            if sep > fontsize:
                txt.set_visible(vis)
                txt.set_size(fontsize)
                leaves_drawn.append(txt)
        treeplot.figure.canvas.draw_idle()
        matplotlib.pyplot.show()

        if leaves_drawn:
            leaves_drawn[0].set_size(fontsize)
    treeplot.figure.canvas.draw_idle()
    matplotlib.pyplot.show()
    
    
    
def add_highlight(treeplot, x=None, vis=True, width=5, color="red"):
    """
    Highlight nodes

    Args:
        x: Str or list of Strs or Node or list of Nodes
        width (float): Width of highlighted lines. Defaults to 5
        color (str): Color of highlighted lines. Defaults to red
        vis (bool): Whether or not the object is visible. Defaults to true
    """
    if x:
        nodes = set()
        if type(x) in types.StringTypes:
            nodes = treeplot.root.findall(x)
        elif isinstance(x, tree.Node):
            nodes = set(x)
        else:
            for n in x:
                if type(n) in types.StringTypes:
                    found = treeplot.root.findall(n)
                    if found:
                        nodes |= set(found)
                elif isinstance(n, tree.Node):
                    nodes.add(n)

        highlighted = nodes
    else:
        highlighted = set()
    
    if len(highlighted)>1:
        mrca = treeplot.root.mrca(highlighted)
        if not mrca:
            return
    else:
        mrca = list(nodes)[0]    
    
    M = Path.MOVETO; L = Path.LINETO
    verts = []
    codes = []
    
    seen = set()
    for node, coords in [ x for x in treeplot.n2c.items() if x[0] in nodes ]:
        x = coords.x; y = coords.y
        p = node.parent
        while p:
            pcoords = treeplot.n2c[p]
            px = pcoords.x; py = pcoords.y
            if node not in seen:
                verts.append((x, y)); codes.append(M)
                verts.append((px, y)); codes.append(L)
                verts.append((px, py)); codes.append(L)
                seen.add(node)
            if p == mrca or node == mrca:
                break
            node = p
            coords = treeplot.n2c[node]
            x = coords.x; y = coords.y
            p = node.parent
    px, py = verts[-1]
    verts.append((px, py)); codes.append(M)

    highlightpath = Path(verts, codes)
    highlightpatch = PathPatch(
        highlightpath, fill=False, linewidth=width, edgecolor=color, visible=vis
        )
    treeplot.add_patch(highlightpatch)
    treeplot.figure.canvas.draw_idle()
    
def add_cbar(treeplot, nodes, vis=True, color=None, label=None, x=None, width=8, xoff=10,
         showlabel=True, mrca=True, leaf_valign="center", leaf_halign="left", 
         leaf_fontsize=10, leaf_offset=4):
        """
        Draw a 'clade' bar (i.e., along the y-axis) indicating a
        clade.  *nodes* are assumed to be one or more nodes in the
        tree.  If just one, it should be the internal node
        representing the clade of interest; otherwise, the clade of
        interest is the most recent common ancestor of the specified
        nodes.  *label* is an optional string to be drawn next to the
        bar, *offset* by the specified number of display units.  If
        *label* is ``None`` then the clade's label is used instead.

        Args:
            nodes: Node or list of nodes or string or list of strings.
            color (str): Color of the bar. Optional, defaults to None.
              If None, will cycle through a color palette
            label (str): Optional label for bar. If None, the clade's
              label is used instead. Defaults to None.
            width (float): Width of bar
            xoff (float): Offset from label to bar
            showlabel (bool): Whether or not to draw the label
            mrca (bool): Whether to draw the bar encompassing all descendants
              of the MRCA of ``nodes``
        """
        xlim = treeplot.get_xlim(); ylim = treeplot.get_ylim()
        if color is None: color = _tango.next()
        transform = treeplot.transData.inverted().transform        
    
        if mrca:
            if isinstance(nodes, tree.Node):
                spec = nodes
            elif type(nodes) in types.StringTypes:
                spec = treeplot.root.get(nodes)
            else:
                spec = treeplot.root.mrca(nodes)

            assert spec in treeplot.root
            label = label or spec.label
            leaves = spec.leaves()

        else:
            leaves = nodes

        n2c = treeplot.n2c    
    
        y = sorted([ n2c[n].y for n in leaves ])
        ymin = y[0]; ymax = y[-1]; y = (ymax+ymin)*0.5
        treeplot.figure.canvas.draw_idle()
        if x is None:
            x = max([ n2c[n].x for n in leaves ])
            _x = 0
            for lf in leaves:
                txt = treeplot.node2label.get(lf)
                #print "Accessing", str(txt), str(id(txt))
                if txt and txt.get_visible():
                    treeplot.figure.canvas.draw_idle()
                    _x = max(_x, transform(txt.get_window_extent())[1,0])
            if _x > x: x = _x

        v = sorted(list(transform(((0,0),(xoff,0)))[:,0]))
        xoff = v[1]-v[0]
        x += xoff

        Axes.plot(treeplot, [x,x], [ymin, ymax], '-', 
                  linewidth=width, color=color, visible=vis)

        if showlabel and label:
            xo = leaf_offset
            if xo > 0:
                xo += width*0.5
            else:
                xo -= width*0.5
            txt = treeplot.annotate(
                label,
                xy=(x, y),
                xytext=(xo, 0),
                textcoords="offset points",
                verticalalignment=leaf_valign,
                horizontalalignment=leaf_halign,
                fontsize=leaf_fontsize,
                clip_on=True,
                picker=False
                )

        treeplot.set_xlim(xlim); treeplot.set_ylim(ylim)

def add_image(treeplot, x, imgfiles, maxdim=100, border=0, xoff=4,
              yoff=4, halign=0.0, valign=0.0, xycoords='data',
              boxcoords=('offset points')):
    """
    Add images to a plot at the given nodes.
    
    Args:
        x: Node/label or list of nodes/labels.
        imgfiles: String or list of strings of image files 
    Note:
        x and imgfiles must be the same length
    """
    assert len(x) == len(imgfiles)
    if x:
        nodes = []
    if type(x) in types.StringTypes:
        nodes = treeplot.root[x]
    elif isinstance(x, tree.Node):
        nodes = [x]
    else:
        for n in x:
            if type(n) in types.StringTypes:
                nodes.append(treeplot.root[n])
            elif isinstance(n, tree.Node):
                nodes.append(n)
    for node, imgfile in zip(nodes, imgfiles):
        coords = treeplot.n2c[node]
        img = Image.open(imgfile)
        if max(img.size) > maxdim:
            img.thumbnail((maxdim, maxdim))
        imgbox = OffsetImage(img)
        abox = AnnotationBbox(imgbox, (coords.x, coords.y),
                              xybox= (xoff, yoff), xycoords=xycoords,
                              box_alignment=(halign, valign),
                              pad=0.0,
                              boxcoords=boxcoords)
        treeplot.add_artist(abox)
    plot.figure.canvas.draw_idle()
    
def add_phylorate(treeplot, rates):
    """
    Add phylorate plot generated from data analyzed with BAMM
    (http://bamm-project.org/introduction.html)
    
    Args:
        rates (array): Array of rates along branches created by (TBA function)
    """
    # Give nodes ape index numbers - possibly should be its own function
    i = 1
    for lf in treeplot.root.leaves():
        lf.apeidx = i
        i += 1
    for n in treeplot.root.clades():
        n.apeidx = i
        i += 1
    
    
                
