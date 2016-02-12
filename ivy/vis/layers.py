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
from matplotlib.patches import PathPatch, Rectangle, Arc, Wedge, Circle
from matplotlib.path import Path
from matplotlib.widgets import RectangleSelector
from matplotlib.transforms import Bbox, offset_copy, IdentityTransform, \
     Affine2D
from matplotlib import cm as mpl_colormap
from matplotlib import colors as mpl_colors
from matplotlib.colorbar import Colorbar
from matplotlib.collections import RegularPolyCollection, LineCollection, \
     PatchCollection, CircleCollection
from matplotlib.lines import Line2D
from matplotlib.cm import coolwarm, afmhot
from matplotlib.cm import RdYlBu_r as RdYlBu
try:
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox, DrawingArea
except ImportError:
    pass
from matplotlib._png import read_png
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ivy.vis import symbols, colors
from ivy.vis import hardcopy as HC
from ivy.vis import events
import numpy as np
from numpy import pi, array
try:
    import Image
except ImportError:
    from PIL import Image


_tango = colors.tango()


def xy(plot, p):
    if isinstance(p, tree.Node):
        c = plot.n2c[p]
        p = (c.x, c.y)
    elif type(p) in types.StringTypes:
        c = plot.n2c[plot.root[p]]
        p = c.x, c.y
    elif isinstance(p, (list, tuple)):
        p = [ xy(plot, x) for x in p ]
    else:
        raise ValueError("Could not coerce %s to node" % [p])
    return p

def add_label(treeplot, labeltype, vis=True, leaf_offset=4,
             leaf_valign="center",
             leaf_halign="left", leaf_fontsize=10, branch_offset=-5,
             branch_valign="center", branch_halign="right",
             fontsize="10"):
    """
    Add text labels to tree

    Args:
        treeplot: treeplot (fig.tree)
        labeltype (str): "leaf" or "branch"

    """
    assert labeltype in ["leaf", "branch"], "invalid label type: %s" % labeltype
    n2c = treeplot.n2c
    if leaf_halign == "right": leaf_offset *= -1 # Padding in correct direction
    if branch_halign == "right": branch_offset *= -1
    for node, coords in n2c.items():
        x = coords.x; y = coords.y
        if node.isleaf and node.label and labeltype == "leaf":
            if treeplot.plottype == "phylogram":
                txt = treeplot.annotate(
                    node.label,
                    xy=(x, y),
                    xytext=(leaf_offset, 0),
                    textcoords="offset points",
                    verticalalignment=leaf_valign,
                    horizontalalignment=leaf_halign,
                    fontsize=fontsize,
                    clip_on=True,
                    picker=True,
                    visible=False,
                    zorder=1
                )
                txt.node = node
                #print "Setting node label to", str(txt), str(id(txt))
                treeplot.node2label[node]=txt
            else:
                if coords.angle < 90 or coords.angle > 270:
                    ha = "left"
                    va = "center"
                    rotate = (coords.angle-360)
                else:
                    ha = "right"
                    va = "center"
                    rotate = (coords.angle-180)

                txt = treeplot.annotate(
                    node.label,
                    xy=(x,y),
                    verticalalignment=va,
                    fontsize=leaf_fontsize,
                    clip_on=True,
                    picker=True,
                    visible=vis,
                    horizontalalignment=ha,
                    rotation=rotate,
                    rotation_mode="anchor",
                    zorder=1)

                txt.node = node
                treeplot.node2label[node]=txt

        if (not node.isleaf) and node.label and labeltype == "branch":
            txt = treeplot.annotate(
                node.label,
                xy=(x, y),
                xytext=(branch_offset,0),
                textcoords="offset points",
                verticalalignment=branch_valign,
                horizontalalignment=branch_halign,
                fontsize=fontsize,
                bbox=dict(fc="lightyellow", ec="none", alpha=0.8),
                clip_on=True,
                picker=True,
                visible=vis,
                zorder=1
            )
            txt.node = node
            treeplot.node2label[node]=txt

    # Drawing the leaves so that only as many labels as will fit get rendered

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
                if treeplot.plottype in ("phylogram", "overview"):
                    verts.append((x, y)); codes.append(M)
                    verts.append((px, y)); codes.append(L)
                    verts.append((px, py)); codes.append(L)
                    seen.add(node)
                elif treeplot.plottype == "radial":
                    v, c = treeplot._path_to_parent(node)
                    verts.extend(v)
                    codes.extend(c)
                    seen.add(node)

            if p == mrca or node == mrca:
                break
            node = p
            coords = treeplot.n2c[node]
            x = coords.x; y = coords.y
            p = node.parent
    if treeplot.plottype in ("phylogram", "overview"):
        px, py = verts[-1]
        verts.append((px, py)); codes.append(M)

    highlightpath = Path(verts, codes)
    highlightpatch = PathPatch(
        highlightpath, fill=False, linewidth=width, edgecolor=color,
        visible=vis, zorder=1)

    treeplot.add_patch(highlightpatch)
    treeplot.figure.canvas.draw_idle()

def add_cbar(treeplot, nodes, vis=True, color=None, label=None, x=None, width=8,
         xoff=10, showlabel=True, mrca=True, leaf_valign="center",
         leaf_halign="left", leaf_fontsize=10, leaf_offset=4):
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
        assert treeplot.plottype != "radial", "No cbar for radial trees"
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
                  linewidth=width, color=color, visible=vis, zorder=1)

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
                picker=False,
                zorder=1
                )

        treeplot.set_xlim(xlim); treeplot.set_ylim(ylim)

def add_image(treeplot, x, imgfiles, maxdim=100, border=0, xoff=4,
              yoff=4, halign=0.0, valign=0.5, xycoords='data',
              boxcoords=('offset points')):
    """
    Add images to a plot at the given nodes.

    Args:
        x: Node/label or list of nodes/labels.
        imgfiles: String or list of strings of image files
    Note:
        x and imgfiles must be the same length
    """
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
    if isinstance(imgfiles, str):
        imgfiles = [imgfiles]
    assert len(nodes) == len(imgfiles), "%s nodes, %s images" % (len(x),
                                                                 len(imgfiles))
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
                              boxcoords=boxcoords,
                              zorder=1)
        treeplot.add_artist(abox)
    treeplot.figure.canvas.draw_idle()

def add_squares(treeplot, nodes, colors='r', size=15, xoff=0, yoff=0, alpha=1.0,
                vis=True):
    """
    Draw a square at given node

    Args:
        p: A node or list of nodes or string or list of strings
        colors: Str or list of strs. Colors of squares to be drawn.
          Optional, defaults to 'r' (red)
        size (float): Size of the squares. Optional, defaults to 15
        xoff, yoff (float): Offset for x and y dimensions. Optional,
          defaults to 0.
        alpha (float): between 0 and 1. Alpha transparency of squares.
          Optional, defaults to 1 (fully opaque)
        zorder (int): The drawing order. Higher numbers appear on top
          of lower numbers. Optional, defaults to 1000.

    """
    points = xy(treeplot, nodes)
    trans = offset_copy(
        treeplot.transData, fig=treeplot.figure, x=xoff, y=yoff, units='points')

    col = RegularPolyCollection(
        numsides=4, rotation=pi*0.25, sizes=(size*size,),
        offsets=points, facecolors=colors, transOffset=trans,
        edgecolors='none', alpha=alpha, zorder=1
        )
    col.set_visible(vis)

    treeplot.add_collection(col)
    treeplot.figure.canvas.draw_idle()

def add_circles(treeplot, nodes, colors="g", size=15, xoff=0, yoff=0, vis=True):
    """
    Draw circles on plot

    Args:
        nodes: A node object or list of Node objects or label or list of labels
        colors: Str or list of strs. Colors of the circles. Optional,
          defaults to 'g' (green)
        size (float): Size of the circles. Optional, defaults to 15
        xoff, yoff (float): X and Y offset. Optional, defaults to 0.

    """
    points = xy(treeplot, nodes)
    trans = offset_copy(
        treeplot.transData, fig=treeplot.figure, x=xoff, y=yoff, units='points'
        )

    col = CircleCollection(
        sizes=(pi*size*size*0.25,),
        offsets=points, facecolors=colors, transOffset=trans,
        edgecolors='none', zorder=1
        )
    col.set_visible(vis)

    treeplot.add_collection(col)
    treeplot.figure.canvas.draw_idle()

def add_pie(treeplot, node, values, colors=None, size=16, norm=True,
        xoff=0, yoff=0,
        halign=0.5, valign=0.5,
        xycoords='data', boxcoords=('offset points')):
    """
    Draw a pie chart

    Args:
    node (Node): A single Node object or node label
    values (list): A list of floats.
    colors (list): A list of strings to pull colors from. Optional.
    size (float): Diameter of the pie chart
    norm (bool): Whether or not to normalize the values so they
      add up to 360
    xoff, yoff (float): X and Y offset. Optional, defaults to 0
    halign, valign (float): Horizontal and vertical alignment within
      box. Optional, defaults to 0.5

    """
    x, y = xy(treeplot, node)
    da = DrawingArea(size, size); r = size*0.5; center = (r,r)
    x0 = 0
    S = 360.0
    if norm: S = 360.0/sum(values)
    if not colors:
        c = _tango
        colors = [ c.next() for v in values ]
    for i, v in enumerate(values):
        theta = v*S
        if v: da.add_artist(Wedge(center, r, x0, x0+theta,
                                  fc=colors[i], ec='none'))
        x0 += theta
    box = AnnotationBbox(da, (x,y), pad=0, frameon=False,
                         xybox=(xoff, yoff),
                         xycoords=xycoords,
                         box_alignment=(halign,valign),
                         boxcoords=boxcoords)
    treeplot.add_artist(box)
    treeplot.figure.canvas.draw_idle()
    return box

def add_text(treeplot, x, y, s, color='black', xoff=0, yoff=0, valign='center',
         halign='left', fontsize=10):
    """
    Add text to the plot.

    Args:
        x, y (float): x and y coordinates to place the text
        s (str): The text to write
        color (str): The color of the text. Optional, defaults to "black"
        xoff, yoff (float): x and y offset
        valign (str): Vertical alignment. Can be: 'center', 'top',
          'bottom', or 'baseline'. Defaults to 'center'.
        halign (str): Horizontal alignment. Can be: 'center', 'right',
          or 'left'. Defaults to 'left'
        fontsize (float): Font size. Optional, defaults to 10

    """
    txt = treeplot.annotate(
        s, xy=(x, y),
        xytext=(xoff, yoff),
        textcoords="offset points",
        verticalalignment=valign,
        horizontalalignment=halign,
        fontsize=fontsize,
        clip_on=True,
        picker=True,
        zorder=1
    )
    txt.set_visible(True)
    return txt

def add_legend(treeplot, colors, labels, shape='rectangle',
               loc='upper left', **kwargs):
    """
    Add legend mapping colors/shapes to labels

    Args:
        colors (list): List of colors
        labels (list): List of labels
        shape (str): Shape of label icon. Either rectangle or circle
        loc (str): Location of label. Defaults to upper left
    """
    handles = []
    if shape == 'rectangle':
        for col, lab in zip(colors, labels):
            handles.append(Patch(color=col, label=lab))
            #shapes = [ CircleCollection([10],facecolors=[c]) for c in colors ]
    elif shape == "circle":
        for col, lab in zip(colors, labels):
            handles.append(matplotlib.pyplot.Line2D(range(1), range(1),
                           color="white",
                           label=lab, marker="o", markersize = 10,
                           markerfacecolor=col))

    treeplot.legend(handles=handles, loc=loc, numpoints=1, **kwargs)


def add_phylorate(treeplot, rates, nodeidx, vis=True):
    """
    Add phylorate plot generated from data analyzed with BAMM
    (http://bamm-project.org/introduction.html)

    Args:
        rates (array): Array of rates along branches
          created by r_funcs.phylorate
        nodeidx (array): Array of node indices matching rates (also created
          by r_funcs.phylorate)

    WARNING:
        Ladderizing the tree can cause incorrect assignment of Ape node index
        numbers. To prevent this, call this function or root.ape_node_idx()
        before ladderizing the tree to assign correct Ape node index numbers.
    """
    if not treeplot.root.apeidx:
        treeplot.root.ape_node_idx()
    segments = []
    values = []

    if treeplot.plottype == "radial":
        radpatches = [] # For use in drawing arcs for radial plots

        for n in treeplot.root.descendants():
            n.rates = rates[nodeidx==n.apeidx]
            c = treeplot.n2c[n]
            pc = treeplot._path_to_parent(n)[0][1]
            xd = c.x - pc[0]
            yd = c.y - pc[1]
            xseg = xd/len(n.rates)
            yseg = yd/len(n.rates)
            for i, rate in enumerate(n.rates):
                x0 = pc[0] + i*xseg
                y0 = pc[1] + i*yseg
                x1 = x0 + xseg
                y1 = y0 + yseg

                segments.append(((x0, y0), (x1, y1)))
                values.append(rate)

            curverts = treeplot._path_to_parent(n)[0][2:]
            curcodes = treeplot._path_to_parent(n)[1][2:]
            curcol = RdYlBu(n.rates[0])

            radpatches.append(PathPatch(
                       Path(curverts, curcodes), lw=2, edgecolor = curcol,
                            fill=False))
    else:
        for n in treeplot.root.descendants():
            n.rates = rates[nodeidx==n.apeidx]
            c = treeplot.n2c[n]
            pc = treeplot.n2c[n.parent]
            seglen = (c.x-pc.x)/len(n.rates)
            for i, rate in enumerate(n.rates):
                x0 = pc.x + i*seglen
                x1 = x0 + seglen
                segments.append(((x0, c.y), (x1, c.y)))
                values.append(rate)
            segments.append(((pc.x, pc.y), (pc.x, c.y)))
            values.append(n.rates[0])

    lc = LineCollection(segments, cmap=RdYlBu, lw=2)
    lc.set_array(np.array(values))
    treeplot.add_collection(lc)
    lc.set_zorder(1)
    if treeplot.plottype == "radial":
        arccol = matplotlib.collections.PatchCollection(radpatches,
                                                        match_original=True)
        treeplot.add_collection(arccol)
        arccol.set_visible(vis)
        arccol.set_zorder(1)
    lc.set_visible(vis)
    colorbar_legend(treeplot, values, RdYlBu, vis=vis)

    treeplot.figure.canvas.draw_idle()




def colorbar_legend(ax, values, cmap, vis=True):
    """
    Add a vertical colorbar legend to a plot
    """
    x_range = ax.get_xlim()[1]-ax.get_xlim()[0]
    y_range = ax.get_ylim()[1]-ax.get_ylim()[0]

    x = [ax.get_xlim()[0]+x_range*0.05]
    y = [ax.get_ylim()[1]-(y_range * 0.25), ax.get_ylim()[1]-(y_range*0.05)]

    segs = []
    vals=[]
    p = (x[0], y[0]+((y[1]-y[0])/256.0))
    for i in range(2, 257):
        n = (x[0], y[0]+((y[1]-y[0])/256.0)*i)
        segs.append((p, n))
        p = segs[-1][-1]
        vals.append(min(values)+((max(values)-min(values))/256.0)*(i-1))
    lcbar =  LineCollection(segs, cmap=cmap, lw=15)
    lcbar.set_visible(vis)
    lcbar.set_array(np.array(vals))
    ax.add_collection(lcbar)
    lcbar.set_zorder(1)


    minlab = str(min(values))[:6]
    maxlab = str(max(values))[:6]

    ax.text(x[0]+x_range*.02, y[0], minlab, verticalalignment="bottom", visible=vis)
    ax.text(x[0]+x_range*.02, y[1], maxlab, verticalalignment="top", visible=vis)

def add_node_heatmap(treeplot, nodelist, vis=True):
    """
    Plot circles on nodes with colors indicating how frequently each node
    appears in nodelist. For use with plotting potential regime
    shift locations

    Args:
        nodelist (list): list of node objects. Repeats allowed (and expected)
    """
    nodeset = list(set(nodelist))
    n = len(nodelist)

    # Don't plot any values that appear so infrequently as to be less than
    # what the minimum color would be
    cutoff = round(n*(1.0/255))

    nodeset = [x for x in nodeset if nodelist.count(x) >= cutoff]

    cols = [ afmhot(float(i)/n) for i in [nodelist.count(x) for x in nodeset] ]



    add_circles(treeplot, nodes=nodeset, colors=cols, size=6, vis=vis)
def add_ancestor_reconstruction(treeplot, internal_vals, vis=True, colors=None, nregime=None, size=8):
    """
    Add piecharts at each node based on probability vector

    Args:
        internal_vals: Array of dimensions [nchar+1, nnodes]. Identical to the
          format returned by hrm_multipass() with returnnodes = True.
          The first nchar number of columns correspond to the likelihood
          of that node being in each state.
         colors: List of length = nchar. The colors for each character. Optional.
         nregime: Number of regimes. If given and colors = None, function will
         automatically color-code characters by regime. (NOT IMPLEMENTED)
    """
    nodes = list(treeplot.root.postiter())
    nchar = internal_vals.shape[1]-1
    if colors is None:
        pass
    for i,n in enumerate(nodes):
        add_pie(treeplot, n, values = list(internal_vals[i][:-1]), colors=colors, size=size)

def add_branchstates(treeplot,vis=True, colors=None):
    """
    Add simulated branch states.

    Treeplot must have simulated states as attributes
    """
    segments = []
    cols = []
    nchar = len(set([ s.sim_char["sim_state"] for s in treeplot.root ]))
    for n in treeplot.root.descendants():
        c = treeplot.n2c[n]
        pc = treeplot.n2c[n.parent]
        segments.append(((pc.x,c.y),(c.x,c.y)))
        cols.append(n.parent.sim_char["sim_state"])
        for s in n.sim_char["sim_hist"]:
            segments.append(((c.x, c.y), (pc.x + s[1],c.y)))
            cols.append(s[0])
        segments.append(((pc.x, pc.y),(pc.x,c.y)))
        cols.append(n.parent.sim_char["sim_state"])
    if not colors:
        c = _tango
        colors = [c.next() for v in range(nchar)]
    colors = np.array(colors)
    lc = LineCollection(segments, colors=colors[cols], lw=2, antialiaseds=[0])
    treeplot.add_collection(lc)

    leg_handles = [None] * nchar
    for i,char in enumerate(range(nchar)):
        leg_handles[i]=matplotlib.patches.Patch(color = colors[i],
                       label=str(i))
    treeplot.legend(handles=leg_handles)
