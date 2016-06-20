"""
Layer functions to add to a tree plot with the addlayer method
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, time, bisect, math, types, os, operator, functools
from collections import defaultdict, Counter
from itertools import chain
import copy
from pprint import pprint
from ivy import tree, bipart
from ivy.layout import cartesian
from ivy.storage import Storage
from ivy import pyperclip as clipboard
#from ..nodecache import NodeCache
import matplotlib
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
from matplotlib.colors import LinearSegmentedColormap
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
from ivy.vis import colors
from ivy.vis import events
import numpy as np
from numpy import pi, array
try:
    import Image
except ImportError:
    from PIL import Image
from colour import Color
import ivy

_tango = colors.tango()

try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]

def xy(plot, p):
    """
    Get xy coordinates of a node

    Args:
        plot (TreeSubplot): treeplot
        p: node or node label (or list of nodes/node labels)
    """
    if isinstance(p, tree.Node):
        c = plot.n2c[p]
        p = (c.x, c.y)
    elif type(p) in StringTypes:
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
    for node, coords in list(n2c.items()):
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
        if type(x) in StringTypes:
            nodes = treeplot.root.findall(x)
        elif isinstance(x, tree.Node):
            nodes = set(x)
        else:
            for n in x:
                if type(n) in StringTypes:
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
    for node, coords in [ x for x in list(treeplot.n2c.items()) if x[0] in nodes ]:
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
            x (float): Distance from tip of tree to bar. Optional,
              if None, calculated based on leaf labels.
            xoff (float): Offset from label to bar
            showlabel (bool): Whether or not to draw the label
            mrca (bool): Whether to draw the bar encompassing all descendants
              of the MRCA of ``nodes``
        """
        #assert treeplot.plottype != "radial", "No cbar for radial trees"
        xlim = treeplot.get_xlim(); ylim = treeplot.get_ylim()
        if color is None: color = next(_tango)
        transform = treeplot.transData.inverted().transform

        if mrca:
            if isinstance(nodes, tree.Node):
                spec = nodes
            elif type(nodes) in StringTypes:
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
        if x is None: # Determining how far bar should be from tips
            x = max([ n2c[n].x for n in leaves ])
            _x = 0
            for lf in leaves:
                txt = treeplot.node2label.get(lf)
                if txt and txt.get_visible():
                    treeplot.figure.canvas.draw_idle()
                    pyplot.pause(0.001) # Pause necessary for getting window extent
                    _x = max(_x, transform(txt.get_window_extent())[1,0])
            if _x > x: x = _x

        v = sorted(list(transform(((0,0),(xoff,0)))[:,0]))
        xoff = v[1]-v[0]
        x += xoff

        if treeplot.plottype in ("phylogram", "overview"):
            Axes.plot(treeplot, [x,x], [ymin, ymax], '-',
                      linewidth=width, color=color, visible=vis, zorder=1)
        else:
            arc_center = [0,0]
            diam = 2.0 + x
            theta1 = n2c[leaves[0]].angle
            theta2 = n2c[leaves[-1]].angle

            p = matplotlib.patches.Arc(arc_center, diam,diam, theta1=theta1, theta2=theta2,
                                       color=color, visible=vis,zorder=1,linewidth=width)
            treeplot.add_patch(p)

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
    if type(x) in StringTypes:
        nodes = treeplot.root[x]
    elif isinstance(x, tree.Node):
        nodes = [x]
    else:
        for n in x:
            if type(n) in StringTypes:
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
    Draw circles on plot at nodes

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


def add_circles_branches(treeplot, nodes, distances, colors="g", size=15,xoff=0,yoff=0,vis=True):
    """
    Draw circles on branches

    Args:
        nodes: A node object or list of Node objects or label or list of labels
        distances: Float or list of floats indicating the distance from the
          **parent** node the branch should be drawn on.
        colors: Str or list of strs. Colors of the circles. Optional,
          defaults to 'g' (green)
        size (float): Size of the circles. Optional, defaults to 15
        xoff, yoff (float): X and Y offset. Optional, defaults to 0.

    """
    points = xy(treeplot, nodes)

    coords = [(x[0]-distances[i],x[1]) for i,x in enumerate(points)]
    trans = offset_copy(
        treeplot.transData, fig=treeplot.figure, x=xoff, y=yoff, units='points'
        )

    col = CircleCollection(
        sizes=(pi*size*size*0.25,),
        offsets=coords, facecolors=colors, transOffset=trans,
        edgecolors='none', zorder=1
        )
    col.set_visible(vis)

    treeplot.add_collection(col)
    treeplot.figure.canvas.draw_idle()

def add_pie(treeplot, node, values, colors=None, size=16, norm=True,
        xoff=0, yoff=0,
        halign=0.5, valign=0.5,
        xycoords='data', boxcoords=('offset points'), vis=True):
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
        colors = [ next(c) for v in values ]
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
    box.set_visible(vis)
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
            handles.append(matplotlib.pyplot.Line2D(list(range(1)), list(range(1)),
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
def add_ancestor_noderecon(treeplot, internal_vals, vis=True, colors=None, nregime=None, size=8):
    """
    Add piecharts at each node based on probability vector

    Args:
        internal_vals: Array of dimensions [nchar+2, nnodes]. Identical to the
          format returned by anc_recon_cat
         colors: List of length = nchar. The colors for each character. Optional.
         nregime: Number of regimes. If given and colors = None, function will
         automatically color-code characters by regime. (NOT IMPLEMENTED)
    """
    nodes = list(treeplot.root.preiter())
    nchar = internal_vals.shape[1]-1
    if colors is None:
        pass
    for i,n in enumerate(nodes):
        add_pie(treeplot, n, values = list(internal_vals[i][:-2]), colors=colors, size=size)

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
        colors = [next(c) for v in range(nchar)]
    colors = np.array(colors)
    lc = LineCollection(segments, colors=colors[cols], lw=2, antialiaseds=[0])
    treeplot.add_collection(lc)

    leg_handles = [None] * nchar
    for i,char in enumerate(range(nchar)):
        leg_handles[i]=matplotlib.patches.Patch(color = colors[i],
                       label=str(i))
    treeplot.legend(handles=leg_handles)

def add_ancrecon(treeplot, liks, vis=True, width=2):
    """
    Plot ancestor reconstruction for a binary mk model
    """
    root = treeplot.root
    horz_seg_collections = [None] * (len(root)-1)
    horz_seg_colors = [None]*(len(root)-1)
    vert_seg_collections = [None] * (len(root)-1)
    vert_seg_colors = [None] * (len(root)-1)

    nchar = liks.shape[1]-2
    for i,n in enumerate(root.descendants()):
        n_lik = liks[i+1]
        par_lik = liks[n.parent.ni]
        n_col = twoS_colormaker(n_lik[:nchar])
        par_col = twoS_colormaker(par_lik[:nchar])

        n_coords = treeplot.n2c[n]
        par_coords = treeplot.n2c[n.parent]

        p1 = (n_coords.x, n_coords.y)
        p2 = (par_coords.x, n_coords.y)

        hsegs,hcols = gradient_segment_horz(p1,p2,n_col.rgb,par_col.rgb)

        horz_seg_collections[i] = hsegs
        horz_seg_colors[i] = hcols

        vert_seg_collections[i] = ([(par_coords.x,par_coords.y),
                                     (par_coords.x, n_coords.y)])
        vert_seg_colors[i] = (par_col.rgb)
    horz_seg_collections = [i for s in horz_seg_collections for i in s]
    horz_seg_colors = [i for s in horz_seg_colors for i in s]
    lc = LineCollection(horz_seg_collections + vert_seg_collections,
                        colors = horz_seg_colors + vert_seg_colors,
                        lw = width)
    lc.set_visible(vis)
    treeplot.add_collection(lc)

    treeplot.figure.canvas.draw_idle()

def add_ancrecon_hrm(treeplot, liks, vis=True, width=2):
    """
    Color branches on tree based on likelihood of being in a state
    based on ancestral state reconstruction of a two-character, two-regime
    hrm model.
    """
    root = treeplot.root
    horz_seg_collections = [None] * (len(root)-1)
    horz_seg_colors = [None]*(len(root)-1)
    vert_seg_collections = [None] * (len(root)-1)
    vert_seg_colors = [None] * (len(root)-1)

    nchar = liks.shape[1]-2

    for i,n in enumerate(root.descendants()):
        n_lik = liks[i+1]
        par_lik = liks[n.parent.ni]
        n_col = twoS_twoR_colormaker(n_lik[:nchar])
        par_col = twoS_twoR_colormaker(par_lik[:nchar])

        n_coords = treeplot.n2c[n]
        par_coords = treeplot.n2c[n.parent]

        p1 = (n_coords.x, n_coords.y)
        p2 = (par_coords.x, n_coords.y)

        hsegs,hcols = gradient_segment_horz(p1,p2,n_col.rgb,par_col.rgb)

        horz_seg_collections[i] = hsegs
        horz_seg_colors[i] = hcols

        vert_seg_collections[i] = ([(par_coords.x,par_coords.y),
                                     (par_coords.x, n_coords.y)])
        vert_seg_colors[i] = (par_col.rgb)
    horz_seg_collections = [i for s in horz_seg_collections for i in s]
    horz_seg_colors = [i for s in horz_seg_colors for i in s]
    lc = LineCollection(horz_seg_collections + vert_seg_collections,
                        colors = horz_seg_colors + vert_seg_colors,
                        lw = width)
    lc.set_visible(vis)
    treeplot.add_collection(lc)

    leg_ax = treeplot.figure.add_axes([0.3, 0.8, 0.1, 0.1])
    leg_ax.tick_params(which = "both",
                       bottom = "off",
                       labelbottom="off",
                       top = "off",
                       left = "off",
                       labelleft = "off",
                       right = "off")

    c1 = twoS_twoR_colormaker([1,0,0,0])
    c2 = twoS_twoR_colormaker([0,1,0,0])
    c3 = twoS_twoR_colormaker([0,0,1,0])
    c4 = twoS_twoR_colormaker([0,0,0,1])

    grid = np.array([[c1.rgb,c2.rgb],[c3.rgb,c4.rgb]])
    leg_ax.imshow(grid, interpolation="bicubic")
    treeplot.figure.canvas.draw_idle()

def add_hrm_hiddenstate_recon(treeplot, liks, nregime,vis=True, width=2, colors=None):
    """
    Color branches based on likelihood of being in hidden states.

    Args:
        liks (np.array): The output of anc_recon run as a hidden-rates reconstruction.
    """
    root = treeplot.root
    horz_seg_collections = [None] * (len(root)-1)
    horz_seg_colors = [None]*(len(root)-1)
    vert_seg_collections = [None] * (len(root)-1)
    vert_seg_colors = [None] * (len(root)-1)

    nchar = liks.shape[1]-2 # Liks has rows equal to nchar+2 (the last two rows are indices)
    nobschar = nchar//nregime

    if colors is None:
        c = _tango
        colors = [next(c) for v in range(nregime)]

    for i,n in enumerate(root.descendants()):
        n_lik = liks[i+1] # Add 1 to the index because the loop is skipping the root
        par_lik = liks[n.parent.ni]
        n_r1 = sum(n_lik[:nchar//2])# Likelihood of node being in regime 1
        p_r1 = sum(par_lik[:nchar//2])# Likelihood of node being in regime 1
        n_col = color_map(n_r1, colors[0], colors[1])
        par_col = color_map(p_r1, colors[0], colors[1])

        n_coords = treeplot.n2c[n]
        par_coords = treeplot.n2c[n.parent]

        p1 = (n_coords.x, n_coords.y)
        p2 = (par_coords.x, n_coords.y)

        hsegs,hcols = gradient_segment_horz(p1,p2,n_col,par_col)

        horz_seg_collections[i] = hsegs
        horz_seg_colors[i] = hcols

        vert_seg_collections[i] = ([(par_coords.x,par_coords.y),
                                     (par_coords.x, n_coords.y)])
        vert_seg_colors[i] = (par_col)
    horz_seg_collections = [i for s in horz_seg_collections for i in s]
    horz_seg_colors = [i for s in horz_seg_colors for i in s]
    lc = LineCollection(horz_seg_collections + vert_seg_collections,
                        colors = horz_seg_colors + vert_seg_colors,
                        lw = width)
    lc.set_visible(vis)
    treeplot.add_collection(lc)

    # leg_ax = treeplot.figure.add_axes([0.3, 0.8, 0.1, 0.1])
    # leg_ax.tick_params(which = "both",
    #                    bottom = "off",
    #                    labelbottom="off",
    #                    top = "off",
    #                    left = "off",
    #                    labelleft = "off",
    #                    right = "off")
    treeplot.figure.canvas.draw_idle()


def twoS_twoR_colormaker(lik):
    """
    Given node likelihood, return appropriate color

    State 0 corresponds to red, state 1 corresponds to blue
    Regime 1 corresponds to grey, regime 2 corresponds to highly saturated
    """
    s0 = sum([lik[0], lik[2]])
    s1 = sum([lik[1], lik[3]])

    r0 = sum([lik[0], lik[1]])
    r1 = sum([lik[2], lik[3]])

    lum = 0.30 + (r0*0.45)

    col = Color(rgb=(0.1,s0,s1))
    col.luminance = lum
    col.saturation = 0.35 + (r1*0.35)
    col.hue *= 1 - (0.2*r0)
    return col
def twoS_colormaker(lik):
    """
    Given node likelihood, return approrpiate color
    """
    s0 = lik[0]
    s1 = lik[1]
    col = Color(rgb=(s0,0.1,s1))
    col.luminance = 0.75
    return col
def gradient_segment_horz(p1, p2, c1, c2, width=4):
    """
    Create a horizontal segment that is filled with a gradient

    Args:
        p1 (tuple): XY coordinates of first point
        p2 (tuple): XY coordinates of second point *Y coord must be same as p1*
        c1 (tuple): RGB of color at point 1
        c2 (tuple): RGB of color at point 2
    Returns:
        list: list of segs and colors to be added to a LineCollection
    """
    nsegs = 255 # Number of sub-segments per segment (each segment gets its own color)
    seglen = (p2[0] - p1[0])/nsegs
    pos = list(zip(np.arange(p1[0], p2[0], seglen), [p1[1]]*nsegs))
    pos.append(p2)
    segs = [[pos[i],pos[i+1]] for i in range(nsegs)]

    cust_cm = LinearSegmentedColormap.from_list("cust_cm",[c1, c2])
    cols = [cust_cm(i) for i in range(1,256)]
    return [segs, cols]
def add_tipstates(treeplot, chars, nodes=None,colors=None, *args, **kwargs):
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in treeplot.root.leaves()]]
    if nodes is None:
        nodes = treeplot.root.leaves()
    if colors is None:
        colors = [ next(_tango) for char in set(chars) ]
    col_list = [ colors[i] for i in chars ]
    add_circles(treeplot, nodes, colors=col_list, size=6, *args, **kwargs)

def add_tree_heatmap(treeplot, locations, vis=True, color=(1,0,0)):
    """
    Plot how often tree coordinates appear in locations

    Args:
         locations (list): List of tuples where first item is node, second
           item is how far from the node's parent the location is.
    """
    color = matplotlib.colors.colorConverter.to_rgb(color)
    color = color + (.02,)
    nodes = zip(*locations)[0]
    nodes = [treeplot.root[n.ni] for n in nodes]
    distances = zip(*locations)[1]

    add_circles_branches(treeplot, nodes, distances, colors=color,vis=vis, size=5)

def add_mkmr_heatmap(treeplot, locations, vis=True, seglen=0.02):
    """
    Heatmap that shows which portions of the tree are most likely
    to contain a switchpoint

    To be used with the output from mk_multi_bayes.

    Args:
        locations (list): List of lists containing node and distance.
          The output from the switchpoint stochastic of mk_multi_bayes.
        seglen (float): The size of segments to break the tree into.
          MUST BE the same as the seglen used in mk_multi_bayes.
    """
    treelen = treeplot.root.max_tippath()
    seglen_px = seglen*treelen
    locations = [tuple([treeplot.root[x[0].ni],round(x[1],7)]) for x in locations]
    segmap = ivy.chars.mk_mr.tree_map(treeplot.root, seglen=seglen)
    segmap = [tuple([x[0],round(x[1],7)]) for x in segmap] # All possible segments to plot

    nrep = len(locations)

    rates = defaultdict(lambda: 0) # Setting heatmap densities
    for l in Counter(locations).items():
        rates[l[0]] = l[1]

    cmap = RdYlBu
    segments = []
    values = []
    # TODO: radial plot type

    for s in segmap:
        node = s[0]
        c = xy(treeplot, node)
        cp = xy(treeplot, node.parent)

        x0 = c[0] - s[1] # Start of segment
        x1 = x0 - seglen_px# End of segment
        if x1 < cp[0]:
            x1 = cp[0]
        y0 = y1 = c[1]

        segments.append(((x0,y0),(x1,y1)))
        values.append(rates[s]/nrep)

        if s[1] == 0.0 and not s[0].isleaf: # Draw vertical segments
            x0 = x1 = c[0]
            y0 = xy(treeplot, node.children[0])[1]
            y1 = xy(treeplot, node.children[-1])[1]
        segments.append(((x0,y0),(x1,y1)))
        values.append(rates[s]/nrep)

    lc = LineCollection(segments, cmap=RdYlBu, lw=2)
    lc.set_array(np.array(values))
    treeplot.add_collection(lc)
    lc.set_zorder(1)
    lc.set_visible(vis)
    treeplot.figure.canvas.draw_idle()


def color_blender_1(value, start, end):
    """
    Smooth transition between two values

    value (float): Percentage along color map
    start: starting value of color map
    end: ending value of color map
    """
    return start + (end-start)*value

def color_map(value, col1, col2):
    """
    Return RGB for value based on minimum and maximum colors
    """
    r = color_blender_1(value, col1[0], col2[0])
    g = color_blender_1(value, col1[1], col2[1])
    b = color_blender_1(value, col1[2], col2[2])

    return(r,g,b)
