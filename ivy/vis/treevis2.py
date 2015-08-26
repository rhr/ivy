"""
interactive viewers for trees, etc. using matplotlib

Re-written to have a layer-based API
"""
import sys, time, bisect, math, types, os, operator
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
from ivy.vis import events, layers
try:
    import Image
except ImportError:
    from PIL import Image
    
_tango = colors.tango()

class TreeFigure(object):
    """
    mpl Figure for plotting trees.
    
    Holds references to all layers:
        - Tree (the base layer: the Axes which all other layers are drawn on)
        - Node labels
        - Tip labels
        - Overview
        - Decorators 
        - Etc.
    
        The navigation toolbar at the bottom is provided by matplotlib
    (http://matplotlib.sf.net/users/navigation_toolbar.html).  Its
    pan/zoom button and zoom-rectangle button provide different modes
    of mouse interaction with the figure.  When neither of these
    buttons are checked, the default mouse bindings are as follows:

    * button 1 drag: select nodes - retrieve by calling fig.selected
    * button 3 drag: pan view
    * scroll up/down: zoom in/out
    * scroll up/down with Control key: zoom y-axis
    * scroll up/down with Shift key: zoom x-axis
    * scroll up/down with 'd' key: pan view up/down
    * scroll up/down with 'e' key: pan view left/right
    * click on overview will center the detail pane on that region

    Default keybindings:

    * t: zoom out to full extent
    * +/-: zoom in/out

    Useful attributes and methods (assume an instance named *fig*):

    * fig.root - the root node (see [Node methods])
    * fig.highlight(s) - highlight and trace nodes with substring *s*
    * fig.zoom_clade(anc) - zoom to view node *anc* and all its descendants
    * fig.toggle_overview() - toggle visibility of the overview pane
    * fig.toggle_branchlabels() - ditto for branch labels
    * fig.toggle_leaflabels() - ditto for leaf labels
    * fig.decorate(func) - decorate the tree with a function (see
      :ref:`decorating TreeFigures <decorating-trees>`)
    
    """
    def __init__(self, data, name=None, scaled=True, div=0.25,
                 branchlabels=True, leaflabels=True, xoff=0, yoff=0):
        self.name = name
        self.scaled = scaled
        self.branchlabels = branchlabels
        self.leaflabels = leaflabels
        self.xoff = xoff
        self.yoff = yoff
        if isinstance(data, tree.Node):
            root = data
        else:
            root = tree.read(data)
        self.root = root
        if not self.root:
            raise IOError, "cannot coerce data into tree.Node"
        self.name = self.name or root.treename        
        pars = SubplotParams(
            left=0, right=1, bottom=0.05, top=1, wspace=0.01
            )
        fig = pyplot.figure(subplotpars=pars, facecolor="white")
        events.connect_events(fig.canvas)
        self.figure = fig
        self.layers = []
        self.initialize_subplots()
    def initialize_subplots(self):
        tp = TreePlot(self.figure, 1, 2, 2, app=self, name=self.name,
                  scaled=self.scaled)
        tree = self.figure.add_subplot(tp)
        tree.set_root(self.root)
        tree.plot_tree()
        self.tree = tree
        self.layers.append(self.tree)
        self.set_positions()
        if self.leaflabels:
            self.addlayer(layers.addlabel, "leaf")
        if self.branchlabels:
            self.addlayer(layers.addlabel, "branch")
        
    def __get_selected_nodes(self):
        return list(self.tree.selected_nodes)

    def __set_selected_nodes(self, nodes):
        self.tree.select_nodes(nodes)

    def __del_selected_nodes(self):
        self.tree.select_nodes(None)
        
    selected = property(__get_selected_nodes,
                        __set_selected_nodes,
                        __del_selected_nodes)

    def on_nodes_selected(self, treeplot):
        pass

    @property
    def axes(self):
        return self.tree

    def picked(self, e):
        try:
            if e.mouseevent.button==1:
                s = e.artist.get_text()
                clipboard.copy(s)
                print s
                sys.stdout.flush()
        except:
            pass

    def ladderize(self, rev=False):
        """
        Ladderize and redraw the tree
        """
        self.root.ladderize(rev)
        self.redraw()

    def show(self):
        """
        Plot the figure in a new window
        """
        self.figure.show()

    def set_positions(self):
        p = self.tree
        height = 1.0-p.xoffset()
        w = 1.0
        p.set_position([0, p.xoffset(), w, height])
        self.figure.canvas.draw_idle()    
    def redraw(self):
        """
        Replot the figure and overview
        """
        self.tree.redraw()
        self.set_positions()
        self.figure.canvas.draw_idle()

    def find(self, x):
        """
        Find nodes

        Args:
            x (str): String to search
        Returns:
            list: A list of node objects found with the Node findall() method
        """
        return self.root.findall(x)
        
    def home(self):
        """
        Return plot to initial size and location.
        """
        self.tree.home()

    def zoom_clade(self, x):
        """
        Zoom to fit a node *x* and all its descendants in the view.

        Args:
            x: Node or str that matches the label of a node
        """
        if not isinstance(x, tree.Node):
            x = self.root[x]
        self.tree.zoom_clade(x)

    def zoom(self, factor=0.1):
        """Zoom both axes by *factor* (relative display size)."""
        self.tree.zoom(factor, factor)
        self.figure.canvas.draw_idle()

    def zx(self, factor=0.1):
        """Zoom x axis by *factor*."""
        self.tree.zoom(factor, 0)
        self.figure.canvas.draw_idle()

    def zy(self, factor=0.1):
        """Zoom y axis by *factor*."""
        self.tree.zoom(0, factor)
        self.figure.canvas.draw_idle()        
    def addlayer(self, func, *args):
        """
        Add a new layer. New layers include:
        
        - Labels
        - Overview
        - Dataplot
        - Decorations
        
        Args:
            func (function): Function that takes a TreePlot (self.tree)
              as input and returns (and draws) an Artist
        
        """
        self.layers.append(func(self.tree, *args))
        
         
class Tree(Axes):
    """
    Subclass for rendering trees
    """
    def __init__(self, fig, rect, *args, **kwargs):
        self.root = None
        self.app = kwargs.pop("app", None)
        self.support = kwargs.pop("support", 70.0)
        self.scaled = kwargs.pop("scaled", True)
        self._mark_named = kwargs.pop("mark_named", True)
        self.name = None
        self.branch_width = kwargs.pop("branch_width", 1)
        self.branch_color = kwargs.pop("branch_color", "black")
        self.interactive = kwargs.pop("interactive", True)
        ## if self.decorators:
        ##     print >> sys.stderr, "got %s decorators" % len(self.decorators)
        self.xoff = kwargs.pop("xoff", 0)
        self.yoff = kwargs.pop("yoff", 0)
        self.smooth_xpos = kwargs.pop("smooth_xpos", 0)
        Axes.__init__(self, fig, rect, *args, **kwargs)
        self.nleaves = 0
        self.pan_start = None
        self.selector = RectangleSelector(self, self.rectselect,
                                          useblit=True)
        self._active = False
        def f(e):
            if e.button != 1: return True
            else: return RectangleSelector.ignore(self.selector, e)
        self.selector.ignore = f
        self.xoffset_value = 0.05
        self.selected_nodes = set()
       # self.leaf_offset = 4
       # self.leaf_valign = "center"
       # self.leaf_halign = "left"
       # self.branch_offset = -5
       # self.branch_valign = "center"
       # self.branch_halign = "right"

        self.spines["top"].set_visible(False)
        self.spines["left"].set_visible(False)
        self.spines["right"].set_visible(False)
        self.xaxis.set_ticks_position("bottom")    
        
    def p2y(self):
        "Convert a single display point to y-units"
        transform = self.transData.inverted().transform
        return transform([0,1])[1] - transform([0,0])[1]

    def p2x(self):
        "Convert a single display point to x-units"
        transform = self.transData.inverted().transform
        return transform([0,0])[1] - transform([1,0])[1]
        
    def xoffset(self):
        """Space below x axis to show tick labels."""
        if self.scaled:
            return self.xoffset_value
        else:
            return 0        
    def set_scaled(self, scaled):
        flag = self.scaled != scaled
        self.scaled = scaled
        return flag        

    def circle_selected_nodes(self, color="green"):
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        get = self.n2c.get
        coords = filter(None, [ get(n) for n in self.selected_nodes ])
        x = [ c.x for c in coords ]
        y = [ c.y for c in coords ]
        if x and y:
            self.__selected_circled_patch = self.scatter(x, y, s=60, c=color,
                                                           zorder=100)
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        self.figure.canvas.draw_idle()
        
    def select_nodes(self, nodes=None, add=False):
        try:
            self.__selected_circled_patch.remove()
            self.figure.canvas.draw_idle()
        except:
            pass
        if add:
            if nodes:
                self.selected_nodes = self.selected_nodes | nodes
            if hasattr(self, "app") and self.app:
                self.app.on_nodes_selected(self)
            self.circle_selected_nodes()
        else:
            if nodes:
                self.selected_nodes = nodes
                if hasattr(self, "app") and self.app:
                    self.app.on_nodes_selected(self)
                self.circle_selected_nodes()
            else:
                self.selected_nodes = set()        
        
    def rectselect(self, e0, e1):
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        s = set()
        x0, x1 = sorted((e0.xdata, e1.xdata))
        y0, y1 = sorted((e0.ydata, e1.ydata))
        add = e0.key == 'shift'
        for n, c in self.n2c.items():
            if (x0 < c.x < x1) and (y0 < c.y < y1):
                s.add(n)
        self.select_nodes(nodes = s, add = add)
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        ## if s:
        ##     print "Selected:"
        ##     for n in s:
        ##         print " ", n        
        
    def picked(self, e):
        if hasattr(self, "app") and self.app:
            self.app.picked(e)

    def window2data(self, expandx=1.0, expandy=1.0):
        """
        return the data coordinates ((x0, y0),(x1, y1)) of the plot
        window, expanded by relative units of window size
        """
        bb = self.get_window_extent()
        bbx = bb.expanded(expandx, expandy)
        return self.transData.inverted().transform(bbx.get_points())        
    def get_visible_nodes(self, labeled_only=False):
        ## transform = self.transData.inverted().transform
        ## bb = self.get_window_extent()
        ## bbx = bb.expanded(1.1,1.1)
        ## ((x0, y0),(x1, y1)) = transform(bbx.get_points())
        ((x0, y0),(x1, y1)) = self.window2data(1.1, 1.1)
        #print "visible_nodes points", x0, x1, y0, y1

        if labeled_only:
            def f(v): return (y0 < v[0] < y1) and (v[2].label)
        else:
            def f(v): return (y0 < v[0] < y1)
        for y, x, n in filter(f, self.coords):
            yield (n, x, y)

    def zoom_cxy(self, x=0.1, y=0.1, cx=None, cy=None):
        """
        Zoom the x and y axes in by the specified proportion of the
        current view, with a fixed data point (cx, cy)
        """
        transform = self.transData.inverted().transform
        xlim = self.get_xlim(); xmid = sum(xlim)*0.5
        ylim = self.get_ylim(); ymid = sum(ylim)*0.5
        bb = self.get_window_extent()
        bbx = bb.expanded(1.0-x,1.0-y)
        points = transform(bbx.get_points())
        x0, x1 = points[:,0]; y0, y1 = points[:,1]
        deltax = xmid-x0; deltay = ymid-y0
        cx = cx or xmid; cy = cy or ymid
        xoff = (cx-xmid)*x
        self.set_xlim(xmid-deltax+xoff, xmid+deltax+xoff)
        yoff = (cy-ymid)*y
        self.set_ylim(ymid-deltay+yoff, ymid+deltay+yoff)
        self.adjust_xspine()
        self.figure.canvas.draw_idle() # Added draw_idle

    def zoom(self, x=0.1, y=0.1, cx=None, cy=None):
        """
        Zoom the x and y axes in by the specified proportion of the
        current view.
        """
        # get the function to convert display coordinates to data
        # coordinates
        transform = self.transData.inverted().transform
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        bb = self.get_window_extent()
        bbx = bb.expanded(1.0-x,1.0-y)
        points = transform(bbx.get_points())
        x0, x1 = points[:,0]; y0, y1 = points[:,1]
        deltax = x0 - xlim[0]; deltay = y0 - ylim[0]
        self.set_xlim(xlim[0]+deltax, xlim[1]-deltax)
        self.set_ylim(ylim[0]+deltay, ylim[1]-deltay)
        self.adjust_xspine()
        self.figure.canvas.draw_idle() # Added draw_idle

    def center_y(self, y):
        """
        Center the y-axis of the canvas on the given y value
        """
        ymin, ymax = self.get_ylim()
        yoff = (ymax - ymin) * 0.5
        self.set_ylim(y-yoff, y+yoff)
        self.adjust_xspine()

    def center_x(self, x, offset=0.3):
        """
        Center the x-axis of the canvas on the given x value
        """
        xmin, xmax = self.get_xlim()
        xspan = xmax - xmin
        xoff = xspan*0.5 + xspan*offset
        self.set_xlim(x-xoff, x+xoff)
        self.adjust_xspine()

    def center_node(self, node):
        """
        Center the canvas on the given node
        """
        c = self.n2c[node]
        y = c.y
        self.center_y(y)
        x = c.x
        self.center_x(x, 0.2)

    def find(self, s):
        """
        Find node(s) matching pattern s and zoom to node(s)
        """
        nodes = list(self.root.find(s))
        if nodes:
            self.zoom_nodes(nodes)

    def zoom_nodes(self, nodes, border=1.2):
        y0, y1 = self.get_ylim(); x0, x1 = self.get_xlim()
        y0 = max(0, y0); y1 = min(1, y1)

        n2c = self.n2c
        v = [ n2c[n] for n in nodes ]
        ymin = min([ c.y for c in v ])
        ymax = max([ c.y for c in v ])
        xmin = min([ c.x for c in v ])
        xmax = max([ c.x for c in v ])
        bb = Bbox(((xmin,ymin), (xmax, ymax)))

        # convert data coordinates to display coordinates
        transform = self.transData.transform
        disp_bb = [Bbox(transform(bb))]


        disp_bb = Bbox.union(disp_bb).expanded(border, border)

        # convert back to data coordinates
        points = self.transData.inverted().transform(disp_bb)
        x0, x1 = points[:,0]
        y0, y1 = points[:,1]
        self.set_xlim(x0, x1)
        self.set_ylim(y0, y1)

    def zoom_clade(self, anc, border=1.2):
        if anc.isleaf:
            self.center_node(anc)

        else:
            self.zoom_nodes(list(anc), border)
        self.figure.canvas.draw_idle() # Added draw_idle
            
    def ypp(self):
        y0, y1 = self.get_ylim()
        p0, p1 = self.transData.transform(((0, y0), (0, y1)))[:,1]
        return (y1-y0)/float(p1-p0)
        
    def redraw(self, home=False, layout=True):
        """
        Replot the tree
        """
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        self.cla()
        if layout:
            self.layout()
        self.plot_tree()
        if home:
            self.home()
        else:
            self.set_xlim(*xlim)
            self.set_ylim(*ylim)

    def set_name(self, name):
        self.name = name
        if name:
            at = AnchoredText(
                self.name, loc=2, frameon=True,
                prop=dict(size=12, weight="bold")
                )
            at.patch.set_linewidth(0)
            at.patch.set_facecolor("white")
            at.patch.set_alpha(0.6)
            self.add_artist(at)
            return at

    def _path_to_parent(self, node):
        """
        For use in drawing branches
        """
        c = self.n2c[node]; x = c.x; y = c.y
        pc = self.n2c[node.parent]; px = pc.x; py = pc.y
        M = Path.MOVETO; L = Path.LINETO
        verts = [(x, y), (px, y), (px, py)]
        codes = [M, L, L]
        return verts, codes
        ## return [PathPatch(Path(verts, codes), fill=False,
        ##                   linewidth=width or self.branch_width,
        ##                   edgecolor=color or self.branch_color)]


    def layout(self):
        self.n2c = cartesian(self.root, scaled=self.scaled, yunit=1.0,
                             smooth=self.smooth_xpos)
        self.labels = [ x.label for x in self.root ]
        for c in self.n2c.values():
            c.x += self.xoff; c.y += self.yoff
        sv = sorted([
            [c.y, c.x, n] for n, c in self.n2c.items()
            ])
        self.coords = sv#numpy.array(sv)

    def set_root(self, root):
        self.root = root
        self.leaves = root.leaves()
        self.nleaves = len(self.leaves)
        self.leaf_hsep = 1.0#/float(self.nleaves)

        for n in root.descendants():
            if n.length is None:
                self.scaled=False; break
        self.layout()

    def plot_tree(self, root=None, **kwargs):
        """
        Draw branches 
        """
        if root and not self.root:
            self.set_root(root)

        if self.interactive: pyplot.ioff()

        self.yaxis.set_visible(False)
        self.create_branch_artists()
        self.mark_named()
        ## self.home()

        self.set_name(self.name)
        self.adjust_xspine()

        if self.interactive: pyplot.ion()
        def fmt(x, pos=None):
            if x<0: return ""
            return ""
        #self.yaxis.set_major_formatter(FuncFormatter(fmt))

        return self

    def clade_dimensions(self):
        n2c = self.n2c
        d = {}
        def recurse(n, n2c, d):
            v = []
            for c in n.children:
                recurse(c, n2c, d)
                if c.isleaf:
                    x, y = n2c[c].point()
                    x0 = x1 = x; y0 = y1 = y
                else:
                    x0, x1, y0, y1 = d[c]
                v.append((x0, x1, y0, y1))
            if v:
                x0 = n2c[n].x
                x1 = max([ x[1] for x in v ])
                y0 = min([ x[2] for x in v ])
                y1 = max([ x[3] for x in v ])
                d[n] = (x0, x1, y0, y1)
        recurse(self.root, n2c, d)
        return d

    def clade_height_pixels(self):
        ypp = self.ypp()
        d = self.clade_dimensions()
        h = {}
        for n, (x0, x1, y0, y1) in d.items():
            h[n] = (y1-y0)/ypp
        return h

    def _decimate_nodes(self, n=500):
        leaves = self.leaves
        nleaves = len(leaves)
        if nleaves > n:
            indices = numpy.linspace(0, nleaves-1, n).astype(int)
            leaves = [ leaves[i] for i in indices ]
            return set(list(chain.from_iterable([ list(x.rootpath())
                                                  for x in leaves ])))
        else:
            return self.root

    def create_branch_artists(self):
        """
        Use MPL Paths to draw branches
        """
        ## patches = []
        verts = []; codes = []
        for node in self.root.descendants():
            v, c = self._path_to_parent(node)
            verts.extend(v); codes.extend(c)
        self.branchpatch = PathPatch(
            Path(verts, codes), fill=False,
            linewidth=self.branch_width,
            edgecolor=self.branch_color
            )
        self.add_patch(self.branchpatch)

    def adjust_xspine(self):
        v = sorted([ c.x for c in self.n2c.values() ])
        try:
            self.spines["bottom"].set_bounds(v[0],v[-1])
        except AttributeError:
            pass
        for t,n,s in self.xaxis.iter_ticks():
            if (n > v[-1]) or (n < v[0]):
                t.set_visible(False)

    def mark_named(self):
        if self._mark_named:
            n2c = self.n2c
            cv = [ c for n, c in n2c.items() if n.label and (not n.isleaf) ]
            x = [ c.x for c in cv ]
            y = [ c.y for c in cv ]
            if x and y:
                self.scatter(x, y, s=5, color='black')

    def home(self):
        td = self.transData
        trans = td.inverted().transform
        xmax = xmin = ymax = ymin = 0

        v = self.n2c.values()
        ymin = min([ c.y for c in v ])
        ymax = max([ c.y for c in v ])
        xmin = min(xmin, min([ c.x for c in v ]))
        xmax = max(xmax, max([ c.x for c in v ]))
        xspan = xmax - xmin; xpad = xspan*0.05
        yspan = ymax - ymin; ypad = yspan*0.05
        self.set_xlim(xmin-xpad, xmax+xpad*2)
        self.set_ylim(ymin-ypad, ymax+ypad)
        self.adjust_xspine()
        self.figure.canvas.draw_idle() # Warning: Had to add this line for this to work. Still don't know why.

    def scroll(self, x, y):
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        xd = (x1-x0)*x
        yd = (y1-y0)*y
        self.set_xlim(x0+xd, x1+xd)
        self.set_ylim(y0+yd, y1+yd)
        self.adjust_xspine()



TreePlot = subplot_class_factory(Tree)







        
