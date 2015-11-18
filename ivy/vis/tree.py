"""
interactive viewers for trees, etc. using matplotlib
"""
import sys, time, bisect, math, types, os, operator
from collections import defaultdict
from itertools import chain
from pprint import pprint
from .. import tree, bipart
from ..layout import cartesian
from ..storage import Storage
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
import symbols, colors
import hardcopy as HC
try:
    import Image
except ImportError:
    from PIL import Image

#matplotlib.rcParams['path.simplify'] = False

_tango = colors.tango()
class TreeFigure(object):
    """
    Window for showing a single tree, optionally with split overview
    and detail panes.

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
                 branchlabels=True, leaflabels=True, mark_named=True,
                 highlight_support=True, xoff=0, yoff=0,
                 overview=True, radial=False):
        self.overview = None
        self.overview_width = div
        self.dataplot = None
        self.dataplot_width = 0.25
        self.name = name
        self.scaled = scaled
        self.branchlabels = branchlabels
        self.leaflabels = leaflabels
        self.mark_named = mark_named
        self.xoff = xoff
        self.yoff = yoff
        self.radial = radial
        if radial:
            self.leaflabels = False
        self.highlighted = set()
        self.highlight_support = highlight_support
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
        connect_events(fig.canvas)
        self.figure = fig
        self.initialize_subplots(overview)
        self.home()

    def initialize_subplots(self, overview=True):
        if not self.radial:
            tp = TreePlot(self.figure, 1, 2, 2, app=self, name=self.name,
                          scaled=self.scaled, branchlabels=self.branchlabels,
                          highlight_support=self.highlight_support,
                          leaflabels=self.leaflabels,
                          mark_named=self.mark_named)
            detail = self.figure.add_subplot(tp)
            detail.set_root(self.root)
            detail.plot_tree()
            self.detail = detail
            tp = OverviewTreePlot(
                self.figure, 121, app=self, scaled=self.scaled,
                branchlabels=False, leaflabels=False,
                mark_named=self.mark_named,
                highlight_support=self.highlight_support,
                target=self.detail
                )
            ov = self.figure.add_subplot(tp)
            ov.set_root(self.root)
            ov.plot_tree()
            self.overview = ov
            if not overview:
                self.toggle_overview(False)
            self.set_positions()

            if self.detail.nleaves < 50:
                self.toggle_overview(False)
        else:
            tp = RadialTreePlot(
                self.figure, 111, app=self, name=self.name,
                scaled=self.scaled, branchlabels=self.branchlabels,
                highlight_support=self.highlight_support,
                leaflabels=self.leaflabels, mark_named=self.mark_named
                )
            ax2 = self.figure.add_subplot(tp)
            ax2.set_root(self.root)
            ax2.plot_tree()
            self.detail = ax2

    def __get_selected_nodes(self):
        return list(self.detail.selected_nodes)

    def __set_selected_nodes(self, nodes):
        self.detail.select_nodes(nodes)

    def __del_selected_nodes(self):
        self.detail.select_nodes(None)

    selected = property(__get_selected_nodes,
                        __set_selected_nodes,
                        __del_selected_nodes)

    ## def selected_nodes(self):
    ##     return self.detail.selected_nodes

    @property
    def axes(self):
        return self.detail

    def add(self, data, name=None, support=70,
            branchlabels=False, leaflabels=True):
        """
        Add a new tree in a new window

        Args:
            data: A node object or tree file.
            name (str): Name of the plot. Defaults to None
            branchlabels (bool): Whether or not to draw branch labels.
              Defaults to False
            leaflabels (bool): Whether or not to draw leaf labels.
              Defaults to True
        """
        newfig = MultiTreeFigure()
        ## newfig.add(self.root, name=self.name, support=self.support,
        ##            branchlabels=self.branchlabels)
        newfig.add(data, name=name, support=support,
                   branchlabels=branchlabels,
                   leaflabels=leaflabels)
        return newfig

    def toggle_leaflabels(self):
        """
        Toggle leaf labels and redraw tree
        """
        self.leaflabels = not self.leaflabels
        self.detail.leaflabels = self.leaflabels
        self.redraw()

    def toggle_branchlabels(self):
        """
        Toggle branch labels and redraw tree
        """
        self.branchlabels = not self.branchlabels
        self.detail.branchlabels = self.branchlabels
        self.redraw()

    def toggle_overview(self, val=None):
        """
        Toggle overview
        """
        if val is None:
            if self.overview.get_visible():
                self.overview.set_visible(False)
                self.overview_width = 0.001
            else:
                self.overview.set_visible(True)
                self.overview_width = 0.25
        elif val:
            self.overview.set_visible(True)
            self.overview_width = val
        else:
            self.overview.set_visible(False)
            self.overview_width = 0.001
        self.set_positions()

    def set_scaled(self, scaled):
        """
        RR: Using this method gives the error:
            redraw takes exactly 1 argument(2 given)-CZ
        Define whether or not the tree is scaled and redraw tree

        Args:
            scaled (bool): Whether or not the tree is scaled.
        """
        for p in self.overview, self.detail:
            p.redraw(p.set_scaled(scaled))
        self.set_positions()

    def on_nodes_selected(self, treeplot):
        pass

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
        ov = self.overview
        p = self.detail
        dp = self.dataplot
        height = 1.0-p.xoffset()
        if ov:
            box = [0, p.xoffset(), self.overview_width, height]
            ov.set_position(box)
        w = 1.0
        if ov:
            w -= self.overview_width
        if dp:
            w -= self.dataplot_width
        p.set_position([self.overview_width, p.xoffset(), w, height])
        if dp:
            box = [1.0-self.dataplot_width, p.xoffset(),
                   self.dataplot_width, height]
            dp.set_position(box)
        self.figure.canvas.draw_idle()

    ## def div(self, v=0.3):
    ##     assert 0 <= v < 1
    ##     self.overview_width = v
    ##     self.set_positions()
    ##     self.figure.canvas.draw_idle()

    def add_dataplot(self):
        """
        Add new plot to the side of existing plot
        """
        np = 3 if self.overview else 2
        if self.dataplot:
            self.figure.delaxes(self.dataplot)
        self.dataplot = self.figure.add_subplot(1, np, np, sharey=self.detail)
        # left, bottom, width, height (proportions)
        dleft, dbottom, dwidth, dheight = self.detail.get_position().bounds
        # give the dataplot one-quarter the width of the detail axes
        w = dwidth * 0.25
        self.detail.set_position([dleft, dbottom, dwidth-w, dheight])
        self.dataplot.set_position([1-w, dbottom, w, dheight])
        self.dataplot.xaxis.set_visible(False)
        self.dataplot.yaxis.set_visible(False)
        for x in self.dataplot.spines.values():
            x.set_visible(False)
        self.figure.canvas.draw_idle()
        return self.dataplot

    def redraw(self):
        """
        Replot the figure and overview
        """
        self.detail.redraw()
        if self.overview: self.overview.redraw()
        self.highlight()
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

    def hlines(self, nodes, width=5, color="red", xoff=0, yoff=0):
        """
        Highlight nodes

        Args:
            nodes (list): A list of node objects
            width (float): Width of highlighted lines. Defaults to 5
            color (str): Color of highlighted lines. Defaults to red
            xoff (float): Number of units to offset lines by. Defaults to 0
            yoff (float): Number of units to offset lines by. Defaults to 0
        """
        self.overview.hlines(nodes, width=width, color=color,
                             xoff=xoff, yoff=yoff)
        self.detail.hlines(nodes, width=width, color=color,
                           xoff=xoff, yoff=yoff)

    def highlight(self, x=None, width=5, color="red"):
        """
        Highlight nodes

        Args:
            x: Str or list of Strs or Node or list of Nodes
            width (float): Width of highlighted lines. Defaults to 5
            color (str): Color of highlighted lines. Defaults to red
        """
        if x:
            nodes = set()
            if type(x) in types.StringTypes:
                nodes = self.root.findall(x)
            elif isinstance(x, tree.Node):
                nodes = set(x)
            else:
                for n in x:
                    if type(n) in types.StringTypes:
                        found = self.root.findall(n)
                        if found:
                            nodes |= set(found)
                    elif isinstance(n, tree.Node):
                        nodes.add(n)

            self.highlighted = nodes
        else:
            self.highlighted = set()
        if self.overview:
            self.overview.highlight(self.highlighted, width=width, color=color)
        self.detail.highlight(self.highlighted, width=width, color=color)
        self.figure.canvas.draw_idle()

    def home(self):
        """
        Return plot to initial size and location.
        """
        if self.overview: self.overview.home()
        self.detail.home()

    def zoom_clade(self, x):
        """
        Zoom to fit a node *x* and all its descendants in the view.

        Args:
            x: Node or str that matches the label of a node
        """
        if not isinstance(x, tree.Node):
            x = self.root[x]
        self.detail.zoom_clade(x)

    def zoom(self, factor=0.1):
        """Zoom both axes by *factor* (relative display size)."""
        self.detail.zoom(factor, factor)
        self.figure.canvas.draw_idle()

    def zx(self, factor=0.1):
        """Zoom x axis by *factor*."""
        self.detail.zoom(factor, 0)
        self.figure.canvas.draw_idle()

    def zy(self, factor=0.1):
        """Zoom y axis by *factor*."""
        self.detail.zoom(0, factor)
        self.figure.canvas.draw_idle()

    def decorate(self, func, *args, **kwargs):
        """
        Decorate the tree.

        Args:
            func (function): A function that takes a TreePlot instance as the
              first parameter, and *args* and *kwargs* as additional
              parameters.  It adds boxes, circles, etc to the TreePlot.

        Notes:
            If *kwargs* contains the key-value pair ('store', *name*),
            then the function is stored as *name* and re-called every time
            the TreePlot is redrawn, i.e., the decoration is persistent.
            Use ``rmdec(name)`` to remove the decorator from the treeplot.
        """
        self.detail.decorate(func, *args, **kwargs)

    def rmdec(self, name):
        "Remove the decoration 'name'."
        self.detail.rmdec(name)
        ## if name in self.detail.decorators:
        ##     del self.detail.decorators[name]

    def cbar(self, node, width=6, color='blue', mrca = True):
        pass
        # self.axes.cbar(nodes = node, width = width, color = color, mrca = mrca)

    def unclutter(self, *args):
        self.detail.unclutter()

    def trace_branches(self, nodes, width=4, color="blue"):
        """
        RR: What is the difference between this and highlight? -CZ
        """
        for p in self.overview, self.detail:
            p.trace_branches(nodes, width, color)

    def plot_continuous(self, *args, **kwargs):
        self.detail.plot_continuous(*args, **kwargs)

    def hardcopy(self, fname=None, relwidth=None, leafpad=1.5):
        if not relwidth:
            bbox = self.detail.get_tightbbox(self.figure.canvas.get_renderer())
            relwidth = bbox.width/bbox.height
        f = self.detail.hardcopy(
            relwidth=relwidth,
            leafpad=leafpad
            )
        f.axes.home()
        #f.axes.set_xlim(*self.detail.get_xlim())
        #f.axes.set_ylim(*self.detail.get_ylim())
        if fname:
            f.savefig(fname)
        return f

    def select_nodes(self, nodes=None):
        """
        Select nodes on the plot

        Args:
            nodes: A node or list of ndoes
        Notes:
            If only one node is given, all of the node's ancestors are
            also selected. If a list of nodes is given (even if it has only
            one node), only the given node(s) are selected.
        """
        self.detail.select_nodes(nodes)

    def decorate(self, func, *args, **kwargs): # RR: is this repeated from above? -CZ
        self.detail.decorate(func, *args, **kwargs)

    ## def dataplot(self):
    ##     ax = self.figure.add_subplot(133, sharey=self.detail)
    ##     ax.yaxis.set_visible(False)
    ##     self.dataplot = ax
    ##     return ax

    def attach_alignment(self, aln, overview=True):
        "leaf labels expected to be sequence ids"
        from Bio.Align import MultipleSeqAlignment
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from Bio.Alphabet import IUPAC
        from alignment import AlignmentFigure, AlignmentPlot
        if not isinstance(aln, MultipleSeqAlignment):
            from .. import align
            aln = align.read(aln)
        d = dict([ (x.id,x) for x in aln ])
        emptyseq = Seq('-'*aln.get_alignment_length(),
                       alphabet=IUPAC.ambiguous_dna)
        aln = MultipleSeqAlignment(
            [ d.get(x.label) or SeqRecord(emptyseq, id=x.label)
              for x in self.root.leaves() ]
            )
        self.aln = aln
        p = AlignmentPlot(self.figure, 133, aln=aln, app=self,
                          sharey=self.detail, showy=False)
        self.alnplot = Storage()
        self.alnplot.detail = self.figure.add_subplot(p)
        detail = self.alnplot.detail
        detail.plot_aln()
        if overview:
            self.alnplot.overview = inset_axes(
                detail, width="30%", height="20%", loc=1
                )
            overview = self.alnplot.overview
            overview.xaxis.set_major_locator(NullLocator())
            overview.yaxis.set_major_locator(NullLocator())
            overview.imshow(
                detail.array, interpolation='nearest', aspect='auto',
                origin='lower'
                )
            rect = UpdatingRect(
                [0,0], 0, 0, facecolor='black', edgecolor='cyan', alpha=0.5
                )
            overview.zoomrect = rect
            rect.target = detail
            detail.callbacks.connect('xlim_changed', rect)
            detail.callbacks.connect('ylim_changed', rect)
            overview.add_patch(rect)
            rect(overview)
        self.toggle_overview(False)
        xoff = self.detail.xoffset()
        self.detail.set_position([0, xoff, 0.3, 1.0-xoff])
        p.set_position([0.3, xoff, 0.7, 1.0-xoff])


class MultiTreeFigure(object):
    """
    Window for showing multiple trees side-by-side.

    TODO: document this
    """
    def __init__(self, trees=None, name=None, support=70,
                 scaled=True, branchlabels=False, radial=False):
        """
        *trees* are assumed to be objects suitable for passing to
        ivy.tree.read()
        """
        self.root = []
        self.name = name
        self.name2plot = {}
        self.plot = []
        self.scaled = scaled
        self.branchlabels = branchlabels
        self.radial = radial
        self.highlighted = set()
        self.divs = []
        pars = SubplotParams(
            left=0, right=1, bottom=0.05, top=1, wspace=0.04
            )
        fig = pyplot.figure(subplotpars=pars)
        connect_events(fig.canvas)
        self.figure = fig

        for x in trees or []:
            self.add(x, support=support, scaled=scaled,
                     branchlabels=branchlabels)

    def on_nodes_selected(self, treeplot):
        pass

    def clear(self):
        self.root = []
        self.name2plot = {}
        self.highlighted = set()
        self.divs = []
        self.figure.clf()

    def picked(self, e):
        try:
            if e.mouseevent.button==1:
                print e.artist.get_text()
                sys.stdout.flush()
        except:
            pass

    def getplot(self, x):
        p = None
        try:
            i = self.root.index(x)
            return self.plot[i]
        except ValueError:
            return self.name2plot.get(x)

    def add(self, data, name=None, support=70, scaled=True,
            branchlabels=False, leaflabels=True):
        root = None
        if isinstance(data, tree.Node):
            root = data
        else:
            root = tree.read(data)
        if not root:
            raise IOError, "cannot coerce data into tree.Node"

        name = name or root.treename
        self.root.append(root)

        fig = self.figure
        N = len(self.plot)+1
        for i, p in enumerate(self.plot):
            p.change_geometry(1, N, i+1)
        plt = TreePlot(fig, 1, N, N, app=self, name=name, support=support,
                       scaled=scaled, branchlabels=branchlabels,
                       leaflabels=leaflabels)
        p = fig.add_subplot(plt)
        p.set_root(root)
        p.plot_tree()
        p.index = N-1
        self.plot.append(p)
        if name:
            assert name not in self.name2plot
            self.name2plot[name] = p

        ## global IP
        ## if IP:
        ##     def f(shell, s):
        ##         self.highlight(s)
        ##         return sorted([ x.label for x in self.highlighted ])
        ##     IP.expose_magic("highlight", f)
        ##     def f(shell, s):
        ##         self.root.ladderize()
        ##         self.redraw()
        ##     IP.expose_magic("ladderize", f)
        ##     def f(shell, s):
        ##         self.show()
        ##     IP.expose_magic("show", f)
        ##     def f(shell, s):
        ##         self.redraw()
        ##     IP.expose_magic("redraw", f)
        return p

    def show(self):
        self.figure.show()

    def redraw(self):
        for p in self.plot:
            p.redraw()
        self.figure.canvas.draw_idle()

    def ladderize(self, reverse=False):
        for n in self.root:
            n.ladderize(reverse)
        self.redraw()

    def highlight(self, s=None, add=False, width=5, color="red"):
        """
        Highlight nodes

        Args:
            s: Str or list of Strs or Node or list of Nodes
            add (bool): Whether to add to existing highlighted nodes or
              overwrite them.
            width (float): Width of highlighted lines. Defaults to 5
            color (str): Color of highlighted lines. Defaults to red
        """
        if not s:
            self.highlighted = set()
        if not add:
            self.highlighted = set()

        nodesets = [ p.root.findall(s) for p in self.plot ]

        for nodes, p in zip(nodesets, self.plot):
            if nodes:
                p.highlight(nodes, width=width, color=color)
            else:
                p.highlight()

        self.highlighted = nodesets
        self.figure.canvas.draw_idle()

        ##     for root in self.root:
        ##         for node in root.iternodes():
        ##             if node.label and (s in node.label):
        ##                 self.highlighted.add(node)
        ## self.highlight()

    def home(self):
        for p in self.plot: p.home()


def connect_events(canvas):
    mpl_connect = canvas.mpl_connect
    mpl_connect("button_press_event", onclick)
    mpl_connect("button_release_event", onbuttonrelease)
    mpl_connect("scroll_event", onscroll)
    mpl_connect("pick_event", onpick)
    mpl_connect("motion_notify_event", ondrag)
    mpl_connect("key_press_event", onkeypress)
    mpl_connect("axes_enter_event", axes_enter)
    mpl_connect("axes_leave_event", axes_leave)

class UpdatingRect(Rectangle):
    def __call__(self, p):
        self.set_bounds(*p.viewLim.bounds)
        p.figure.canvas.draw_idle()

class Tree(Axes):
    """
    matplotlib.axes.Axes subclass for rendering trees.
    """
    def __init__(self, fig, rect, *args, **kwargs):
        self.root = None
        self.app = kwargs.pop("app", None)
        self.support = kwargs.pop("support", 70.0)
        self.scaled = kwargs.pop("scaled", True)
        self.leaflabels = kwargs.pop("leaflabels", True)
        self.branchlabels = kwargs.pop("branchlabels", True)
        self._mark_named = kwargs.pop("mark_named", True)
        self.name = None
        self.leaf_fontsize = kwargs.pop("leaf_fontsize", 10)
        self.branch_fontsize = kwargs.pop("branch_fontsize", 10)
        self.branch_width = kwargs.pop("branch_width", 1)
        self.branch_color = kwargs.pop("branch_color", "black")
        self.interactive = kwargs.pop("interactive", True)
        self.decorators = kwargs.pop("decorators", [])
        ## if self.decorators:
        ##     print >> sys.stderr, "got %s decorators" % len(self.decorators)
        self.xoff = kwargs.pop("xoff", 0)
        self.yoff = kwargs.pop("yoff", 0)
        self.highlight_support = kwargs.pop("highlight_support", True)
        self.smooth_xpos = kwargs.pop("smooth_xpos", 0)
        Axes.__init__(self, fig, rect, *args, **kwargs)
        self.nleaves = 0
        self.highlighted = None
        self.highlightpatch = None
        self.pan_start = None
        if not self.decorators:
            self.decorators = [
                ("__selected_nodes__", (Tree.highlight_selected_nodes, [], {}))
                ]
        self.name2dec = dict([ (x[0], i) for i, x in
                               enumerate(self.decorators) ])
        self._active = False

        if self.interactive:
            self.callbacks.connect("ylim_changed", self.draw_labels)
        self.selector = RectangleSelector(self, self.rectselect,
                                          useblit=True)
        def f(e):
            if e.button != 1: return True
            else: return RectangleSelector.ignore(self.selector, e)
        self.selector.ignore = f
        self.xoffset_value = 0.05
        self.selected_nodes = set()
        self.leaf_offset = 4
        self.leaf_valign = "center"
        self.leaf_halign = "left"
        self.branch_offset = -5
        self.branch_valign = "center"
        self.branch_halign = "right"

        self.spines["top"].set_visible(False)
        self.spines["left"].set_visible(False)
        self.spines["right"].set_visible(False)
        self.xaxis.set_ticks_position("bottom")

    def p2y(self):
        "Convert a single display point to y-units"
        transform = self.transData.inverted().transform
        return transform([0,1])[1] - transform([0,0])[1]

    def p2x(self):
        "Convert a single display point to y-units"
        transform = self.transData.inverted().transform
        return transform([0,0])[1] - transform([1,0])[1]

    def decorate(self, func, *args, **kwargs):
        """
        Decorate the tree with function *func*.  If *kwargs* contains
        the key-value pair ('store', *name*), the decorator function
        is stored in self.decorators and called upon every redraw.
        """
        name = kwargs.pop("store", None)
        if name:
            if name in self.name2dec:
                i = self.name2dec[name]
                self.decorators[i] = (name, (func, args, kwargs))
            else:
                self.decorators.append((name, (func, args, kwargs)))
                self.name2dec = dict([ (x[0], i) for i, x in
                                       enumerate(self.decorators) ])

        func(self, *args, **kwargs)

    def rmdec(self, name):
        if name in self.name2dec:
            i = self.name2dec[name]
            del self.decorators[i]
            self.name2dec = dict([ (x[0], i) for i, x in
                                   enumerate(self.decorators) ])


    def flip(self):
        """
        Reverse the direction of the x-axis.
        """
        self.leaf_offset *= -1
        self.branch_offset *= -1
        ha = self.leaf_halign
        self.leaf_halign = "right" if ha == "left" else "left"
        ha = self.branch_halign
        self.branch_halign = "right" if ha == "left" else "left"
        self.invert_xaxis()
        self.redraw()

    def xoffset(self):
        """Space below x axis to show tick labels."""
        if self.scaled:
            return self.xoffset_value
        else:
            return 0

    def save_newick(self, filename):
        """
        Save tree as a newick file.

        Args:
            filename (str): Path to file.

        """
        if os.path.exists(filename):
            s = raw_input("File %s exists, enter 'y' to overwrite ").strip()
            if (s and s.lower() != 'y') or (not s):
                return
        import newick
        f = file(filename, "w")
        f.write(newick.string(self.root))
        f.close()

    def set_scaled(self, scaled):
        flag = self.scaled != scaled
        self.scaled = scaled
        return flag

    def cbar(self, nodes, color=None, label=None, x=None, width=8, xoff=10,
             showlabel=True, mrca=True):
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
            nodes: Node or list of nodes
            color (str): Color of the bar. Optional, defaults to None.
            label (str): Optional label for bar. If None, the clade's
              label is used instead. Defaults to None.
            width (float): Width of bar
            xoff (float): Offset from label to bar
            showlabel (bool): Whether or not to draw the label
            mrca: RR: Not quite sure what this does -CZ

        """
        xlim = self.get_xlim(); ylim = self.get_ylim()
        if color is None: color = _tango.next()
        transform = self.transData.inverted().transform

        if mrca:
            if isinstance(nodes, tree.Node):
                spec = nodes
            elif type(nodes) in types.StringTypes:
                spec = self.root.get(nodes)
            else:
                spec = self.root.mrca(nodes)

            assert spec in self.root
            label = label or spec.label
            leaves = spec.leaves()

        else:
            leaves = nodes

        n2c = self.n2c

        y = sorted([ n2c[n].y for n in leaves ])
        ymin = y[0]; ymax = y[-1]; y = (ymax+ymin)*0.5

        if x is None:
            x = max([ n2c[n].x for n in leaves ])
            _x = 0
            for lf in leaves:
                txt = self.node2label.get(lf)
                if txt and txt.get_visible():
                    _x = max(_x, transform(txt.get_window_extent())[1,0])
            if _x > x: x = _x

        v = sorted(list(transform(((0,0),(xoff,0)))[:,0]))
        xoff = v[1]-v[0]
        x += xoff

        Axes.plot(self, [x,x], [ymin, ymax], '-', linewidth=width, color=color)

        if showlabel and label:
            xo = self.leaf_offset
            if xo > 0:
                xo += width*0.5
            else:
                xo -= width*0.5
            txt = self.annotate(
                label,
                xy=(x, y),
                xytext=(xo, 0),
                textcoords="offset points",
                verticalalignment=self.leaf_valign,
                horizontalalignment=self.leaf_halign,
                fontsize=self.leaf_fontsize,
                clip_on=True,
                picker=False
                )

        self.set_xlim(xlim); self.set_ylim(ylim)

    def anctrace(self, anc, descendants=None, width=4, color="blue"):
        """
        RR: This function gives me a 'list index out of range' error
        when I try to use it -CZ
        """
        if not descendants:
            descendants = anc.leaves()
        else:
            for d in descendants:
                assert d in anc

        nodes = []
        for d in descendants:
            v = d.rootpath(anc)
            if v:
                nodes.extend(v)
        nodes = set(nodes)
        nodes.remove(anc)
        self.trace_branches(nodes, width, color)

    def trace_branches(self, nodes, width=4, color="blue"):
        n2c = self.n2c
        M = Path.MOVETO; L = Path.LINETO
        verts = []
        codes = []
        for c, pc in [ (n2c[x], n2c[x.parent]) for x in nodes
                       if (x in n2c) and x.parent ]:
            x = c.x; y = c.y
            px = pc.x; py = pc.y
            verts.append((x, y)); codes.append(M)
            verts.append((px, y)); codes.append(L)
            verts.append((px, py)); codes.append(L)
        px, py = verts[-1]
        verts.append((px, py)); codes.append(M)

        p = PathPatch(Path(verts, codes), fill=False,
                      linewidth=width, edgecolor=color)
        self.add_patch(p)
        self.figure.canvas.draw_idle()
        return p

    def highlight_selected_nodes(self, color="green"):
        xlim = self.get_xlim()
        ylim = self.get_ylim()
        get = self.n2c.get
        coords = filter(None, [ get(n) for n in self.selected_nodes ])
        x = [ c.x for c in coords ]
        y = [ c.y for c in coords ]
        if x and y:
            self.__selected_highlight_patch = self.scatter(x, y, s=60, c=color,
                                                           zorder=100)
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        self.figure.canvas.draw_idle()

    def select_nodes(self, nodes=None, add=False):
        try:
            self.__selected_highlight_patch.remove()
            self.figure.canvas.draw_idle()
        except:
            pass
        if add:
            if nodes:
                self.selected_nodes = self.selected_nodes | nodes
            if hasattr(self, "app") and self.app:
                self.app.on_nodes_selected(self)
            self.highlight_selected_nodes()
        else:
            if nodes:
                self.selected_nodes = nodes
                if hasattr(self, "app") and self.app:
                    self.app.on_nodes_selected(self)
                self.highlight_selected_nodes()
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
            def f(v): return (y0 < v[0] < y1) and (v[2] in self.node2label)
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

    def do_highlight_support(self):
        """
        TODO: reconfigure this, insert into self.decorators
        """
        if self.support:
            lim = float(self.support)

        M = Path.MOVETO; L = Path.LINETO

        verts = []; codes = []
        segments = []
        def f(n):
            if n.isleaf or not n.parent: return False
            try: return float(n.label) >= lim
            except:
                try: return float(n.support) >= lim
                except: pass
            return False

        for node, coords in [ x for x in self.n2c.items() if f(x[0]) ]:
            x = coords.x; y = coords.y
            p = node.parent
            pcoords = self.n2c[p]
            px = pcoords.x; py = y
            if self.app and self.app.radial:
                pc = self.n2c[node.parent]; theta2 = pc.angle
                px = math.cos(math.radians(coords.angle))*pc.depth
                py = math.sin(math.radians(coords.angle))*pc.depth

            ## segments.append([(x, y),(px, y)])
            verts.append((x,y)); codes.append(M)
            verts.append((px,py)); codes.append(L)

        if verts:
            patch = PathPatch(Path(verts, codes), fill=False,
                              linewidth=3, edgecolor='black')
            self.add_patch(patch)

            ## self.add_artist(Line2D(
            ##     [x,px], [y,py], lw=3, solid_capstyle="butt", color="black"
            ##     ))

    def hl(self, s):
        nodes = self.root.findall(s)
        if nodes:
            self.highlight(nodes)

    def hlines(self, nodes, width=5, color="red", xoff=0, yoff=0):
        offset = IdentityTransform()
        segs = []; w = []; o = []
        for n in filter(lambda x:x.parent, nodes):
            c = self.n2c[n]; p = self.n2c[n.parent]
            segs.append(((p.x,c.y),(c.x,c.y)))
            w.append(width); o.append((xoff,yoff))
        lc = LineCollection(segs, linewidths=w, transOffset=offset, offsets=o)
        lc.set_color(color)
        Axes.add_collection(self, lc)
        ## self.drawstack.append(("hlines", [nodes], dict(width=width,
        ##                                                color=color,
        ##                                                xoff=xoff,
        ##                                                yoff=yoff)))
        self.figure.canvas.draw_idle()
        return lc
        
        
    

    def hardcopy(self, relwidth=0.5, leafpad=1.5):
        p = HC.TreeFigure(self.root, relwidth=relwidth, leafpad=leafpad,
                          name=self.name, support=self.support,
                          leaf_fontsize=self.leaf_fontsize,
                          branch_fontsize=self.branch_fontsize,
                          branch_width=self.branch_width,
                          branch_color=self.branch_color,
                          highlight_support=self.highlight_support,
                          branchlabels=self.branchlabels,
                          decorators=self.decorators,
                          leaflabels=self.leaflabels,
                          xlim=self.get_xlim(),
                          ylim=self.get_ylim())
        return p

    def highlight(self, nodes=None, width=5, color="red"):
        if self.highlightpatch:
            try:
                self.highlightpatch.remove()
            except:
                pass
        if not nodes:
            return

        if len(nodes)>1:
            mrca = self.root.mrca(nodes)
            if not mrca:
                return
        else:
            mrca = list(nodes)[0]

        M = Path.MOVETO; L = Path.LINETO
        verts = []
        codes = []
        seen = set()
        for node, coords in [ x for x in self.n2c.items() if x[0] in nodes ]:
            x = coords.x; y = coords.y
            p = node.parent
            while p:
                pcoords = self.n2c[p]
                px = pcoords.x; py = pcoords.y
                if node not in seen:
                    verts.append((x, y)); codes.append(M)
                    verts.append((px, y)); codes.append(L)
                    verts.append((px, py)); codes.append(L)
                    seen.add(node)
                if p == mrca or node == mrca:
                    break
                node = p
                coords = self.n2c[node]
                x = coords.x; y = coords.y
                p = node.parent
        px, py = verts[-1]
        verts.append((px, py)); codes.append(M)

        self.highlightpath = Path(verts, codes)
        self.highlightpatch = PathPatch(
            self.highlightpath, fill=False, linewidth=width, edgecolor=color
            )
        return self.add_patch(self.highlightpatch)

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
        for n in nodes:
            if n.isleaf:
                txt = self.node2label[n]
                if txt.get_visible():
                    disp_bb.append(txt.get_window_extent())

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

    def draw_leaf_labels(self, *args):
        leaves = list(filter(lambda x:x[0].isleaf,
                             self.get_visible_nodes(labeled_only=True)))
        psep = self.leaf_pixelsep()
        fontsize = min(self.leaf_fontsize, max(psep, 8))
        n2l = self.node2label
        transform = self.transData.transform
        sub = operator.sub

        for n in leaves:
            n2l[n[0]].set_visible(False)

        # draw leaves
        leaves_drawn = []
        for n, x, y in leaves:
            txt = self.node2label[n]
            if not leaves_drawn:
                txt.set_visible(True)
                leaves_drawn.append(txt)
                self.figure.canvas.draw_idle()
                continue

            txt2 = leaves_drawn[-1]
            y0 = y; y1 = txt2.xy[1]
            sep = sub(*transform(([0,y0],[0,y1]))[:,1])
            if sep > fontsize:
                txt.set_visible(True)
                txt.set_size(fontsize)
                leaves_drawn.append(txt)
        self.figure.canvas.draw_idle()

        if leaves_drawn:
            leaves_drawn[0].set_size(fontsize)

        return fontsize

    def draw_labels(self, *args):
        fs = max(10, self.draw_leaf_labels())
        nodes = self.get_visible_nodes(labeled_only=True)
        ## print [ x[0].id for x in nodes ]
        branches = list(filter(lambda x:(not x[0].isleaf), nodes))
        n2l = self.node2label
        for n, x, y in branches:
            t = n2l[n]
            t.set_visible(True)
            t.set_size(fs)

    def unclutter(self, *args):
        nodes = self.get_visible_nodes(labeled_only=True)
        branches = list(filter(lambda x:(not x[0].isleaf), nodes))
        psep = self.leaf_pixelsep()
        n2l = self.node2label
        fontsize = min(self.leaf_fontsize*1.2, max(psep, self.leaf_fontsize))

        drawn = []
        for n, x, y in branches:
            txt = n2l[n]
            try:
                bb = txt.get_window_extent().expanded(2, 2)
                vis = True
                for n2 in reversed(drawn):
                    txt2 = n2l[n2]
                    if bb.overlaps(txt2.get_window_extent()):
                        txt.set_visible(False)
                        vis = False
                        self.figure.canvas.draw_idle()
                        break
                if vis:
                    txt.set_visible(True)
                    txt.set_size(fontsize)
                    self.figure.canvas.draw_idle()
                    drawn.append(n)
            except RuntimeError:
                pass
                ## txt.set_visible(True)
                ## txt.set_size(fontsize)
                ## drawn.append(n)
                ## self.figure.canvas.draw_idle()

    def leaf_pixelsep(self):
        y0, y1 = self.get_ylim()
        y0 = max(0, y0)
        y1 = min(self.nleaves, y1)
        display_points = self.transData.transform(((0, y0), (0, y1)))
        # height in pixels (visible y data extent)
        height = operator.sub(*reversed(display_points[:,1]))
        pixelsep = height/((y1-y0)/self.leaf_hsep)
        return pixelsep

    def ypp(self):
        y0, y1 = self.get_ylim()
        p0, p1 = self.transData.transform(((0, y0), (0, y1)))[:,1]
        return (y1-y0)/float(p1-p0)

    def draw_labels_old(self, *args):
        if self.nleaves:
            y0, y1 = self.get_ylim()
            y0 = max(0, y0); y1 = min(1, y1)

            display_points = self.transData.transform(((0, y0), (0, y1)))
            # height in pixels (visible y data extent)
            height = operator.sub(*reversed(display_points[:,1]))
            pixelsep = height/((y1-y0)/self.leaf_hsep)
            fontsize = min(max(pixelsep-2, 8), 12)

            if pixelsep >= 8:
                for node, txt in self.node2label.items():
                    if node.isleaf:
                        if self.leaflabels:
                            c = self.n2c[node]
                            x = c.x; y = c.y
                            if (y0 < y < y1):
                                txt.set_size(fontsize)
                                txt.set_visible(True)
                    else:
                        if self.branchlabels:
                            c = self.n2c[node]
                            x = c.x; y = c.y
                            if (y0 < y < y1):
                                txt.set_size(fontsize)
                                txt.set_visible(True)
            elif pixelsep >= 4:
                for node, txt in self.node2label.items():
                    if node.isleaf:
                        txt.set_visible(False)
                    else:
                        if self.branchlabels:
                            c = self.n2c[node]
                            x = c.x; y = c.y
                            if (y0 < y < y1):
                                txt.set_size(fontsize)
                                txt.set_visible(True)
            else:
                for node, txt in self.node2label.items():
                    txt.set_visible(False)
            self.figure.canvas.draw_idle()

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
        if self.interactive:
            self.callbacks.connect("ylim_changed", self.draw_labels)

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
        for c in self.n2c.values():
            c.x += self.xoff; c.y += self.yoff
        sv = sorted([
            [c.y, c.x, n] for n, c in self.n2c.items()
            ])
        self.coords = sv#numpy.array(sv)
        ## n2c = self.n2c
        ## self.node2linesegs = {}
        ## for node, coords in n2c.items():
        ##     x = coords.x; y = coords.y
        ##     v = [(x,y)]
        ##     if node.parent:
        ##         pcoords = n2c[node.parent]
        ##         px = pcoords.x; py = pcoords.y
        ##         v.append((px,y))
        ##         v.append((px,py))
        ##         self.node2linesegs[node] = v

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
        Draw branches and labels
        """
        if root and not self.root:
            self.set_root(root)

        if self.interactive: pyplot.ioff()

        if "branchlabels" in kwargs:
            self.branchlabels = kwargs["branchlabels"]
        if "leaflabels" in kwargs:
            self.leaflabels = kwargs["leaflabels"]
        self.yaxis.set_visible(False)
        self.create_branch_artists()
        self.create_label_artists()
        if self.highlight_support:
            self.do_highlight_support()
        self.mark_named()
        ## self.home()

        for k, v in self.decorators:
            func, args, kwargs = v
            func(self, *args, **kwargs)

        self.set_name(self.name)
        self.adjust_xspine()

        if self.interactive: pyplot.ion()

        labels = [ x.label for x in self.root.leaves() ]
        def fmt(x, pos=None):
            if x<0: return ""
            try: return labels[int(round(x))]
            except: pass
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
        ## for node in self._decimate_nodes():
        ##     if node.parent:
        ##         for p in self._path_to_parent(node):
        ##             patches.append(p)
        ## self.branch_patches = PatchCollection(patches, match_original=True)
        ## self.add_collection(self.branch_patches)

        ## print "enter: create_branch_artists"
        ## self.node2branch = {}
        ## for node, segs in self.node2linesegs.items():
        ##     line = Line2D(
        ##         [x[0] for x in segs], [x[1] for x in segs],
        ##         lw=self.branch_width, color=self.branch_color
        ##         )
        ##     line.set_visible(False)
        ##     Axes.add_artist(self, line)
        ##     self.node2branch[node] = line

        ## d = self.node2linesegs
        ## segs = [ d[n] for n in self.root if (n in d) ]

        ## dims = self.clade_dimensions(); ypp = self.ypp()
        ## def recurse(n, dims, clades, terminals):
        ##     stop = False
        ##     h = None
        ##     v = dims.get(n)
        ##     if v: h = (v[3]-v[2])/ypp
        ##     if (h and (h < 20)) or (not h):
        ##         stop = True
        ##         terminals.append(n)
        ##     if not stop:
        ##         clades.append(n)
        ##         for c in n.children:
        ##             recurse(c, dims, clades, terminals)
        ## clades = []; terminals = []
        ## recurse(self.root, dims, clades, terminals)
        ## segs = [ d[n] for n in self.root if (n in d) and (n in clades) ]
        ## for t in terminals:
        ##     if t.isleaf:
        ##         segs.append(d[t])
        ##     else:
        ##         x0, x1, y0, y1 = dims[t]
        ##         x, y = self.n2c[t].point()
        ##         px, py = self.n2c[t.parent].point()
        ##         segs.append(((px,py), (px,y), (x,y), (x1, y0), (x1,y1), (x,y)))

        ## lc = LineCollection(segs, linewidths=self.branch_width,
        ##                     colors = self.branch_color)
        ## self.branches_linecollection = Axes.add_collection(self, lc)
        ## print "leave: create_branch_artists"

    def create_label_artists(self):
        ## print "enter: create_label_artists"
        self.node2label = {}
        n2c = self.n2c
        for node, coords in n2c.items():
            x = coords.x; y = coords.y
            if node.isleaf and node.label and self.leaflabels:
                txt = self.annotate(
                    node.label,
                    xy=(x, y),
                    xytext=(self.leaf_offset, 0),
                    textcoords="offset points",
                    verticalalignment=self.leaf_valign,
                    horizontalalignment=self.leaf_halign,
                    fontsize=self.leaf_fontsize,
                    clip_on=True,
                    picker=True
                )
                txt.node = node
                txt.set_visible(False)
                self.node2label[node] = txt

            if (not node.isleaf) and node.label and self.branchlabels:
                txt = self.annotate(
                    node.label,
                    xy=(x, y),
                    xytext=(self.branch_offset,0),
                    textcoords="offset points",
                    verticalalignment=self.branch_valign,
                    horizontalalignment=self.branch_halign,
                    fontsize=self.branch_fontsize,
                    bbox=dict(fc="lightyellow", ec="none", alpha=0.8),
                    clip_on=True,
                    picker=True
                )
                ## txt.set_visible(False)
                txt.node = node
                self.node2label[node] = txt
        ## print "leave: create_label_artists"

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
        if self.node2label:
            try:
                v = [ x.get_window_extent() for x in self.node2label.values()
                      if x.get_visible() ]
                if v:
                    xmax = trans((max([ x.xmax for x in v ]),0))[0]
                    xmin = trans((min([ x.xmin for x in v ]),0))[0]
            except RuntimeError:
                pass

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

    def scroll(self, x, y):
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        xd = (x1-x0)*x
        yd = (y1-y0)*y
        self.set_xlim(x0+xd, x1+xd)
        self.set_ylim(y0+yd, y1+yd)
        self.adjust_xspine()

    def plot_labelcolor(self, nodemap, state2color=None):
        if state2color is None:
            c = colors.tango()
            states = sorted(set(nodemap.values()))
            state2color = dict(zip(states, c))

        for node, txt in self.node2label.items():
            s = nodemap.get(node)
            if s is not None:
                c = state2color[s]
                if c:
                    txt.set_color(c)
        self.figure.canvas.draw_idle()

    def node_image(self, node, imgfile, maxdim=100, border=0):
        xoff = self.leaf_offset
        n = self.root[node]; c = self.n2c[n]; p = (c.x, c.y)
        img = Image.open(imgfile)
        if max(img.size) > maxdim:
            img.thumbnail((maxdim, maxdim))
        imgbox = OffsetImage(img)
        xycoords = self.node2label.get(node) or "data"
        if xycoords != "data": p = (1, 0.5)
        abox = AnnotationBbox(imgbox, p,
                              xybox=(xoff, 0.0),
                              xycoords=xycoords,
                              box_alignment=(0.0,0.5),
                              pad=0.0,
                              boxcoords=("offset points"))
        self.add_artist(abox)

    def plot_discrete(self, data, cmap=None, name=None,
                      xoff=10, yoff=0, size=15, legend=1):
        root = self.root
        if cmap is None:
            import ivy
            c = colors.tango()
            states = sorted(set(data.values()))
            cmap = dict(zip(states, c))
        n2c = self.n2c
        points = []; c = []
        d = dict([ (n, data.get(n)) for n in root if data.get(n) is not None ])
        for n, v in d.items():
            coord = n2c[n]
            points.append((coord.x, coord.y)); c.append(cmap[v])

        boxes = symbols.squares(self, points, c, size, xoff=xoff, yoff=yoff)

        if legend:
            handles = []; labels = []
            for v, c in sorted(cmap.items()):
                handles.append(Rectangle((0,0),0.5,1,fc=c))
                labels.append(str(v))
            self.legend(handles, labels, loc=legend)

        self.figure.canvas.draw_idle()
        return boxes

    def plot_continuous(self, data, mid=None, name=None, cmap=None,
                        size=15, colorbar=True):
        area = (size*0.5)*(size*0.5)*numpy.pi
        values = data.values()
        vmin = min(values); vmax = max(values)
        if mid is None:
            mid = (vmin+vmax)*0.5
            delta = vmax-vmin*0.5
        else:
            delta = max(abs(vmax-mid), abs(vmin-mid))
        norm = mpl_colors.Normalize(mid-delta, mid+delta)
        ## if cmap is None: cmap = mpl_colormap.binary
        if cmap is None: cmap = mpl_colormap.hot
        n2c = self.n2c
        X = numpy.array(
            [ (n2c[n].x, n2c[n].y, v) for n, v in data.items() if n in n2c ]
            )
        circles = self.scatter(
            X[:,0], X[:,1], s=area, c=X[:,2], cmap=cmap, norm=norm,
            zorder=1000
            )
        if colorbar:
            cbar = self.figure.colorbar(circles, ax=self, shrink=0.7)
            if name:
                cbar.ax.set_xlabel(name)

        self.figure.canvas.draw_idle()

class RadialTree(Tree):
    def layout(self):
        from ..layout_polar import calc_node_positions
        start = self.start if hasattr(self, 'start') else 0
        end = self.end if hasattr(self, 'end') else None
        self.n2c = calc_node_positions(self.root, scaled=self.scaled,
                                       start=start, end=end)
        sv = sorted([
            [c.y, c.x, n] for n, c in self.n2c.items()
            ])
        self.coords = sv

    ## def _path_to_parent(self, node, width=None, color=None):
    ##     c = self.n2c[node]; theta1 = c.angle; r = c.depth
    ##     M = Path.MOVETO; L = Path.LINETO
    ##     pc = self.n2c[node.parent]; theta2 = pc.angle
    ##     px1 = math.cos(math.radians(c.angle))*pc.depth
    ##     py1 = math.sin(math.radians(c.angle))*pc.depth
    ##     verts = [(c.x,c.y),(px1,py1)]; codes = [M,L]
    ##     #verts.append((pc.x,pc.y)); codes.append(L)
    ##     path = PathPatch(Path(verts, codes), fill=False,
    ##                      linewidth=width or self.branch_width,
    ##                      edgecolor=color or self.branch_color)
    ##     diam = pc.depth*2
    ##     t1, t2 = tuple(sorted((theta1,theta2)))
    ##     arc = Arc((0,0), diam, diam, theta1=t1, theta2=t2,
    ##               edgecolor=color or self.branch_color,
    ##               linewidth=width or self.branch_width)
    ##     return [path, arc]

    def _path_to_parent(self, node):
        c = self.n2c[node]; theta1 = c.angle; r = c.depth
        M = Path.MOVETO; L = Path.LINETO
        pc = self.n2c[node.parent]; theta2 = pc.angle
        px1 = math.cos(math.radians(c.angle))*pc.depth
        py1 = math.sin(math.radians(c.angle))*pc.depth
        verts = [(c.x,c.y),(px1,py1)]; codes = [M,L]
        t1, t2 = tuple(sorted((theta1,theta2)))
        diam = pc.depth*2
        arc = Arc((0,0), diam, diam, theta1=t1, theta2=t2)
        arcpath = arc.get_path()
        av = arcpath.vertices * pc.depth
        ac = arcpath.codes
        verts.extend(av.tolist())
        codes.extend(ac.tolist())
        return verts, codes

    def highlight(self, nodes=None, width=5, color="red"):
        if self.highlightpatch:
            try:
                self.highlightpatch.remove()
            except:
                pass
        if not nodes:
            return

        if len(nodes)>1:
            mrca = self.root.mrca(nodes)
            if not mrca:
                return
        else:
            mrca = list(nodes)[0]

        M = Path.MOVETO; L = Path.LINETO
        verts = []
        codes = []
        seen = set()
        patches = []
        for node, coords in [ x for x in self.n2c.items() if x[0] in nodes ]:
            x = coords.x; y = coords.y
            p = node.parent
            while p:
                pcoords = self.n2c[p]
                px = pcoords.x; py = pcoords.y
                if node not in seen:
                    v, c = self._path_to_parent(node)
                    verts.extend(v)
                    codes.extend(c)
                    seen.add(node)
                if p == mrca or node == mrca:
                    break
                node = p
                coords = self.n2c[node]
                x = coords.x; y = coords.y
                p = node.parent
        ## px, py = verts[-1]
        ## verts.append((px, py)); codes.append(M)
        self.highlightpath = Path(verts, codes)
        self.highlightpatch = PathPatch(
            self.highlightpath, fill=False, linewidth=width, edgecolor=color
            )
        self.add_patch(self.highlightpatch)
        ## self.highlight_patches = PatchCollection(patches, match_original=True)
        ## self.add_collection(self.highlight_patches)


class OverviewTree(Tree):
    def __init__(self, *args, **kwargs):
        kwargs["leaflabels"] = False
        kwargs["branchlabels"] = False
        Tree.__init__(self, *args, **kwargs)
        self.xaxis.set_visible(False)
        self.spines["bottom"].set_visible(False)
        self.add_overview_rect()

    def set_target(self, target):
        self.target = target

    def add_overview_rect(self):
        rect = UpdatingRect([0, 0], 0, 0, facecolor='black', edgecolor='red')
        rect.set_alpha(0.2)
        rect.target = self.target
        rect.set_bounds(*self.target.viewLim.bounds)
        self.zoomrect = rect
        self.add_patch(rect)
        ## if pyplot.isinteractive():
        self.target.callbacks.connect('xlim_changed', rect)
        self.target.callbacks.connect('ylim_changed', rect)

    def redraw(self):
        Tree.redraw(self)
        self.add_overview_rect()
        self.figure.canvas.draw_idle()

def axes_enter(e):
    ax = e.inaxes
    ax._active = True

def axes_leave(e):
    ax = e.inaxes
    ax._active = False

def onselect(estart, estop):
    b = estart.button
    ## print b, estart.key

def onkeypress(e):
    ax = e.inaxes
    k = e.key
    if ax and k == 't':
        ax.home()
    if ax and k == "down":
        ax.scroll(0, -0.1)
        ax.figure.canvas.draw_idle()
    if ax and k == "up":
        ax.scroll(0, 0.1)
        ax.figure.canvas.draw_idle()
    if ax and k == "left":
        ax.scroll(-0.1, 0)
        ax.figure.canvas.draw_idle()
    if ax and k == "right":
        ax.scroll(0.1, 0)
        ax.figure.canvas.draw_idle()
    if ax and k and k in '=+':
        ax.zoom(0.1,0.1)
    if ax and k == '-':
        ax.zoom(-0.1,-0.1)

def ondrag(e):
    ax = e.inaxes
    button = e.button
    if ax and button == 2:
        if not ax.pan_start:
            ax.pan_start = (e.xdata, e.ydata)
            return
        x, y = ax.pan_start
        xdelta = x - e.xdata
        ydelta = y - e.ydata
        x0, x1 = ax.get_xlim()
        xspan = x1-x0
        y0, y1 = ax.get_ylim()
        yspan = y1 - y0
        midx = (x1+x0)*0.5
        midy = (y1+y0)*0.5
        ax.set_xlim(midx+xdelta-xspan*0.5, midx+xdelta+xspan*0.5)
        ax.set_ylim(midy+ydelta-yspan*0.5, midy+ydelta+yspan*0.5)
        ax.adjust_xspine()

def onbuttonrelease(e):
    ax = e.inaxes
    button = e.button
    if button == 2:
        ## print "pan end"
        ax.pan_start = None

def onpick(e):
    ax = e.mouseevent.inaxes
    if ax:
        ax.picked(e)

def onscroll(e):
    ax = e.inaxes
    if ax:
        b = e.button
        ## print b
        k = e.key
        if k == None and b =="up":
            ax.zoom(0.1,0.1)
        if k == None and b =="down":
            ax.zoom(-0.1,-0.1)
        if k == "shift" and b == "up":
            ax.zoom_cxy(0.1, 0, e.xdata, e.ydata)
        if k == "shift" and b == "down":
            ax.zoom_cxy(-0.1, 0, e.xdata, e.ydata)
        if k == "control" and b == "up":
            ax.zoom_cxy(0, 0.1, e.xdata, e.ydata)
        if k == "control" and b == "down":
            ax.zoom_cxy(0, -0.1, e.xdata, e.ydata)
        if k == "d" and b == "up":
            ax.scroll(0, 0.1)
        if (k == "d" and b == "down"):
            ax.scroll(0, -0.1)
        if k == "c" and b == "up":
            ax.scroll(-0.1, 0)
        if k == "c" and b == "down":
            ax.scroll(0.1, 0)
        try: ax.adjust_xspine()
        except: pass
        ax.figure.canvas.draw_idle()

def onclick(e):
    ax = e.inaxes
    if ax and e.button==1 and hasattr(ax, "zoomrect") and ax.zoomrect:
        # overview clicked; reposition zoomrect
        r = ax.zoomrect
        x = e.xdata
        y = e.ydata
        arr = ax.transData.inverted().transform(r.get_extents())
        xoff = (arr[1][0]-arr[0][0])*0.5
        yoff = (arr[1][1]-arr[0][1])*0.5
        r.target.set_xlim(x-xoff,x+xoff)
        r.target.set_ylim(y-yoff,y+yoff)
        r(r.target)
        ax.figure.canvas.draw_idle()

    if ax and e.button==2:
        ## print "pan start", (e.xdata, e.ydata)
        ax.pan_start = (e.xdata, e.ydata)


def test_decorate(treeplot):
    import evolve
    data = evolve.brownian(treeplot.root)
    values = data.values()
    vmin = min(values); vmax = max(values)
    norm = mpl_colors.Normalize(vmin, vmax)
    cmap = mpl_colormap.binary
    n2c = treeplot.n2c
    X = numpy.array(
        [ (n2c[n].x, n2c[n].y, v)
          for n, v in data.items() if n in n2c ]
        )
    circles = treeplot.scatter(
        X[:,0], X[:,1], s=200, c=X[:,2], cmap=cmap, norm=norm,
        zorder=100
        )

class Decorator(object):
    def __init__(self, treeplot):
        self.plot = treeplot

class VisToggle(object):
    def __init__(self, name, treeplot=None, value=False):
        self.name = name
        self.plot = treeplot
        self.value = value

    def __nonzero__(self):
        return self.value

    def __repr__(self):
        return "%s: %s" % (self.name, self.value)

    def redraw(self):
        if self.plot:
            self.plot.redraw()

    def toggle(self):
        self.value = not self.value
        self.redraw()

    def show(self):
        if self.value == False:
            self.value = True
            self.redraw()

    def hide(self):
        if self.value == True:
            self.value = False
            self.redraw()


TreePlot = subplot_class_factory(Tree)
RadialTreePlot = subplot_class_factory(RadialTree)
OverviewTreePlot = subplot_class_factory(OverviewTree)

if __name__ == "__main__":
    import evolve
    root, data = evolve.test_brownian()
    plot_continuous(root, data, name="Brownian", mid=0.0)
