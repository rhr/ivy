import os, matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tree
from axes_utils import adjust_limits
import tempfile

## class TreeFigure:
##     def __init__(self):
##         pass

matplotlib.rcParams["xtick.direction"] = "out"

class TreeFigure:
    def __init__(self, root, relwidth=0.5, leafpad=1.5, name=None,
                 support=70.0, scaled=True, mark_named=True,
                 leaf_fontsize=10, branch_fontsize=10,
                 branch_width=1, branch_color="black",
                 highlight_support=True,
                 branchlabels=True, leaflabels=True, decorators=[],
                 xoff=0, yoff=0,
                 xlim=None, ylim=None,
                 height=None, width=None):
        self.root = root
        self.relwidth = relwidth
        self.leafpad = leafpad
        self.name = name
        self.support = support
        self.scaled = scaled
        self.mark_named = mark_named
        self.leaf_fontsize = leaf_fontsize
        self.branch_fontsize = branch_fontsize
        self.branch_width = branch_width
        self.branch_color = branch_color
        self.highlight_support = highlight_support
        self.branchlabels = branchlabels
        self.leaflabels = leaflabels
        self.decorators = decorators
        self.xoff = xoff
        self.yoff = yoff

        nleaves = len(root.leaves())
        self.dpi = 72.0
        h = height or (nleaves*self.leaf_fontsize*self.leafpad)/self.dpi
        self.height = h
        self.width = width or self.height*self.relwidth
        ## p = min(self.width, self.height)*0.1
        ## self.height += p
        ## self.width += p
        self.figure = Figure(figsize=(self.width, self.height), dpi=self.dpi)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_axes(
            tree.TreePlot(self.figure, 1,1,1,
                          support=self.support,
                          scaled=self.scaled,
                          mark_named=self.mark_named,
                          leaf_fontsize=self.leaf_fontsize,
                          branch_fontsize=self.branch_fontsize,
                          branch_width=self.branch_width,
                          branch_color=self.branch_color,
                          highlight_support=self.highlight_support,
                          branchlabels=self.branchlabels,
                          leaflabels=self.leaflabels,
                          interactive=False,
                          decorators=self.decorators,
                          xoff=self.xoff, yoff=self.yoff,
                          name=self.name).plot_tree(self.root)
            )
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["left"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["bottom"].set_smart_bounds(True)
        self.axes.xaxis.set_ticks_position("bottom")

        for v in self.axes.node2label.values():
            v.set_visible(True)

        ## for k, v in self.decorators:
        ##     func, args, kwargs = v
        ##     func(self.axes, *args, **kwargs)

        self.canvas.draw()
        ## self.axes.home()
        ## adjust_limits(self.axes)
        self.axes.set_position([0.05,0.05,0.95,0.95])

    @property
    def detail(self):
        return self.axes
        
    def savefig(self, fname):
        root, ext = os.path.splitext(fname)
        buf = tempfile.TemporaryFile()
        for i in range(3):
            self.figure.savefig(buf, format=ext[1:].lower())
            self.home()
            buf.seek(0)
        buf.close()
        self.figure.savefig(fname)

    def set_relative_width(self, relwidth):
        w, h = self.figure.get_size_inches()
        self.figure.set_figwidth(h*relwidth)

    def autoheight(self):
        "adjust figure height to show all leaf labels"
        nleaves = len(self.root.leaves())
        h = (nleaves*self.leaf_fontsize*self.leafpad)/self.dpi
        self.height = h
        self.figure.set_size_inches(self.width, self.height)
        self.axes.set_ylim(-2, nleaves+2)

    def home(self):
        self.axes.home()
