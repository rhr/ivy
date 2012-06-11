import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tree
from axes_utils import adjust_limits

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
                 xlim=None, ylim=None):
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

        nleaves = len(root.leaves())
        self.dpi = 72.0
        self.leaf_fontsize = 10.0
        self.height = (nleaves*self.leaf_fontsize*self.leafpad)/self.dpi
        self.width = self.height*self.relwidth
        self.height += 2
        self.width += 2
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
        self.axes.home()
        adjust_limits(self.axes)
        self.axes.set_position([0.05,0.05,0.95,0.95])
        
    def savefig(self, fname):
        self.figure.savefig(fname)
