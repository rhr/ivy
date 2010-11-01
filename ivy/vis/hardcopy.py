import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from .. import tree
import matplot
from axes_utils import adjust_limits

## class TreeFigure:
##     def __init__(self):
##         pass

matplotlib.rcParams["xtick.direction"] = "out"

class TreeFigure:
    def __init__(self, root, relwidth=0.5, leafpad=1.5):
        self.root = root
        self.relwidth = relwidth
        self.leafpad = leafpad

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
            matplot.TreePlot(self.figure, 1,1,1,
                             branchlabels=True,
                             leaflabels=True,
                             leaf_fontsize=self.leaf_fontsize,
                             interactive=False,
                             mark_named=False).plot_tree(self.root)
            )
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["left"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["bottom"].set_smart_bounds(True)
        self.axes.xaxis.set_ticks_position("bottom")

        for v in self.axes.node2label.values():
            v.set_visible(True)

        self.canvas.draw()
        adjust_limits(self.axes)
        self.axes.set_position([0.05,0.05,0.95,0.95])

