from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import ivy.vis
from .axes_utils import adjust_limits
from PyPDF2 import PdfFileWriter, PdfFileReader
from io import StringIO
import functools

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
                 branchlabels=True, leaflabels=True, layers=[],
                 xoff=0, yoff=0,
                 xlim=None, ylim=None,
                 height=None, width=None, plottype="phylogram"):
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
        self.layers = layers
        self.xoff = xoff
        self.yoff = yoff

        nleaves = len(root.leaves())
        self.dpi = 72.0
        h = height or (nleaves*self.leaf_fontsize*self.leafpad)/self.dpi
        self.height = h
        self.width = width or self.height*self.relwidth
        self.plottype = plottype
        ## p = min(self.width, self.height)*0.1
        ## self.height += p
        ## self.width += p
        self.treefigure = ivy.vis.treevis.TreeFigure(self.root,
                          scaled=self.scaled,
                          mark_named=self.mark_named,
                          branchlabels=self.branchlabels,
                          leaflabels=self.leaflabels,
                          interactive=False,
                          xoff=self.xoff, yoff=self.yoff,
                          name=self.name, overview=False, radial=self.plottype=="radial")

        self.axes = self.treefigure.tree
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["left"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["bottom"].set_smart_bounds(True)
        self.axes.xaxis.set_ticks_position("bottom")

        self.treefigure.figure.set_size_inches(self.width, self.height)
        self.treefigure.redraw(keeptemp=True)
        for lay in self.layers: # Draw the layers.
            self.layers[lay].func(self.axes, *self.layers[lay].args[1:], **self.layers[lay].keywords)
            plt.draw()
        plt.close()
        ## self.axes.home()
        ## adjust_limits(self.axes)

    @property
    def detail(self):
        return self.axes

    def savefig(self, fname, format="pdf"):
        self.treefigure.figure.savefig(fname, format = format)

    def set_relative_width(self, relwidth):
        w, h = self.treefigure.figure.get_size_inches()
        self.treefigure.figure.set_figwidth(h*relwidth)

    def autoheight(self):
        "adjust figure height to show all leaf labels"
        nleaves = len(self.root.leaves())
        h = (nleaves*self.leaf_fontsize*self.leafpad)/self.dpi
        self.height = h
        self.treefigure.figure.set_size_inches(self.width, self.height)
        self.axes.set_ylim(-2, nleaves+2)

    def home(self):
        self.axes.home()


    def render_multipage(self, outfile, pagesize = [8.5, 11.0],
                         dims = None, border = 0.393701, landscape = False):
        """
        Create a multi-page PDF document where the figure is cut into
        multiple pages. Used for printing large figures.


        Args:
            outfile (string): The path to the output file.
            pagesize (list): Two floats. Page size of each individual page
              in inches. Defaults to 8.5 x 11.0.
            dims (list): Two floats. The dimensions of the final figure in
              inches. Defaults to the original size of the figure.
            border (float): The amount of overlap (in inches) between each page
              to make taping them together easier. Defaults to 0.393701 (1 cm)
            landscape (bool): Whether or not each page will be in landscape
              orientation. Defaults to false.
        """
        pgwidth, pgheight = pagesize if not landscape \
                            else (pagesize[1], pagesize[0])
        #print "drawing width, height:", drawing.width/inch, drawing.height/inch
        if dims:
            self.width = dims[0]
            self.height = dims[1]
        else:
            self.width, self.height = self.treefigure.figure.get_size_inches()
        if self.width > pgwidth - 2*border:
            scalefact = min(
                [(self.width-((self.width/pgwidth-1)*border*2))/self.width,
                 (self.height-((self.height/pgheight-1)*border*2))/self.height])
            #self.treefigure.figure.set_size_inches(scalefact*self.width, scalefact*self.height)
            #self.width = scalefact*self.width; self.height = scalefact*self.height
        else:
            scalefact = 1.0

        self.width *= scalefact # In inches
        self.height *= scalefact
        self.treefigure.figure.set_size_inches([self.width, self.height])

        #border *= scalefact
        dwidth = self.width * 72.0 # In pixels (72 DPI)
        dheight = self.height * 72.0

        output = PdfFileWriter()
        outfile = file(outfile, "wb")

        buf = StringIO()
        self.savefig(buf)
        pgwidth = pgwidth*72
        pgheight = pgheight*72

        upper = border
        lower = 0
        right = pgwidth
        left = 0

        pgnum = 0
        vpgnum = 0
        hpgnum = 0

        border = border*72 # Converting to pixels in 72 DPI

        while upper < dheight:
            #if vpgnum == 0:
            #    vdelta = 0.0
            #else:
            #    vdelta = 2*border*vpgnum
            buf.seek(0)
            tmp = PdfFileReader(buf)
            page = tmp.getPage(0)
            box = page.mediaBox
            upper += pgheight-border
            lower = upper-pgheight
            #uly = float(box.getUpperLeft_y())
            #ulx = float(box.getUpperLeft_x())
            #upper = uly+border+vdelta-vpgnum*pgheight
            #lower = uly+border+delta-(pgnum+1)*pgheight
            #lower = upper-pgheight
            box.setUpperRight((right, upper))
            box.setUpperLeft((left, upper))
            box.setLowerRight((right, lower))
            box.setLowerLeft((left, lower))
            output.addPage(page)
            pgnum += 1
            vpgnum += 1
            if (upper >= dheight) & (right < dwidth):
                lower = 0
                upper = border
                right += pgwidth-border
                left = right-pgwidth
                vpgnum = 0

        output.write(outfile)
        return pgnum, scalefact


if __name__ == "__main__":
    import ivy
    from ivy.interactive import *
    r = ivy.tree.read("examples/plants.newick")
    f = treefig(r)
    h = f.hardcopy()
    h.render_multipage(outfile = "test.pdf")
