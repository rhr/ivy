import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tree
from axes_utils import adjust_limits
from pyPdf import PdfFileWriter, PdfFileReader
from StringIO import StringIO

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
        
    def savefig(self, fname, format="pdf"):
        self.figure.savefig(fname, format = format)

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
        
        
    def render_multipage(self, opts, outbuf=None):
        pagesize = opts.pagesize or [8.5, 11.0]
        border = opts.border or 0.393701 # border = 1cm (mpl works in inches)
        landscape = opts.landscape or False
        pgwidth, pgheight = pagesize if not landscape \
                            else (pagesize[1], pagesize[0])
        if opts.dims:
            self.width = opts.dims[0] # In inches
            self.height = opts.dims[1]
            self.figure.set_size_inches(opts.dims)
        #print "drawing width, height:", drawing.width/inch, drawing.height/inch
        if self.width > pgwidth - 2*border:
            scalefact = (pgwidth - 2*border)/float(self.width)
            #self.figure.set_size_inches(scalefact*self.width, scalefact*self.height)
            #self.width = scalefact*self.width; self.height = scalefact*self.height

        else:
            scalefact = 1.0
        #border *= scalefact
        dwidth = self.width * 72.0
        dheight = self.height * 72.0

        output = PdfFileWriter()
        if not outbuf:
            outfile = file(opts.outfile, "wb")
        else:
            outfile = outbuf

        buf = StringIO()
        self.savefig(buf)
        pgwidth = pgwidth*72
        pgheight = pgheight*72
        lower = dheight
        right = pgwidth
        left = 0
        pgnum = 0
        vpgnum = 0
        border = border*72 # Converting to pixels in 72 DPI
        while lower >= 0:
            if pgnum == 0:
                delta = 0.0
            else:
                delta = 2*border*vpgnum
            buf.seek(0)
            tmp = PdfFileReader(buf)
            page = tmp.getPage(0)
            box = page.mediaBox
            uly = float(box.getUpperLeft_y())
            ulx = float(box.getUpperLeft_x())
            upper = uly+border+delta-vpgnum*pgheight
            #lower = uly+border+delta-(pgnum+1)*pgheight
            lower = upper-pgheight
            box.setUpperRight(((right+border), upper))
            box.setUpperLeft(((left+border), upper))
            box.setLowerRight(((right+border), lower))
            box.setLowerLeft(((left+border), lower))
            output.addPage(page)
            pgnum += 1
            vpgnum += 1
            if (lower < 0) & (right < dwidth):
                lower = dheight
                right = right+pgwidth+2*border
                left = left+pgwidth+2*border
                vpgnum = 0

        output.write(outfile)
        return pgnum, scalefact
