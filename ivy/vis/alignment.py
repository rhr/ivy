"""
interactive viewers for trees, etc. using matplotlib
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import defaultdict
from ..storage import Storage
from .. import align, sequtil
import matplotlib, numpy, types
import matplotlib.pyplot as pyplot
from matplotlib.figure import SubplotParams, Figure
from matplotlib.axes import Axes, subplot_class_factory
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.widgets import RectangleSelector
from matplotlib.transforms import Bbox, offset_copy, IdentityTransform
from matplotlib import colors as mpl_colors
from matplotlib.ticker import MaxNLocator, FuncFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Bio.Align import MultipleSeqAlignment

matplotlib.rcParams['path.simplify'] = False

class UpdatingRect(Rectangle):
    def __call__(self, p):
        self.set_bounds(*p.viewLim.bounds)
        p.figure.canvas.draw_idle()

class AlignmentFigure:
    def __init__(self, aln, name=None, div=0.25, overview=True):
        if isinstance(aln, MultipleSeqAlignment):
            self.aln = aln
        else:
            self.aln = align.read(aln)
        self.name = name
        self.div_value = div
        pars = SubplotParams(
            left=0.2, right=1, bottom=0.05, top=1, wspace=0.01
            )
        fig = pyplot.figure(subplotpars=pars, facecolor="white")
        self.figure = fig
        self.initialize_subplots(overview)
        self.show()
        self.connect_events()
        
    def initialize_subplots(self, overview=False):
        ## p = AlignmentPlot(self.figure, 212, aln=self.aln)
        p = AlignmentPlot(self.figure, 111, aln=self.aln, app=self)
        self.detail = self.figure.add_subplot(p)
        self.detail.plot_aln()
        if overview:
            self.overview = inset_axes(
                self.detail, width="30%", height="20%", loc=1
                )
            self.overview.xaxis.set_major_locator(NullLocator())
            self.overview.yaxis.set_major_locator(NullLocator())
            self.overview.imshow(
                self.detail.array, interpolation='nearest', aspect='auto',
                origin='lower'
                )
            rect = UpdatingRect(
                [0,0], 0, 0, facecolor='black', edgecolor='cyan', alpha=0.5
                )
            self.overview.zoomrect = rect
            rect.target = self.detail
            self.detail.callbacks.connect('xlim_changed', rect)
            self.detail.callbacks.connect('ylim_changed', rect)
            self.overview.add_patch(rect)
            rect(self.overview)

        else:
            self.overview = None
        
    def show(self):
        self.figure.show()

    def connect_events(self):
        mpl_connect = self.figure.canvas.mpl_connect
        mpl_connect("button_press_event", self.onclick)
        mpl_connect("button_release_event", self.onbuttonrelease)
        mpl_connect("scroll_event", self.onscroll)
        mpl_connect("pick_event", self.onpick)
        mpl_connect("motion_notify_event", self.ondrag)
        mpl_connect("key_press_event", self.onkeypress)
        mpl_connect("axes_enter_event", self.axes_enter)
        mpl_connect("axes_leave_event", self.axes_leave)

    @staticmethod
    def axes_enter(e):
        ax = e.inaxes
        ax._active = True

    @staticmethod
    def axes_leave(e):
        ax = e.inaxes
        ax._active = False

    @staticmethod
    def onselect(estart, estop):
        b = estart.button
        ## print b, estart.key

    @staticmethod
    def onkeypress(e):
        ax = e.inaxes
        k = e.key
        if ax and k:
            if k == 't':
                ax.home()
            elif k == "down":
                ax.scroll(0, -0.1)
            elif k == "up":
                ax.scroll(0, 0.1)
            elif k == "left":
                ax.scroll(-0.1, 0)
            elif k == "right":
                ax.scroll(0.1, 0)
            elif k in '=+':
                ax.zoom(0.1,0.1)
            elif k == '-':
                ax.zoom(-0.1,-0.1)
            ax.figure.canvas.draw_idle()

    @staticmethod
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
            ax.figure.canvas.draw_idle()

    @staticmethod
    def onbuttonrelease(e):
        ax = e.inaxes
        button = e.button
        if button == 2:
            ## print "pan end"
            ax.pan_start = None
            ax.figure.canvas.draw_idle()

    @staticmethod
    def onpick(e):
        ax = e.mouseevent.inaxes
        if ax:
            ax.picked(e)
            ax.figure.canvas.draw_idle()

    @staticmethod
    def onscroll(e):
        ax = e.inaxes
        if ax:
            b = e.button
            ## print b
            k = e.key
            if k == None and b =="up":
                ax.zoom(0.1,0.1)
            elif k == None and b =="down":
                ax.zoom(-0.1,-0.1)
            elif k == "shift" and b == "up":
                ax.zoom_cxy(0.1, 0, e.xdata, e.ydata)
            elif k == "shift" and b == "down":
                ax.zoom_cxy(-0.1, 0, e.xdata, e.ydata)
            elif k == "control" and b == "up":
                ax.zoom_cxy(0, 0.1, e.xdata, e.ydata)
            elif k == "control" and b == "down":
                ax.zoom_cxy(0, -0.1, e.xdata, e.ydata)
            elif k == "d" and b == "up":
                ax.scroll(0, 0.1)
            elif (k == "d" and b == "down"):
                ax.scroll(0, -0.1)
            elif k == "c" and b == "up":
                ax.scroll(-0.1, 0)
            elif k == "c" and b == "down":
                ax.scroll(0.1, 0)
            ax.figure.canvas.draw_idle()

    @staticmethod
    def onclick(e):
        ax = e.inaxes
        if (ax and e.button==1 and hasattr(ax, "zoomrect") and ax.zoomrect):
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

        elif ax and e.button==2:
            ## print "pan start", (e.xdata, e.ydata)
            ax.pan_start = (e.xdata, e.ydata)
            ax.figure.canvas.draw_idle()

        elif ax and hasattr(ax, "aln") and ax.aln:
            x = int(e.xdata+0.5); y = int(e.ydata+0.5)
            aln = ax.aln
            if (0 <= x <= ax.nchar) and (0 <= y <= ax.ntax):
                seq = aln[y]; char = seq[x]
                if char not in '-?':
                    v = sequtil.gapidx(seq)
                    i = (v[1]==x).nonzero()[0][0]
                    print(("%s: row %s, site %s: '%s', seqpos %s"
                           % (seq.id, y, x, char, i)))
                else:
                    print("%s: row %s, site %s: '%s'" % (seq.id, y, x, char))

    def zoom(self, factor=0.1):
        "Zoom both axes by *factor* (relative display size)."
        self.detail.zoom(factor, factor)
        self.figure.canvas.draw_idle()

    def __get_selection(self):
        return self.detail.extract_selected()
    selected = property(__get_selection)
                
class Alignment(Axes):
    """
    matplotlib.axes.Axes subclass for rendering sequence alignments.
    """
    def __init__(self, fig, rect, *args, **kwargs):
        self.aln = kwargs.pop("aln")
        nrows = len(self.aln)
        ncols = self.aln.get_alignment_length()
        self.alnidx = numpy.arange(ncols)
        self.app = kwargs.pop("app", None)
        self.showy = kwargs.pop('showy', True)
        Axes.__init__(self, fig, rect, *args, **kwargs)
        rgb = mpl_colors.colorConverter.to_rgb
        gray = rgb('gray')
        d = defaultdict(lambda:gray)
        d["A"] = rgb("red")
        d["a"] = rgb("red")
        d["C"] = rgb("blue")
        d["c"] = rgb("blue")
        d["G"] = rgb("green")
        d["g"] = rgb("green")
        d["T"] = rgb("yellow")
        d["t"] = rgb("yellow")
        self.cmap = d
        self.selector = RectangleSelector(
            self, self.select_rectangle, useblit=True
            )
        def f(e):
            if e.button != 1: return True
            else: return RectangleSelector.ignore(self.selector, e)
        self.selector.ignore = f
        self.selected_rectangle = Rectangle(
            [0,0],0,0, facecolor='white', edgecolor='cyan', alpha=0.3
            )
        self.add_patch(self.selected_rectangle)
        self.highlight_find_collection = None

    def plot_aln(self):
        cmap = self.cmap
        self.ntax = len(self.aln); self.nchar = self.aln.get_alignment_length()
        a = numpy.array([ [ cmap[base] for base in x.seq ]
                          for x in self.aln ])
        self.array = a
        self.imshow(a, interpolation='nearest', aspect='auto', origin='lower')
        y = [ i+0.5 for i in range(self.ntax) ]
        labels = [ x.id for x in self.aln ]
        ## locator.bin_boundaries(1,ntax)
        ## locator.view_limits(1,ntax)
        if self.showy:
            locator = MaxNLocator(nbins=50, integer=True)
            self.yaxis.set_major_locator(locator)
            def fmt(x, pos=None):
                if x<0: return ""
                try: return labels[int(round(x))]
                except: pass
                return ""
            self.yaxis.set_major_formatter(FuncFormatter(fmt))
        else:
            self.yaxis.set_major_locator(NullLocator())
        
        return self

    def select_rectangle(self, e0, e1):
        x0, x1 = list(map(int, sorted((e0.xdata+0.5, e1.xdata+0.5))))
        y0, y1 = list(map(int, sorted((e0.ydata+0.5, e1.ydata+0.5))))
        self.selected_chars = (x0, x1)
        self.selected_taxa = (y0, y1)
        self.selected_rectangle.set_bounds(x0-0.5,y0-0.5,x1-x0+1,y1-y0+1)
        self.app.figure.canvas.draw_idle()

    def highlight_find(self, substr):
        if not substr:
            if self.highlight_find_collection:
                self.highlight_find_collection.remove()
                self.highlight_find_collection = None
            return
            
        N = len(substr)
        v = []
        for y, x in align.find(self.aln, substr):
            r = Rectangle(
                [x-0.5,y-0.5], N, 1,
                facecolor='cyan', edgecolor='cyan', alpha=0.7
                )
            v.append(r)
        if self.highlight_find_collection:
            self.highlight_find_collection.remove()
        c = PatchCollection(v, True)
        self.highlight_find_collection = self.add_collection(c)
        self.app.figure.canvas.draw_idle()

    def extract_selected(self):
        r0, r1 = self.selected_taxa
        c0, c1 = self.selected_chars
        return self.aln[r0:r1+1,c0:c1+1]

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

    def center_y(self, y):
        ymin, ymax = self.get_ylim()
        yoff = (ymax - ymin) * 0.5
        self.set_ylim(y-yoff, y+yoff)

    def center_x(self, x, offset=0.3):
        xmin, xmax = self.get_xlim()
        xspan = xmax - xmin
        xoff = xspan*0.5 + xspan*offset
        self.set_xlim(x-xoff, x+xoff)

    def scroll(self, x, y):
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        xd = (x1-x0)*x
        yd = (y1-y0)*y
        self.set_xlim(x0+xd, x1+xd)
        self.set_ylim(y0+yd, y1+yd)

    def home(self):
        self.set_xlim(0, self.nchar)
        self.set_ylim(self.ntax, 0)

AlignmentPlot = subplot_class_factory(Alignment)

