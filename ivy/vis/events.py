"""
Events for tree figures
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
try:
    import Image
except ImportError:
    from PIL import Image

_tango = colors.tango()

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
        ax.figure.canvas.draw_idle()
    if ax and k == '-':
        ax.zoom(-0.1,-0.1)
        ax.figure.canvas.draw_idle() # Had to add these lines, still don't know why

def ondrag(e):
    ax = e.inaxes
    try:
        if ax is None or ax.plottype=="overview":
            return
    except:
        return
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
        try:
            ax.draw_labels()
        except:
            pass
        ax.figure.canvas.draw_idle() # WARNING: I had to add this line
                                     # for button2 panning to work, but it isn't
                                     # present in the original. I'm not sure
                                     # why my re-written code needs this
                                     # line but the original doesn't.

def onbuttonrelease(e):
    ax = e.inaxes
    try:
        if ax is None or ax.plottype=="overview":
            return
    except:
        return
    button = e.button
    if ax and button == 2:
        ## print "pan end"
        ax.pan_start = None

def onpick(e):
    ax = e.mouseevent.inaxes
    try:
        if ax is None or ax.plottype=="overview":
            return
    except:
        return
    if ax:
        ax.picked(e)

def onscroll(e):
    ax = e.inaxes
    try:
        if ax is None or ax.plottype=="overview":
            return
    except:
        return
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
    # if ax and e.button==1 and hasattr(ax, "zoomrect") and ax.zoomrect:
    #     # overview clicked; reposition zoomrect
    #     r = ax.zoomrect
    #     x = e.xdata
    #     y = e.ydata
    #     arr = ax.transData.inverted().transform(r.get_extents())
    #     xoff = (arr[1][0]-arr[0][0])*0.5
    #     yoff = (arr[1][1]-arr[0][1])*0.5
    #     r.target.set_xlim(x-xoff,x+xoff)
    #     r.target.set_ylim(y-yoff,y+yoff)
    #     r(r.target)
    #     ax.figure.canvas.draw_idle()

    if ax and e.button==2:
        try:
            if ax is None or ax.plottype=="overview":
                return
        except:
            return
        ## print "pan start", (e.xdata, e.ydata)
        ax.pan_start = (e.xdata, e.ydata)

class UpdatingRect(Rectangle):
    def __call__(self, p):
        self.set_bounds(*p.viewLim.bounds)
        p.figure.canvas.draw_idle()



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
