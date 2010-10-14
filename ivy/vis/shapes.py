"""
Convenience functions for drawing shapes on TreePlots.
"""
from numpy import pi
from matplotlib.collections import RegularPolyCollection
from matplotlib.transforms import offset_copy

def squares(plot, points, colors, size=15, xoff=0, yoff=0):
    trans = offset_copy(
        plot.transData, fig=plot.figure, x=xoff, y=yoff, units='points'
        )

    col = RegularPolyCollection(
        numsides=4, rotation=pi*0.25, sizes=(size*size,),
        offsets=points, facecolors=colors, transOffset=trans,
        edgecolors='none'
        )

    return plot.add_collection(col)

def circles(plot, points, colors, size=15, xoff=0, yoff=0):
    trans = offset_copy(
        plot.transData, fig=plot.figure, x=xoff, y=yoff, units='points'
        )

    col = CircleCollection(
        sizes=(pi*size*size*0.25,),
        offsets=points, facecolors=colors, transOffset=trans,
        edgecolors='none'
        )

    return plot.add_collection(col)
