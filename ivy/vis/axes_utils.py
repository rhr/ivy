from __future__ import absolute_import, division, print_function, unicode_literals

from matplotlib import transforms, pyplot

def iter_bboxes(axes):
    for x in ("artists", "collections", "patches", "texts"):
        for y in getattr(axes, x):
            if y.get_visible():
                try:
                    yield y.get_window_extent()
                except:
                    pass

def artist_data_extents(axes):
    v = list(iter_bboxes(axes))
    if v:
        b = transforms.Bbox.union(v)
        w = min(b.width, b.height)*0.1
        b = b.expanded((b.width+w)/b.width,(b.height+w)/b.height)
        b = b.inverse_transformed(axes.transData)
        return b
    else:
        return transforms.Bbox.unit()

def adjust_limits(axes):
    v = list(axes.n2c.values())
    vx = [ c.x for c in v ]; vy = [ c.y for c in v ]
    bd = artist_data_extents(axes)
    x0, x1 = axes.get_xlim()
    y0, y1 = axes.get_ylim()

    ## print x0, x1, y0, y1
    ## print bd.x0, bd.x1, bd.y0, bd.y1
    rv = []
    if bd.x0 < x0:
        x0 = bd.x0; rv.append(bd.x0-x0)
    if bd.x1 > x1:
        x1 = bd.x1; rv.append(bd.x1-x1)
    if bd.y0 < y0:
        y0 = bd.y0; rv.append(y0-bd.y0)
    if bd.y1 > y1:
        y1 = bd.y1; rv.append(bd.y1-y1)

    ## print rv
    if rv:
        axes.set_xlim(bd.x0, bd.x1, emit=False)
        axes.set_ylim(bd.y0, bd.y1, emit=False)
        axes.adjust_xspine()
        return True
    return False
