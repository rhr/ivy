from . import tree, alignment, colors, symbols, hardcopy
TreeFigure = tree.TreeFigure
MultiTreeFigure = tree.MultiTreeFigure
AlignmentFigure = alignment.AlignmentFigure
JuxtaposerFigure = tree.JuxtaposerFigure

def make_legend(axes, entries, marker='rect', **kwargs):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    if marker == 'rect':
        m = Patch
    elif marker == 'line':
        m = Line2D
        kwargs['xdata'] = []
        kwargs['ydata'] = []
    else:
        raise ValueError('unknown marker {}'.format(marker))

    handles = [m(color=color, label=label, **kwargs)
               for color, label in entries]
    return axes.legend(handles=handles)
