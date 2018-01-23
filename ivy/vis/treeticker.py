from matplotlib.ticker import FixedLocator
import numpy as np

class TreeTicker(FixedLocator):
    def __init__(self, treeplot, nbins=100):
        leaves = treeplot.root.leaves()
        cv = [ treeplot.n2c[n] for n in leaves ]
        locs = [ c.y for c in cv ]
        self.y2c = dict(zip(locs, cv))
        super().__init__(locs, nbins=nbins)

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        ticks = self.tick_values(vmin, vmax)
        y2c = self.y2c
        for c in y2c.values():
            try:
                hl = getattr(c, 'hline')
                hl.set_visible(False)
            except AttributeError:
                continue
        if self.axis.axes._axis_leaflines:
            for y in ticks:
                c = y2c[y]
                try:
                    hl = getattr(c, 'hline')
                except AttributeError:
                    ax = self.axis.axes
                    hl = ax.annotate(
                        '', xy=(c.x, c.y), xycoords='data',
                        xytext=(1, c.y), textcoords=('axes fraction', 'data'),
                        arrowprops=dict(
                            arrowstyle='-',
                            connectionstyle='arc3',
                            lw=0.5,
                            color='gray',
                            fc='gray',
                            ls='dashed'))
                    c.hline = hl
                hl.set_visible(True)
        return ticks
        
    def tick_values(self, vmin, vmax):
        if self.nbins is None:
            return self.locs
        locs = self.locs[self.locs>=vmin]
        locs = locs[locs<=vmax]
        step = max(int(0.99 + len(locs) / float(self.nbins)), 1)
        ticks = locs[::step]
        for i in range(1, step):
            ticks1 = locs[i::step]
            if np.abs(ticks1).min() < np.abs(ticks).min():
                ticks = ticks1
        return self.raise_if_exceeds(ticks)
        
