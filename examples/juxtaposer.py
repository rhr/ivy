#!/usr/bin/env python
import ivy
from ivy.interactive import *

def compare(node1, node2):
    """
    Test whether nodes match and should be highlighted.
    """
    s1 = tuple(node1.label.split(".")[:-1])
    s2 = tuple(node2.label.split(".")[:-1])
    if s1 == s2:
        return True
    return False

class JuxtaposerFigure(ivy.vis.MultiTreeFigure):
    def on_nodes_selected(self, treeplot):
        for p in self.plot:
            p.highlight()
        nodes = treeplot.selected_nodes
        if not nodes:
            return
        if len(nodes) == 1:
            anc = list(nodes)[0]
        else:
            anc = treeplot.root.mrca(nodes)

        if not anc.isleaf:
            leaves = anc.leaves()
        else:
            leaves = [anc]

        other_plots = [ x for x in self.plot if x != treeplot ]
        for p in other_plots:
            other_leaves = p.root.leaves()
            hits = []
            for lf in leaves:
                hits.extend([ x for x in other_leaves if compare(lf, x) ])
            if hits:
                p.highlight(hits, 4, "green")
                p.figure.canvas.draw_idle()

fig = JuxtaposerFigure()
fig.add("examples/its.newick")
fig.add("examples/matk.newick")
fig.ladderize()
fig.show()
