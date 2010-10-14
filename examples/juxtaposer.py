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

class JuxtaposerFigure(ivy.matplot.MultiTreeFigure):
    def on_nodes_selected(self, treeplot):
        for p in self.plots:
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

        other_plots = [ x for x in self.plots if x != treeplot ]
        for p in other_plots:
            other_leaves = p.root.leaves()
            hits = []
            for lf in leaves:
                hits.extend([ x for x in other_leaves if compare(lf, x) ])
            if hits:
                p.highlight(hits, 4, "green")
                p.figure.canvas.draw_idle()

fig = JuxtaposerFigure()
fig.add("/home/rree/Dropbox/Pedicularis-NSF/phylo/RAxML_bipartitions.its")
fig.add("/home/rree/Dropbox/Pedicularis-NSF/phylo/RAxML_bipartitions.matk")
fig.ladderize()
