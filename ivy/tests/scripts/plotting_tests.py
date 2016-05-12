import ivy
from ivy.interactive import *
from ivy.vis import layers


tree = ivy.tree.read("support/hrm_600tips.newick")

fig = treefig(tree)

fig.toggle_leaflabels()
fig.toggle_leaflabels()

fig.find("t477")
fig.zoom_clade(tree[50])
fig.home()

fig.toggle_overview()
fig.toggle_overview()

fig.highlight(tree[300])
fig.cbar(tree[580])
fig.redraw()

fig.tip_chars([0,1]*300)
# Layers
fig.add_layer(layers.add_squares, "t83")
fig.add_layer(layers.add_circles, "t200", store="circle")

fig.redraw()
fig.layers
fig.remove_layer("circle")

fig.add_layer(layers.add_pie, "t20", [.5,.5])

h = fig.hardcopy()

# Radial
fig2 = treefig(tree, radial=True)
fig2.highlight(tree[221])
fig2.cbar(tree[221])
