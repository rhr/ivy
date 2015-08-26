"""
Layer functions to add to a tree plot with the addlayer method
"""

def addlabel(treeplot, labeltype, leaf_offset = 4, leaf_valign = "center",
             leaf_halign = "left", leaf_fontsize = 10, branch_offset = -5,
             branch_valign = "center", branch_halign = "right", 
             branch_fontsize = "10"):
    """
    Add text labels to tree
    
    Args:
        treeplot: treeplot (fig.tree)
        labeltype (str): "leaf" or "branch"
        
    """
    assert labeltype in ["leaf", "branch"]
    treeplot.node2label = {}
    n2c = treeplot.n2c
    txt = []
    for node, coords in n2c.items():
        x = coords.x; y = coords.y
        if node.isleaf and node.label and labeltype == "leaf":
            ntxt = treeplot.annotate(
                node.label,
                xy=(x, y),
                xytext=(leaf_offset, 0),
                textcoords="offset points",
                verticalalignment=leaf_valign,
                horizontalalignment=leaf_halign,
                fontsize=leaf_fontsize,
                clip_on=True,
                picker=True
            )
            ntxt.node = node
            ntxt.set_visible(True)
            txt.append(ntxt)

        if (not node.isleaf) and node.label and labeltype == "branch":
            ntxt = treeplot.annotate(
                node.label,
                xy=(x, y),
                xytext=(branch_offset,0),
                textcoords="offset points",
                verticalalignment=branch_valign,
                horizontalalignment=branch_halign,
                fontsize=branch_fontsize,
                bbox=dict(fc="lightyellow", ec="none", alpha=0.8),
                clip_on=True,
                picker=True
            )
            ## txt.set_visible(False)
            ntxt.node = node
            ntxt.set_visible(True)
            txt.append(ntxt)
    treeplot.figure.canvas.draw_idle()
    return txt
                
