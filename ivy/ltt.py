"""
Compute lineages through time
"""
import pandas as pd
from math import isclose

def ltt(node):
    """
    Calculate lineages through time. The tree is assumed to be an
    ultrametric chronogram (extant leaves, with branch lengths
    proportional to time).

    Args: node (Node): The root node object. All descendants should
        have branch lengths.

    Returns:
        pandas data frame with columns for time (t) and diversity (n)

    """
    t = 0.0
    def it():
        yield t, len(node.children)
        v = [ (t+x.length, x) for x in node.children if x.children ]
        while v:
            w = []
            for ct, c in v:
                yield ct, len(c.children)-1
                w.extend([ (ct+x.length, x) for x in c.children if x.children ])
            v = w
    data = pd.DataFrame.from_records(sorted(it()), columns=('t','n'))
    data.n = data.n.cumsum()
    return data

def lttbd(root):
    n2t = {root:0.0}
    def traverse(n, t=0.0):
        u = t + n.length
        if n.parent:
            n2t[n] = u
        for c in n.children:
            traverse(c, u)

    traverse(root)
    T = max(n2t.values())
    v = []
    for n in root.preiter():
        nc = len(n.children) - 1
        t = n2t[n]
        if nc >= 1:  # internal node
            v.append((t, nc))
        else:  # tip
            if not isclose(t, T, rel_tol=1e-3):  # not extant
                v.append((t, -1))
    df = pd.DataFrame.from_records(sorted(v), columns=('t','n'))
    df.n = df.n.cumsum()
    return df

def test():
    import newick, ascii
    n = newick.parse("(((a:1,b:2):3,(c:3,d:1):1,(e:0.5,f:3):2.5):1,g:4);")
    v = ltt(n)
    print(ascii.render(n, scaled=1))
    for t, n in v:
        print(t, n)

if __name__ == "__main__":
    test()
