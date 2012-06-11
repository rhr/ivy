"""
Functions for dealing with trees as matrices.
"""
from collections import defaultdict

def vcv(root):
    "leaf variances and covariances"
    leafdists = root.leaf_distances()
    var = defaultdict(float)
    cov = defaultdict(float)
    for node in root.postiter(lambda x: x.children and x.length):
        c = node.length + sum([ x.length for x in node.rootpath(root)
                                if x.parent ])
        dists = leafdists[node]
        leaves = dists.keys()
        for lf1 in leaves:
            for lf2 in leaves:
                if lf1 is not lf2:
                    k = frozenset((lf1, lf2))
                    v = dists[lf1] + dists[lf2]
                    if k not in var:
                        var[k] = v
                        cov[k] = c
    return var, cov

## def vcv(root, labels=None):
##     """
##     Compute the variance-covariance matrix.
##     """
##     labels = labels or [ lf.label for lf in root.leaves() ]
##     N = len(labels)
##     var = [ [ 0 for x in labels ] for y in labels ]
##     cov = [ [ None for x in labels ] for y in labels ]
##     d = root.leaf_distances()
##     for i in range(N):
##         for j in range(i+1, N):
##             li = labels[i]
##             lj = labels[j]
##             for n in root.postiter():
##                 l2d = d[n]
##                 if (not n.isleaf) and (li in l2d) and (lj in l2d):
##                     dist = l2d[li] + l2d[lj]
##                     var[i][j] = dist
##                     cov[i][j] = sum([ x.length for x in n.rootpath()
##                                       if x.parent ])
##                     break
##     return var, cov

if __name__ == "__main__":
    import tree, ascii
    from pprint import pprint
    n = tree.read("(((a:1,b:2):3,(c:3,d:1):1,(e:0.5,f:3):2.5):1,g:4);")
    var, covar = vcv(n)
    for x in n:
        if not x.label: x.label = str(x.length or "")
        else: x.label = "%s %s" % (x.length, x.label)
    print ascii.render(n, scaled=1)
    
    for k, v in var.items():
        print [ x.label.split()[-1] for x in k ], v, covar[k]
