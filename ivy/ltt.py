"""
Compute lineages through time
"""
import numpy

def ltt(node):
    """
    Calculate lineages through time.  The tree is assumed to be an
    ultrametric chronogram (extant leaves, with branch lengths
    proportional to time).

    Args:
        node (Node): A node object. All nodes should have branch lengths.

    Returns:
        tuple: (times, diversity) - 1D-arrays containing the results.
    """
    t = 0.0
    def it():
        yield t
        v = [ (t+x.length, x) for x in node.children if x.children ]
        while v:
            w = []
            for ct, c in v:
                yield ct
                w.extend([ (ct+x.length, x) for x in c.children if x.children ])
            v = w
    times = sorted(it())
    return times, list(range(2, len(times)+2))
            

def test():
    import newick, ascii
    n = newick.parse("(((a:1,b:2):3,(c:3,d:1):1,(e:0.5,f:3):2.5):1,g:4);")
    v = ltt(n)
    print(ascii.render(n, scaled=1))
    for t, n in v:
        print(t, n)

if __name__ == "__main__":
    test()
