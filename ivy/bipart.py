import sys
from pprint import pprint
from glob import glob
from storage import Storage
from collections import defaultdict

## class BipartSet(object):
##     "A set of bipartitions"
##     def __init__(self, elements):
##         self.elements = frozenset(elements)
##         self.ref = sorted(elements)[0]
##         self.node2bipart = Storage()

##     def add(self, subset, node):
##         # filter out elements of subset not in 'elements'
##         subset = (frozenset(subset) & self.elements)
##         if self.ref not in self.subset:
##             self.subset = self.elements - self.subset

class Bipart(object):
    """
    A class representing a bipartition.
    """
    def __init__(self, elements, subset, node=None, support=None):
        """
        'elements' and 'subset' are set objects
        """
        self.subset = subset
        self.compute(elements)
        self.node = node
        self.support = support

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        assert self.elements == other.elements
        return ((self.subset == other.subset) or
                (self.subset == (self.elements - other.subset)))

    def __repr__(self):
        v = sorted(self.subset)
        return "(%s)" % " ".join(map(str, v))

    def compute(self, elements):
        self.elements = frozenset(elements)
        self.ref = sorted(elements)[0]
        # filter out elements of subset not in 'elements'
        self.subset = (frozenset(self.subset) & self.elements)
        self._hash = hash(self.subset)
        if self.ref not in self.subset:
            self.subset = self.elements - self.subset
        self.complement = self.elements - self.subset

    def iscompatible(self, other):
        ## assert self.elements == other.elements
        if (self.subset.issubset(other.subset) or
            other.subset.issubset(self.subset)):
            return True
        if (((self.subset | other.subset) == self.elements) or
            (not (self.subset & other.subset))):
            return True
        return False

def conflict(bp1, bp2, support=None):
    if ((support and (bp1.support >= support) and (bp2.support >= support))
        or (not support)):
        if not bp1.iscompatible(bp2):
            return True
    return False

class TreeSet:
    def __init__(self, root, elements=None):
        self.root = root
        self.node2labels = root.leafsets(labels=True)
        self.elements = elements or self.node2labels.pop(root)
        self.biparts = [ Bipart(self.elements, v, node=k,
                                support=int(k.label or 0))
                         for k, v in self.node2labels.items() ]

def compare_trees(r1, r2, support=None):
    e = (set([ x.label for x in r1.leaves() ]) &
         set([ x.label for x in r2.leaves() ]))
    bp1 = [ Bipart(e, v, node=k, support=int(k.label or 0))
            for k, v in r1.leafsets(labels=True).items() ]
    bp2 = [ Bipart(e, v, node=k, support=int(k.label or 0))
            for k, v in r2.leafsets(labels=True).items() ]
    return compare(bp1, bp2, support)

def compare(set1, set2, support=None):
    hits1 = []; hits2 = []
    conflicts1 = defaultdict(set); conflicts2 = defaultdict(set)
    for bp1 in set1:
        for bp2 in set2:
            if bp1 == bp2:
                hits1.append(bp1.node); hits2.append(bp2.node)
            if conflict(bp1, bp2, support):
                conflicts1[bp1.node].add(bp2.node)
                conflicts2[bp2.node].add(bp1.node)
    return hits1, hits2, conflicts1, conflicts2
    
## a = Bipart("abcdef", "abc")
## b = Bipart("abcdef", "def")
## c = Bipart("abcdef", "ab")
## d = Bipart("abcdef", "cd")
## print a == b
## print a.iscompatible(b)
## print a.iscompatible(c)
## print a.iscompatible(d)
## print c.iscompatible(d)
## sys.exit()    
