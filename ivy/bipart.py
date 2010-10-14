import sys
from pprint import pprint
from glob import glob

class Bipart(object):
    """
    A class representing a bipartition.
    """
    def __init__(self, elements, subset, node=None, support=None):
        """
        'elements' and 'subset' are set objects
        """
        self.elements = frozenset(elements)
        self.ref = sorted(elements)[0]
        # filter out elements of subset not in 'elements'
        self.subset = (frozenset(subset) & self.elements)
        self._hash = hash(self.subset)
        if self.ref not in self.subset:
            self.subset = self.elements - self.subset
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

    def iscompatible(self, other):
        assert self.elements == other.elements
        if (self.subset.issubset(other.subset) or
            other.subset.issubset(self.subset)):
            return True
        return False
    
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
