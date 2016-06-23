## from cython.operator cimport dereference as deref, preincrement as inc
## from cython import address as addr
import numpy as np
cimport numpy as np
from numpy.math cimport NAN

cdef class Node

cdef class Node(object):
    """

    Node in a rooted, potentially multifurcating tree.

    """
    def __cinit__(self):
        self.length = NAN

    property isleaf:
        def __get__(self):
            return self.nchildren == 0

    property isroot:
        def __get__(self):
            return self.parent is None

    property children:
        def __get__(self):
            return list(self.iterchildren())

    property leaves:
        def __get__(self):
            pass

    def __iter__(self):
        return self.iternodes()

    cpdef add_child(self, Node n):
        cdef Node c
        if self.nchildren == 0:
            self.leftchild = n
        else:
            c = self.leftchild
            while c.rightsib is not None: c = c.rightsib
            c.rightsib = n
            n.leftsib = c
        n.parent = self
        self.nchildren += 1

    cpdef Node prune(self):
        cdef Node p = self.parent, lsib = self.leftsib, rsib = self.rightsib
        if p is not None:
            if lsib is None and rsib is None:
                # self is the only child of parent
                return p
            if lsib is None and rsib is not None:
                # self is the first child of parent
                p.leftchild = rsib
                rsib.leftsib = None
            elif lsib is not None and rsib is None:
                # self is the last child of parent
                lsib.rightsib = None
            elif lsib is not None and rsib is not None:
                # self has both left and right sibs
                lsib.rightsib = rsib
                rsib.leftsib = lsib
            else:
                pass
            p.nchildren -= 1
        self.parent = None
        self.leftsib = None
        self.rightsib = None
        return p

    def iterchildren(self):
        cdef Node n = self.leftchild
        while n is not None:
            yield n
            n = n.rightsib

    def iternodes(self):
        """
        iterate (preorder) over nodes descendant from self - including self
        """
        cdef Node child, n
        yield self
        for child in self.iterchildren():
            for n in child.iternodes():
                yield n

    def postiter(self):
        """
        iterate (postorder) over nodes descendant from self - including self
        """
        cdef Node n, child
        for child in self.iterchildren():
            for n in child.postiter():
                yield n
        yield self

    def bft(self):
        "breadth-first traversal of descendants"
        v = self.children
        while v:
            w = []
            for n in v:
                yield n
                w.extend(n.iterchildren())
            v = w




cdef class Tree(object):
    """
    Convenience class for creating/storing node relationships and values
    as arrays.
    """
    property nleaves:
        def __get__(self):
            cdef int n = 0
            for i in range(self.nnodes):
                if self.nchildren[i] == 0:
                    n += 1
            return n

    def __cinit__(self, root):
        # allocate attribute storage
        self.root = None
        cdef int i = 0
#        cdef Node n
        for i, n in enumerate(root.iternodes()):
            i += 1
        self.nnodes = i
        self.parent = np.empty(self.nnodes, dtype=np.intp)
        self.leftchild = np.empty(self.nnodes, dtype=np.intp)
        self.rightsib = np.empty(self.nnodes, dtype=np.intp)
        self.nchildren = np.empty(self.nnodes, dtype=np.int32)
        self.postorder = np.empty(self.nnodes, dtype=np.intp)
        self.length = np.empty(self.nnodes, dtype=np.double)
        self.label = []
        self.index(root)

    cpdef index(self, root=None):
        "call on __cinit__, or when topology changes but no. nodes is the same"
#        cdef Node n
        cdef Py_ssize_t i
        if root is None:
            root = self.root
        else:
            self.root = root
        for i, n in enumerate(root.iternodes()):
            n.ni = i
            self.parent[i] = n.parent.ni if n.parent else -1
            self.leftchild[i] = n.leftchild.ni if n.leftchild is not None else -1
            self.rightsib[i] = n.rightsib.ni if n.rightsib is not None else -1
            self.nchildren[i] = n.nchildren
            self.length[i] = n.length if n.length is not None else NAN
            self.label.append(n.label or '')

        for i, n in enumerate(root.postiter()):
            self.postorder[i] = n.ni

    cpdef double rootpathlen(self, Py_ssize_t i, Py_ssize_t j=0):
        cdef double x = 0
        while 1:
            if i == j:
                break
            x += self.length[i]
            i = self.parent[i]
        return x

def tree_from_ivy(ivytree):
    """
    Convert ivy Node to cython Tree
    """
    ivytree.rightsib = None
    for n in ivytree.iternodes():
        if n.children:
            n.leftchild = n.children[0]
        else:
            n.leftchild = None
        rightsib = None
        for c in n.children[::-1]: # Iterate through children backwards to assign rightsibs
            c.rightsib = rightsib
            rightsib = c
    out = Tree(ivytree)


# Example functions: calculate the number of tips for each node
# recursive array indexing
cpdef cladesizes1(Tree t, np.int_t[:] sizes, int node=0):
    cdef Py_ssize_t i, child
    cdef int nchildren = t.nchildren[node]
    sizes[node] = 0
    if nchildren == 0:
        sizes[node] = 1
        return
    child = t.leftchild[node]
    for i in range(nchildren):
        cladesizes1(t, sizes, child)
        sizes[node] += sizes[child]
        child = t.rightsib[child]

# recusive node traversal: a little more than 3x slower
cpdef cladesizes2(Node r, np.int_t[:] sizes):
    cdef Node child
    sizes[r.ni] = 0
    if r.nchildren == 0:
        sizes[r.ni] = 1
        return
    for child in r.iterchildren():
        cladesizes2(child, sizes)
        sizes[r.ni] += sizes[child.ni]

# nonrecursive array indexing - fastest!
# about 4x faster than cladesizes1
cpdef cladesizes3(Tree t, np.int_t[:] sizes):
    cdef Py_ssize_t i, j, nc, child, node
    for i in range(t.nnodes):
        node = t.postorder[i]
        nc = t.nchildren[node]
        if nc == 0:
            sizes[node] = 1
            continue
        sizes[node] = 0
        child = t.leftchild[node]
        for j in range(nc):
            sizes[node] += sizes[child]
            child = t.rightsib[child]

# with Python iteration over memoryview
# about 30x slower than cladesizes3
cpdef cladesizes4(Tree t, np.int_t[:] sizes):
    cdef Py_ssize_t nc, child, node
    for node in t.postorder:
        nc = t.nchildren[node]
        if nc == 0:
            sizes[node] = 1
            continue
        sizes[node] = 0
        child = t.leftchild[node]
        for j in range(nc):
            sizes[node] += sizes[child]
            child = t.rightsib[child]
