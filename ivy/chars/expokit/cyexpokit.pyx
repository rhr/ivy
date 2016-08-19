# cython: profile=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from numpy.math cimport INFINITY
cimport cython
import random
import pymc

DTYPE = np.double # Fixing a datatype for the arrays
ctypedef np.double_t DTYPE_t

cdef extern:
    void f_dexpm(int nstates, double* H, double t, double* expH)
    void f_dexpm_wsp(int nstates, double* H, double t, int i,
                     double* wsp, double* expH)

cimport numpy as np
import numpy as np


cdef void quicksort(Py_ssize_t[:] A, int hi, int lo=0):
    cdef int p
    if lo < hi:
        p = partition(A, hi, lo)
        quicksort(A, p-1, lo)
        quicksort(A, hi, p+1)

cdef int partition(Py_ssize_t[:] A, int hi, int lo=0):
    cdef pivot = A[hi]
    cdef int i = lo
    cdef int j
    for j in range(lo, hi):
        if A[j] <= pivot:
            A[i], A[j] = A[j], A[i]
            i += 1
    A[i], A[hi] = A[hi], A[i]
    return i


@cython.boundscheck(False)
cdef void dexpm3(double[:,:,:] q, double[:] t, Py_ssize_t[:] qi,
                 Py_ssize_t[:] tmask,
                 double[:,:,:] p, int ideg, double[:] wsp):
    """
    Compute transition probabilities exp(q*t) for a 'stack' of q
    matrices, over all values of t from a 1-d array, where q is selected
    from the stack by indices in qi. Uses pre-allocated arrays for
    intermediate calculations and output, to minimize overhead of
    repeated calls (e.g. for ML optimization or MCMC).
    Args:
        q (double[m,k,k]): stack of m square rate matrices of dimension k
        t (double[n]): 1-d array of times (branch lengths) of length n
        qi (int[n]): 1-d array indicating assigning q matrices to times
        tmask (bint[n]): 1-d array indicating which t values to process

        p (double[n,k,k]): stack of n square p matrices holding results
          of exponentiation, i.e., p[i] = exp(q[qi[i]]*t[i])
        ideg (int): used in expokit Fortran code; a good default is 6
        wsp (double[:]): expokit "workspace" array, must have
          min. length = 4*k*k+ideg+1
    """
    cdef Py_ssize_t i
    cdef int nstates = q.shape[1]
    for i in range(t.shape[0]):
        if tmask[i]:
            f_dexpm_wsp(nstates, &q[qi[i],0,0], t[i], ideg, &wsp[0], &p[i,0,0])

cdef void ln_dexpm3(double[:,:,:] q, double[:] t, Py_ssize_t[:] qi,
            Py_ssize_t[:] tmask,
            double[:,:,:] p, int ideg, double[:] wsp):
    """
    Compute transition probabilities exp(q*t) for a 'stack' of q
    matrices, over all values of t from a 1-d array, where q is selected
    from the stack by indices in qi. Uses pre-allocated arrays for
    intermediate calculations and output, to minimize overhead of
    repeated calls (e.g. for ML optimization or MCMC).
    Args:
        q (double[m,k,k]): stack of m square rate matrices of dimension k
        t (double[n]): 1-d array of times (branch lengths) of length n
        qi (int[n]): 1-d array indicating assigning q matrices to times
        p (double[n,k,k]): stack of n square p matrices holding results
          of exponentiation, i.e., p[i] = exp(q[qi[i]]*t[i])
        ideg (int): used in expokit Fortran code; a good default is 6
        wsp (double[:]): expokit "workspace" array, must have
          min. length = 4*k*k+ideg+1
    """
    cdef Py_ssize_t i,j,k
    cdef int nstates = q.shape[1]
    for i in range(t.shape[0]):
        if tmask[i]:
            f_dexpm_wsp(nstates, &q[qi[i],0,0], t[i], ideg, &wsp[0], &p[i,0,0])
            for j in range(nstates):
                for k in range(nstates):
                    p[i,j,k] = log(p[i,j,k])


def lndexpm3(double[:,:,:] q, double[:] t, Py_ssize_t[:] qi,
            double[:,:,:] p, int ideg, np.ndarray[dtype=DTYPE_t, ndim=1] wsp,
            np.ndarray[dtype=np.int64_t, ndim=1] pmask):
    """
    Compute transition probabilities exp(q*t) for a 'stack' of q
    matrices, over all values of t from a 1-d array, where q is selected
    from the stack by indices in qi. Uses pre-allocated arrays for
    intermediate calculations and output, to minimize overhead of
    repeated calls (e.g. for ML optimization or MCMC).
    Args:
        q (double[m,k,k]): stack of m square rate matrices of dimension k
        t (double[n]): 1-d array of times (branch lengths) of length n
        qi (int[n]): 1-d array indicating assigning q matrices to times
        p (double[n,k,k]): stack of n square p matrices holding results
          of exponentiation, i.e., p[i] = exp(q[qi[i]]*t[i])
        ideg (int): used in expokit Fortran code; a good default is 6
        wsp (double[:]): expokit "workspace" array, must have
          min. length = 4*k*k+ideg+1
    """
    cdef Py_ssize_t i,j,k
    cdef int nstates = q.shape[1]
    for i in range(t.shape[0]):
        if pmask[i]:
            f_dexpm_wsp(nstates, &q[qi[i],0,0], t[i], ideg, &wsp[0], &p[i,0,0])
            for j in range(nstates):
                for k in range(nstates):
                    p[i,j,k] = log(p[i,j,k])

def inds_from_switchpoint_p(np.ndarray switches_ni,
                          np.ndarray cladesize_preorder,
                          np.ndarray clades_postorder_preorder,
                          np.ndarray inds):
    cdef int i, s, n, nr
    nr = len(switches_ni)-1

    # cdef np.ndarray switches_key #Argsort creates new array. Is there a better way to do this?
    switches_key = np.argsort(cladesize_preorder[switches_ni])
    # all descendants of that node to that switchpoint's regime.
    # Switchpoints that are more downstream will overwrite upstream
    # switchpoints.
    for i,s in enumerate(switches_ni[switches_key][::-1]):
        for n in clades_postorder_preorder[s]:
            if n == -1:
                break
            if not n==len(cladesize_preorder)-1: # Ignore the root
                inds[n] = switches_key[nr-i]
@cython.boundscheck(False)
cdef inds_from_switchpoint(Py_ssize_t[:] switches_ni,
                          Py_ssize_t[:] switch_q_tracker,
                          Py_ssize_t[:,:] clades_preorder,
                          Py_ssize_t[:] qi):
    cdef Py_ssize_t i, s, n, nr
    qi[:] = 0 # Start by assigning all nodes to root q (0)
    nr = len(switches_ni)
    # We are going to sort the switchpoints by preorder index. Before we do that,
    # we need to keep track of which switchpoints are associated with which q-indices
    for i in range(nr):
        switch_q_tracker[switches_ni[i]] = i+1
    # Now we can sort the switchpoints
    quicksort(switches_ni,nr-1)
    for i in range(nr):
        s = switches_ni[::-1][i]
        for n in clades_preorder[s]:
            if n == -1:
                break
            if qi[n] == 0: # Skip over node if it has already been assigned
                qi[n] = switch_q_tracker[s] # Assign the correct q matrix to this index.
@cython.boundscheck(False)
def make_mklnl_func(root, data, int k, int nq, Py_ssize_t[:,:] qidx):
    root_copy = root.copy()
    for node in root_copy.descendants():
        node.meta["cached"]=False
        node.bisect_branch(1e-15,reindex=False)
    root_copy.reindex()
    root_copy.set_iternode_cache()
    for node in root_copy.iternodes():
        node.meta["cached"] = True
    # root_copy is a fully bisected copy of the original tree
    cdef list nodes = list(root_copy.iternodes())
    cdef int nnodes = len(nodes)
    cdef Py_ssize_t[:] postorder = np.array(
        [ nodes.index(n) for n in root_copy.postiter() if n.children ], dtype=np.intp)
    cdef Py_ssize_t i, j, N = len(postorder)
    cdef double[:] t = np.array([ n.length for n in nodes ], dtype=np.double)
    cdef double[:] t_copy = t.copy()
    cdef np.ndarray p = np.empty((nnodes, k, k), dtype=np.double)
    cdef int ideg = 6
    cdef double[:] wsp = np.empty(4*k*k+ideg+1)
    cdef Py_ssize_t[:] qi = np.zeros(nnodes, dtype=np.intp)
    cdef Py_ssize_t[:] tmask = np.ones(nnodes, dtype=np.intp)
    cdef double[:,:] fraclnl = np.empty((nnodes, k), dtype=np.double)
    fraclnl[:] = -INFINITY
    for lf in root_copy.leaves():
        i = nodes.index(lf)
        fraclnl[i, data[lf.label]] = 0
    for nd in root_copy.internals(): # Setting internal log-likelihoods to 0
        i = nodes.index(nd)
        fraclnl[i,:] = 0
    cdef double[:,:] fraclnl_copy = fraclnl.copy()
    cdef double[:] tmp = np.empty(k)
    cdef int c = max([ len(n.children) for n in root_copy if n.children ])
    cdef Py_ssize_t[:,:] children = np.zeros((N, c), dtype=np.intp)
    children[:] = -1
    for i in range(len(postorder)):
        for j, child in enumerate(nodes[postorder[i]].children):
            children[i,j] = nodes.index(child)
    cdef double[:,:,:] q = np.zeros((nq,k,k), dtype=np.double)

    cdef double[:] rootprior = np.empty([k])
    rootprior[:] = np.log(1.0/k) # Flat root prior


    cdef np.ndarray clades_preorder # preorder indices of all descendants of each node, in preorder sequence
    clades_preorder = np.zeros([nnodes,nnodes],dtype=np.intp)
    clades_preorder -= 1
    for n in range(nnodes):
        node = nodes[n]
        clades_preorder[n][:len(node)] = [x.ni for x in node]

    # Hackish way of keeping track of switchpoints after sorting them in inds_from_switchpoint
    cdef Py_ssize_t[:] switch_q_tracker = np.zeros([nnodes],dtype=np.intp) # keep track of which switchpoint nodes are associated with which q matrices
    cdef Py_ssize_t[:] switches_copy = np.zeros([nq-1],dtype=np.intp) # Store pre-sorted switches


    # Keeping track of previously calculated values to avoid redundant calculations
    cdef double[:] prev_t = t.copy()
    prev_t[:] = -1
    cdef double[:,:,:] prev_q = q.copy()
    prev_q[:] = -1
    cdef Py_ssize_t[:] prev_qi = qi.copy()
    prev_qi[:] = -1
    cdef Py_ssize_t[:] qdif = np.zeros([nq],dtype=np.intp) # Which q-matrices are different

    @cython.boundscheck(False)
    def f(double[:] params,Py_ssize_t[:] switches, double[:] lengths, Py_ssize_t[:,:] qidx=qidx):
        """
        params: array of free rate parameters, assigned to q by indices in qidx
        qidx columns:
            0, 1, 2 - index axes of q
            3 - index of params
        This scheme allows flexible specification of models. E.g.:
        Symmetric mk2:
            params = [0.2]; qidx = [[0,0,1,0],[0,1,0,0]]

        Asymmetric mk2:
            params = [0.2,0.6]; qidx = [[0,0,1,0],[0,1,0,1]]
        """
        cdef Py_ssize_t r, a, b, c, d, si
        cdef int nr = len(switches)
        # switches *= 2 # The corresponding ni of a node in the bisected tree
        #               # is 2x the original ni
        for r in range(nr):
            switches[r] = switches[r]*2
        switches_copy[:] = switches[:]
        cdef double x = 0

        for r in range(qidx.shape[0]):
            a = qidx[r,0]
            b = qidx[r,1]
            c = qidx[r,2]
            d = qidx[r,3]
            q[a,b,c] = params[d]
        for r in range(nq):
            for i in range(k):
                x = 0
                for j in range(k):
                    if i != j:
                        x -= q[r,i,j]
                q[r,i,i] = x
        t[:] = t_copy[:] # Reset branch lengths
        for r in range(nq-1): # Branch lengths for switchpoints
            si = switches[r]
            t[si] = lengths[r]
            t[si-1] -= lengths[r]

        # Q indices
        inds_from_switchpoint(switches_copy,switch_q_tracker,clades_preorder,qi)

        # tmask creation
        tmask[:]=0
        # q parameters changed?
        qdif[:] = 0
        for r in range(nq):
            for a in range(q.shape[1]):
                for b in range(q.shape[1]):
                    if (q[r,a,b] != prev_q[r,a,b]):
                        qdif[r] = 1
        for r in range(nnodes):
            if qdif[qi[r]]: # Q parameters changed
                tmask[r] = 1
            if qi[r] != prev_qi[r]: # Q index changed
                tmask[r] = 1
            if t[r] != prev_t[r]: # Branch lengths changed
                tmask[r] = 1


        # P matrix exponentiation
        # np.exp(p, out=p) # Necessary for handling p-matrices that aren't recalculated
        ln_dexpm3(q, t, qi, tmask, p, ideg, wsp)
        # np.log(p, out=p)

        # Likelihood calculation
        fraclnl[:] = fraclnl_copy[:]
        mklnl(fraclnl, p, k, tmp, postorder, children)

        # Storing parameters
        prev_t[:] = t[:]
        prev_q[:] = q[:]
        prev_qi[:] = qi[:]

        # The root likelihood is the first row of fraclnl
        for r in range(k):
            fraclnl[0][r] += rootprior[r]

        # switches /= 2 # Reset switches back to original value
        for r in range(nr):
            switches[r] = switches[r]/2
        return logsumexp(fraclnl[0])

    # attached allocated arrays to function object
    f.fraclnl = fraclnl
    f.fraclnl_copy = fraclnl_copy
    f.q = q
    f.p = p
    f.qi = qi
    f.tmask = tmask
    f.postorder = postorder
    f.children = children
    f.t = t
    f.t_copy = t_copy
    f.rootprior = rootprior
    f.clades_preorder = clades_preorder
    f.switch_q_tracker = switch_q_tracker
    return f

@cython.boundscheck(False)
def make_hrmlnl_func(root, data, int k, int nq, Py_ssize_t[:,:] qidx,findmin=False):
    cdef int nobschar = k/nq
    cdef list nodes = list(root.iternodes())
    cdef int nnodes = len(nodes)
    cdef Py_ssize_t[:] postorder = np.array(
        [ nodes.index(n) for n in root.postiter() if n.children ], dtype=np.intp)
    cdef Py_ssize_t i, j, N = len(postorder)
    cdef Py_ssize_t ki, ch
    cdef double[:] t = np.array([ n.length for n in nodes ], dtype=np.double)
    cdef np.ndarray p = np.empty((nnodes, k, k), dtype=np.double)
    cdef int ideg = 6
    cdef double[:] wsp = np.empty(4*k*k+ideg+1)
    cdef double[:,:] fraclnl = np.empty((nnodes, k), dtype=np.double)
    fraclnl[:] = -INFINITY
    for lf in root.leaves():
        i = nodes.index(lf)
        for ki in range(nq):
            ch = data[lf.label]+nobschar*ki
            fraclnl[i, ch] = 0 # Setting observed likelihoods to log(1)
    for nd in root.internals(): # Setting internal log-likelihoods to 0
        i = nodes.index(nd)
        fraclnl[i,:] = 0
    cdef double[:,:] fraclnl_copy = fraclnl.copy()
    cdef double[:] tmp = np.empty(k)
    cdef int c = max([ len(n.children) for n in root if n.children ])
    cdef Py_ssize_t[:,:] children = np.zeros((N, c), dtype=np.intp)
    children[:] = -1
    for i in range(len(postorder)):
        for j, child in enumerate(nodes[postorder[i]].children):
            children[i,j] = nodes.index(child)
    cdef double[:,:,:] q = np.zeros((1,k,k), dtype=np.double)
    cdef Py_ssize_t[:] qi = np.zeros(nnodes,dtype=np.intp)
    cdef Py_ssize_t[:] tmask = np.ones(nnodes,dtype=np.intp)

    cdef double[:] rootprior = np.empty([k])
    rootprior[:] = np.log(1.0/k) # Flat root prior
    @cython.boundscheck(False)
    def f(np.ndarray[dtype=double,ndim=1] params, grad=None, Py_ssize_t[:,:] qidx=qidx):
        """
        params: array of free rate parameters, assigned to q by indices in qidx
        qidx columns:
            0, 1, 2 - index axes of q
            3 - index of params
        This scheme allows flexible specification of models. E.g.:
        Symmetric mk2:
            params = [0.2]; qidx = [[0,0,1,0],[0,1,0,0]]

        Asymmetric mk2:
            params = [0.2,0.6]; qidx = [[0,0,1,0],[0,1,0,1]]
        """
        cdef Py_ssize_t r, a, b, c, d, si

        cdef double x = 0
        for r in range(qidx.shape[0]):
            a = qidx[r,0]
            b = qidx[r,1]
            c = qidx[r,2]
            q[0,a,b] = params[c]
        for i in range(k):
            x = 0
            for j in range(k):
                if i != j:
                    x -= q[0,i,j]
            q[0,i,i] = x
        # P matrix exponentiation
        ln_dexpm3(q, t, qi, tmask, p, ideg, wsp)

        # Likelihood calculation
        fraclnl[:] = fraclnl_copy[:]
        mklnl(fraclnl, p, k, tmp, postorder, children)


        # The root likelihood is the first row of fraclnl
        for r in range(k):
            fraclnl[0][r] += rootprior[r]

        if findmin:
            return -1*logsumexp(fraclnl[0])
        else:
            return logsumexp(fraclnl[0])

    # attached allocated arrays to function object
    f.qidx = qidx
    f.fraclnl = fraclnl
    f.fraclnl_copy = fraclnl_copy
    f.q = q
    f.p = p
    f.qi = qi
    f.tmask = tmask
    f.postorder = postorder
    f.children = children
    f.t = t
    f.rootprior = rootprior

    return f

cdef dexpm_slice_log(np.ndarray q, double t, np.ndarray p, int i):
    """
    Compute exp(q*t) for one branch on a tree and place result in pre-
    allocated array p

    Args:
        q (np.array): Q matrix
        t (np.array): Double indicating branch length
        p (np.array): Pre-allocated array to store results
        i (int): Index of branch length and p-array

    Returns:
        np.array: 3-D Array of P matrices
    """
    cdef DTYPE_t[:,::1] qview = q
    cdef DTYPE_t[:,::1] pview = p[i]
    f_dexpm(q.shape[0], &qview[0,0], t, &pview[0,0])
    np.log(p[i], out=p[i])

cdef dexpm_slice(np.ndarray q, double t, np.ndarray p, int i):
    """
    Compute exp(q*t) for one branch on a tree and place result in pre-
    allocated array p

    Args:
        q (np.array): Q matrix
        t (np.array): Double indicating branch length
        p (np.array): Pre-allocated array to store results
        i (int): Index of branch length and p-array

    Returns:
        np.array: 3-D Array of P matrices
    """
    cdef DTYPE_t[:,::1] qview = q
    cdef DTYPE_t[:,::1] pview = p[i]
    f_dexpm(q.shape[0], &qview[0,0], t, &pview[0,0])

def dexpm_tree_log(np.ndarray[dtype = DTYPE_t, ndim = 2] q, np.ndarray t):
    """
    Compute exp(q*t) for all branches on tree and return array of all
    p-matrices
    """
    assert q.shape[0]==q.shape[1], 'q must be square'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    cdef np.ndarray[DTYPE_t, ndim=3] p = np.empty([len(t), q.shape[0], q.shape[1]], dtype = DTYPE, order="C")
    for i, blen in enumerate(t):
        dexpm_slice_log(q, blen, p, i)


def dexpm_tree(np.ndarray[dtype = DTYPE_t, ndim = 2] q, np.ndarray t):
    """
    Compute exp(q*t) for all branches on tree and return array of all
    p-matrices
    """
    assert q.shape[0]==q.shape[1], 'q must be square'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    cdef np.ndarray[DTYPE_t, ndim=3] p = np.empty([len(t), q.shape[0], q.shape[1]], dtype = DTYPE, order="C")
    for i, blen in enumerate(t):
        dexpm_slice(q, blen, p, i)

    return p

def dexpm_tree_preallocated_p_log(np.ndarray[dtype=DTYPE_t, ndim=2] q, np.ndarray t, np.ndarray[dtype=DTYPE_t, ndim=3] p):
    assert q.shape[0]==q.shape[1], 'q must be square'
    assert np.allclose(q.sum(1), 0, atol= 1e-6), 'rows of q must sum to zero'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    for i, blen in enumerate(t):
        dexpm_slice(q, blen, p, i)
    np.log(p, out=p)
def dexpm_tree_preallocated_p(np.ndarray[dtype=DTYPE_t, ndim=2] q, np.ndarray t, np.ndarray[dtype=DTYPE_t, ndim=3] p):
    assert q.shape[0]==q.shape[1], 'q must be square'
    assert np.allclose(q.sum(1), 0, atol= 1e-6), 'rows of q must sum to zero'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    for i, blen in enumerate(t):
        dexpm_slice(q, blen, p, i)
def dexpm_treeMulti_preallocated_p(np.ndarray[dtype=DTYPE_t, ndim=3] q,
                     np.ndarray t, np.ndarray[dtype=DTYPE_t, ndim=3] p,
                     np.ndarray ind):
    assert q.shape[1]==q.shape[2], 'qs must be square'
    assert np.allclose(q.sum(2), 0, atol= 1e-6), 'rows of q must sum to zero'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    for i, blen in enumerate(t):
        dexpm_slice(q[ind[i]], blen, p, i)

def dexpm_treeMulti_preallocated_p_log(np.ndarray[dtype=DTYPE_t, ndim=3] q,
                     np.ndarray t, np.ndarray[dtype=DTYPE_t, ndim=3] p,
                     np.ndarray ind, np.ndarray pmask):
    assert q.shape[1]==q.shape[2], 'qs must be square'
    assert np.allclose(q.sum(2), 0, atol= 1e-6), 'rows of q must sum to zero'

    assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    for i, blen in enumerate(t):
        if pmask[i]:
            dexpm_slice_log(q[ind[i]], blen, p, i)

@cython.boundscheck(False)
cdef void mklnl(double[:,:] fraclnl,
                double[:,:,:] p,
                int k,
                double[:] tmp,
                Py_ssize_t[:] postorder,
                Py_ssize_t[:,:] children) nogil:
    """
    Standard Mk log-likelihood calculator.
    Args:
    fraclnl (double[m,k]): array to hold computed fractional log-likelihoods,
      where m = number of nodes, including leaf nodes; k = number of states
      * fraclnl[i,j] = fractional log-likelihood of node i for charstate j
      * leaf node values should be pre-filled, e.g. 0 for observed state,
        -np.inf everywhere else
      * Internal nodes should be filled with 0
      * this function calculates the internal node values (where a node
        could be a branch 'knee')
    p (double[m,k,k]): p matrix
    k (int): number of states
    tmp (double[k]): to hold intermediate values
    postorder (Py_ssize_t[n]): postorder array of n internal node indices (n < m)
    children (Py_ssize_t[n,c]): array of the children of internal nodes, where
      c = max number of children for any internal node
      * children[i,j] = index of jth child of node i
      * rows should be right-padded with -1 if the number of children < c
    """

    cdef Py_ssize_t i, parent, j, child, ancstate, childstate
    cdef Py_ssize_t c = children.shape[1]
    # For each internal node (in postorder sequence)...
    for i in range(postorder.shape[0]):
        # parent indexes the current internal node
        parent = postorder[i]
        # For each child of this node...
        for j in range(c):
            child = children[i,j] # fraclnl index of the jth child of node
            if child == -1: # -1 is the empty value for this array
                break
            for ancstate in range(k):
                for childstate in range(k):
                    # Multiply child's likelihood by p-matrix entry
                    tmp[childstate] = (p[child,ancstate,childstate] +
                                       fraclnl[child,childstate])
                # Sum of log-likelihoods of children
                fraclnl[parent,ancstate] += logsumexp(tmp)


def cy_mk(double[:,:] fraclnl,
                double[:,:,:] p,
                int k,
                double[:] tmp,
                Py_ssize_t[:] postorder,
                Py_ssize_t[:,:] children):
    """
    Standard Mk log-likelihood calculator.
    Args:
    fraclnl (double[m,k]): array to hold computed fractional log-likelihoods,
      where m = number of nodes, including leaf nodes; k = number of states
      * fraclnl[i,j] = fractional log-likelihood of node i for charstate j
      * leaf node values should be pre-filled, e.g. 0 for observed state,
        -np.inf everywhere else
      * Internal nodes should be filled with 0
      * this function calculates the internal node values (where a node
        could be a branch 'knee')
    p (double[m,k,k]): p matrix
    k (int): number of states
    tmp (double[k]): to hold intermediate values
    postorder (Py_ssize_t[n]): postorder array of n internal node indices (n < m)
    children (Py_ssize_t[n,c]): array of the children of internal nodes, where
      c = max number of children for any internal node
      * children[i,j] = index of jth child of node i
      * rows should be right-padded with -1 if the number of children < c
    """

    cdef Py_ssize_t i, parent, j, child, ancstate, childstate
    cdef Py_ssize_t c = children.shape[1]

    # For each internal node (in postorder sequence)...
    for i in range(postorder.shape[0]):
        # parent indexes the current internal node
        parent = postorder[i]
        # For each child of this node...
        for j in range(c):
            child = children[parent,j] # fraclnl index of the jth child of node
            if child == -1: # -1 is the empty value for this array
                break
            for ancstate in range(k):
                for childstate in range(k):
                    # Multiply child's likelihood by p-matrix entry
                    tmp[childstate] = (p[child,ancstate,childstate] +
                                       fraclnl[child,childstate])
                # Sum of log-likelihoods of children
                fraclnl[parent,ancstate] += logsumexp(tmp)
def cy_mk_log(
          DTYPE_t[:,:] nodelist,
          DTYPE_t[:,:,:] p,
          int nchar,
          double[:] tmp_ar,
          Py_ssize_t[:] intnode_list,
          Py_ssize_t[:,:] child_ar):


    cdef Py_ssize_t intnode,ind,ancstate,childstate,i # "internal node"

    for i in range(intnode_list.shape[0]): # For each internal node (in postorder sequence)...
        intnode = intnode_list[i]
        for ind in child_ar[intnode]: # For each child of this node...
            if not ind == -1: # -1 is the empty value for this array. -1 indicates that this node has no more children
                li = nodelist[ind]
                for ancstate in range(nchar):
                    for childstate in range(nchar):
                        tmp_ar[childstate] = p[ind,ancstate,childstate]+li[childstate] # Multiply child's likelihood by p-matrix
                    nodelist[intnode,ancstate] += logsumexp(tmp_ar) # Sum of log-likelihoods of children

def cy_anc_recon(np.ndarray[dtype=DTYPE_t, ndim=3] p,
                 np.ndarray[dtype=DTYPE_t, ndim=2] d_nl,
                 list charlist,
                 np.ndarray[dtype=object, ndim=1] childlist,
                 np.ndarray[dtype=DTYPE_t, ndim=2] u_nl,
                 np.ndarray[dtype=DTYPE_t, ndim=2] m_nl,
                 np.ndarray[dtype=DTYPE_t, ndim=2] pp_nl,
                 np.ndarray[dtype=DTYPE_t, ndim=3] p_nl,
                 np.ndarray[dtype=np.int64_t, ndim=1] ci,
                 np.ndarray[dtype=DTYPE_t, ndim=1] root_equil,
                 np.ndarray[dtype=DTYPE_t, ndim=1] temp_dotprod,
                 int nregime):

    cdef int nchar = len(charlist)
    cdef int i # Iterator
    cdef int intnode #integer node number
    cdef int ind # Index
    cdef int ch # Character state
    cdef int st # Character state

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] l
    cdef int spi # Self postorder index
    cdef int ppi # Parent postorder index
    # ------------- downpass
    for intnode in map(int, sorted(set(d_nl[:-1,nchar]))):
        nextli = d_nl[intnode]
        for chi, child in enumerate(childlist[intnode]):
            li = d_nl[child] # Likelihood
            p_li = p_nl[intnode][chi] # Parent likelihood
            for ch in charlist:
                p_li[ch] = sum([ p[child][ch,st] for st in charlist ]
                               * li[:nchar])
                nextli[ch] *= p_li[ch]
        nextli[:nchar] /= sum(nextli[:nchar])
    cdef int root_posti = u_nl.shape[0] - 1

    # -------------- uppass
    for i,l in enumerate(u_nl):
        if i == 0:
            l[:nchar] = root_equil
            m_nl[i][:nchar] = (l[:nchar] * d_nl[-1][:nchar])
            m_nl[i][:nchar] /= sum(m_nl[i][:nchar])
        else:
            spi = int(l[nchar+1]) # Self postorder index
            ppi = int(l[nchar]) # Parent postorder index
            if ppi == root_posti:
                pp_nl[spi] = (p_nl[ppi].take(list(range(ci[ppi]))+list(range(ci[ppi]+1,p_nl[ppi].shape[0])),0) * root_equil)
            else:
                np.dot(p[ppi].T, pp_nl[ppi], out=temp_dotprod)
                pp_nl[spi] = (p_nl[ppi].take(list(range(ci[ppi]))+list(range(ci[ppi]+1,p_nl[ppi].shape[0])),0) * temp_dotprod)
            l[:nchar] = np.dot(p[spi].T, pp_nl[spi])

            ci[ppi] += 1

            m_nl[i][:nchar] = l[:nchar] * d_nl[l[nchar+1]][:nchar]
            m_nl[i][:nchar] /= sum(m_nl[i][:nchar])

    return m_nl


@cython.boundscheck(False)
cdef double logsumexp(double[:] a) nogil:
    """
    nbviewer.jupyter.org/gist/sebastien-bratieres/285184b4a808dfea7070
    Faster than scipy.misc.logsumexp
    """
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double result = 0.0, largest_in_a = a[0]
    for i in range(1, n):
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(n):
        result += exp(a[i] - largest_in_a)
    return largest_in_a + log(result)


cdef double round_step_size(double step_size, double seg_size) nogil:
    """
    Round step_size to the nearest segment
    """
    if (step_size%seg_size) > (seg_size/2):
        return step_size + (seg_size-(step_size%seg_size)) + 1e-15
    else:
        return step_size - (step_size%seg_size) + 1e-15
def random_tree_location(seg_map):
    """
    Select random location on tree with uniform probability

    Returns:
        tuple: node and float, which represents the how far along the branch to
          the parent node the switch occurs on
    """
    cdef Py_ssize_t i
    i = random.choice(range(len(seg_map)))
    return seg_map[i]

def local_step(prev_location,double stepsize,double seg_size, double adaptive_scale_factor):
    cdef double cur_len,step_size
    cdef int direction
    cur_node = prev_location[0]
    cur_len = prev_location[1]
    direction = random.choice([-1,1])
    step_size = random.uniform(0,stepsize) * adaptive_scale_factor
    step_size = round_step_size(step_size, seg_size)
    while 1:
        if direction == 1: # Rootward
            if step_size < (cur_node.length - cur_len): # Stay on same branch
                new_location = (cur_node, cur_len+step_size+1e-15)
                break
            else: # If step goes past the parent
                if not cur_node.parent.isroot: # Jump to parent node
                    step_size = step_size - ((cur_node.length//seg_size)*seg_size-cur_len)
                    cur_node = cur_node.parent
                    cur_len = 1e-15
                else: # If parent is root, jump to a child of the root
                    step_size = step_size - ((cur_node.length//seg_size)*seg_size-cur_len)
                    valid_nodes = [n for n in cur_node.parent.children if n != cur_node]
                    cur_node = random.choice(valid_nodes)

                    cur_len = (cur_node.length//seg_size)*seg_size
                    direction = -1
        else: # tipward
            # Stay on same branch
            if step_size < cur_len:
                new_location = (cur_node, cur_len-step_size)
                break
            else: # Move to new branch
                if not cur_node.isleaf: #Move to child
                    step_size = step_size - cur_len + 1e-15
                    valid_nodes = cur_node.children
                    cur_node = random.choice(valid_nodes)
                    cur_len = (cur_node.length//seg_size)*seg_size
                else: # Bounce up from tip
                    step_size = step_size - cur_len
                    cur_len = 1e-15
                    direction = 1
    return new_location
