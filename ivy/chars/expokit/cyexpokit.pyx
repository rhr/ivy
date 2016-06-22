# cython: profile=True

import numpy as np
cimport numpy as np
from libc.math cimport exp, log
from numpy.math cimport INFINITY
cimport cython

ctypedef np.double_t DTYPE_t

# Indexing arrays - what type to use?
# -----------------------------------
#
# TL;DR: int is probably fine; BUT
# 
# In Cython, for maximum correctness and portability, variables used
# for indexing arrays should be declared Py_ssize_t, not int. This
# raises a question: If you want to create/use a numpy array of
# indices, what type should it be? Py_ssize_t has no defined
# equivalent in numpy. The closest seems to be np.intp: see
# 
# https://github.com/scikit-learn/scikit-learn/wiki/C-integer-types:-the-missing-manual
#
# and
#
# https://github.com/numpy/numpy/issues/1654
#
# So, e.g:
#
# cdef np.ndarray[np.intp_t, ndim=1] my_index_array = np.arange(10, dtype=np.intp)
#
# (Note np.intp_t on the cdef side, np.intp on the Python side.)

cdef extern:
    void f_dexpm(int nstates, double* H, double t, double* expH) nogil
    void f_dexpm_wsp(int nstates, double* H, double t, int i,
                     double* wsp, double* expH) nogil

@cython.boundscheck(False)
cdef void dexpm3(double[:,:,:] q, double[:] t, Py_ssize_t[:] qi,
                 np.uint8_t[:] tmask,
                 double[:,:,:] p, int ideg, double[:] wsp) nogil:
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

@cython.boundscheck(False)
cdef void lndexpm3(double[:,:,:] q, double[:] t, Py_ssize_t[:] qi,
                   np.uint8_t[:] tmask,
                   double[:,:,:] p, int ideg, double[:] wsp) nogil:
    "same as dmexp3, but log-transforms p"
    cdef Py_ssize_t i, j, k
    cdef int nstates = q.shape[1]
    for i in range(t.shape[0]):
        if tmask[i]:
            f_dexpm_wsp(nstates, &q[qi[i],0,0], t[i], ideg, &wsp[0], &p[i,0,0])
            for j in range(nstates):
                for k in range(nstates):
                    p[i,j,k] = log(p[i,j,k])

def test_dexpm3():
    m = 3  # number of q matrices
    k = 4  # number of states
    q = np.zeros((m,k,k))  # 3-d 'stack' of q matrices
    for i in range(m):
        # fill the off-diagonals of each q matrix, incrementing:
        # 0.1, 0.2, ...
        a = q[i]
        a += 0.1*(i+1)
        # make the rows sum to zero
        np.fill_diagonal(a, 0)
        a[np.diag_indices_from(a)] = -a.sum(axis=1)
    t = np.ones(3)
    tmask = np.ones(3, dtype=np.uint8)
    qi = np.array([0,1,2], dtype=np.intp)
    
    cdef int ideg = 6
    wsp = np.empty(4*k*k+ideg+1)
    n = len(t)
    p = np.empty((n,k,k))
    
    # with all arrays allocated, can call dexpm3
    dexpm3(q, t, qi, tmask, p, ideg, wsp)
    return p

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
                if fraclnl[parent,ancstate] == -INFINITY:
                    fraclnl[parent,ancstate] = logsumexp(tmp)
                else:
                    fraclnl[parent,ancstate] += logsumexp(tmp)

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

def make_mklnl_func(root, data, int k, int nq, Py_ssize_t[:,:] qidx):
    cdef list nodes = list(root.iternodes())
    cdef nnodes = len(nodes)
    cdef np.ndarray postorder = np.array(
        [ nodes.index(n) for n in root.postiter() if n.children ], dtype=np.intp)
    cdef Py_ssize_t i, j, N = len(postorder)
    cdef np.ndarray t = np.array([ n.length for n in nodes ], dtype=np.double)
    cdef np.ndarray p = np.empty((nnodes, k, k), dtype=np.double)
    cdef int ideg = 6
    cdef np.ndarray wsp = np.empty(4*k*k+ideg+1)
    cdef np.ndarray qi = np.zeros(nnodes, dtype=np.intp)
    cdef tmask = np.ones(nnodes, dtype=np.uint8)
    cdef np.ndarray fraclnl = np.empty((nnodes, k), dtype=np.double)
    fraclnl.fill(-INFINITY)
    for lf in root.leaves():
        i = nodes.index(lf)
        fraclnl[i, data[lf.label]] = 0
    cdef np.ndarray tmp = np.empty(k)
    cdef int c = max([ len(n.children) for n in root if n.children ])
    cdef np.ndarray children = np.zeros((N, c), dtype=np.intp)
    children -= 1
    for i in range(len(postorder)):
        for j, child in enumerate(nodes[postorder[i]].children):
            children[i,j] = nodes.index(child)
    cdef np.ndarray q = np.zeros((nq,k,k), dtype=np.double)
    
    def f(double[:] params, Py_ssize_t[:,:] qidx=qidx):
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
        cdef Py_ssize_t r, a, b, c, d
        cdef double x = 0
        for r in range(qidx.shape[0]):
            a = qidx[r,0]; b = qidx[r,1]; c = qidx[r,2]; d = qidx[r,3]
            q[a,b,c] = params[d]
        for r in range(nq):
            for i in range(k):
                x = 0
                for j in range(k):
                    if i != j:
                        x -= q[r,i,j]
                q[r,i,i] = x
        dexpm3(q, t, qi, tmask, p, ideg, wsp)
        np.log(p, out=p)
        mklnl(fraclnl, p, k, tmp, postorder, children)
        return logsumexp(fraclnl[postorder[-1]])

    # attached allocated arrays to function object
    f.fraclnl = fraclnl
    f.q = q
    f.p = p
    f.qi = qi
    f.tmask = tmask
    f.postorder = postorder
    f.children = children
    f.t = t
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

    cdef np.ndarray[DTYPE_t, ndim=3] p = np.empty([len(t), q.shape[0], q.shape[1]], dtype = long, order="C")
    for i, blen in enumerate(t):
        dexpm_slice_log(q, blen, p, i)

def dexpm_tree(np.ndarray[dtype = DTYPE_t, ndim = 2] q, np.ndarray t):
    """
    Compute exp(q*t) for all branches on tree and return array of all
    p-matrices
    """
    ## assert q.shape[0]==q.shape[1], 'q must be square'

    ## assert (t > 0).all(), "All branch lengths must be greater than zero"

    cdef int i
    cdef double blen

    cdef np.ndarray[DTYPE_t, ndim=3] p = np.empty([len(t), q.shape[0], q.shape[1]], dtype = long, order="C")
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

def cy_mk(np.ndarray[dtype=DTYPE_t, ndim=2] nodelist,
          np.ndarray[dtype=DTYPE_t, ndim=3] p,
          int nchar):

    cdef int intnode
    cdef int ind
    cdef int ch
    cdef int st

    for intnode in sorted(set(nodelist[:-1,nchar])):

        nextli = nodelist[intnode]

        for ind in np.where(nodelist[:,nchar]==intnode)[0]:
            li = nodelist[ind]

            for ch in range(nchar):
                tmp = 0
                for st in range(nchar):
                    tmp += p[ind,ch,st] * li[st]

                nextli[ch] *= tmp

def cy_mk_log(np.ndarray[dtype=DTYPE_t, ndim=2] nodelist,
          np.ndarray[dtype=DTYPE_t, ndim=3] p,
          int nchar,
          np.ndarray[dtype=DTYPE_t, ndim=1] tmp_ar,
          np.ndarray[dtype=DTYPE_t, ndim=1] intnode_list,
          np.ndarray[dtype=np.int64_t, ndim=2] child_ar):

    cdef int intnode # "internal node"
    cdef int ind # index
    cdef int ch # parent character state
    cdef int st # child character state


    for intnode in intnode_list: # For each internal node (in postorder sequence)...
        nextli = nodelist[intnode]
        for ind in child_ar[intnode]: # For each child of this node...
            if not ind == -1: # -1 is the empty value for this array
                li = nodelist[ind]
                for ch in range(nchar):
                    for st in range(nchar):
                        tmp_ar[st] = p[ind,ch,st]+li[st] # Multiply child's likelihood by p-matrix
                    nextli[ch] += lse_cython(tmp_ar) # Sum of log-likelihoods of children

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


cpdef lse_cython(np.ndarray[DTYPE_t, ndim=1] a):
    """
    nbviewer.jupyter.org/gist/sebastien-bratieres/285184b4a808dfea7070
    Faster than scipy.misc.logsumexp
    """
    cdef int i
    cdef double result = 0.0
    cdef double largest_in_a = a[0]
    for i in range(1,a.shape[0]):
        if (a[i] > largest_in_a):
            largest_in_a = a[i]
    for i in range(a.shape[0]):
        result += exp(a[i] - largest_in_a)
    return largest_in_a + log(result)

cpdef lse_cython_ab(DTYPE_t a, DTYPE_t b):
    """
    The sum of two logs
    """
    cdef int i
    cdef DTYPE_t result = 0.0
    cdef DTYPE_t largest = a
    if a<b:
        largest = b
    result += exp(a - largest)
    result += exp(b - largest)
    return largest + log(result)
