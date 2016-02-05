import numpy as np
cimport numpy as np

DTYPE = np.double # Fixing a datatype for the arrays
ctypedef np.double_t DTYPE_t

cdef extern:
    void f_dexpm(int nstates, double* H, double t, double* expH)

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


def cy_mk(np.ndarray[dtype=DTYPE_t, ndim=2] nodelist,
          np.ndarray[dtype=DTYPE_t, ndim=3] p,
          list charlist):

    cdef int nchar = len(charlist)
    cdef int intnode
    cdef int ind
    cdef int ch
    cdef int st

    for intnode in sorted(set(nodelist[:-1,nchar])):

        nextli = nodelist[intnode]

        for ind in np.where(nodelist[:,nchar]==intnode)[0]:
            li = nodelist[ind]
            for ch in charlist:
                nextli[ch] *= sum([ p[ind][ch,st] for st in charlist ] * li[:-1])

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
                 np.ndarray[dtype=DTYPE_t, ndim=1] temp_dotprod):

    cdef int nchar = len(charlist)
    cdef int i
    cdef int intnode
    cdef int ind
    cdef int ch
    cdef int st

    cdef np.ndarray[dtype=DTYPE_t, ndim=1] l
    cdef int spi
    cdef int ppi
    # ------------- downpass
    for intnode in map(int, sorted(set(d_nl[:-1,nchar]))):
        nextli = d_nl[intnode]
        for chi, child in enumerate(childlist[intnode]):
            li = d_nl[child]
            p_li = p_nl[intnode][chi]
            for ch in charlist:
                p_li[ch] = sum([ p[child][ch,st] for st in charlist ]
                               * li[:nchar])
                nextli[ch] *= p_li[ch]
    cdef int root_posti = u_nl.shape[0] - 1

    # -------------- uppass
    for i,l in enumerate(u_nl):
        if i == 0:
            l[:nchar] = root_equil
            m_nl[i][:nchar] = (l[:nchar] * d_nl[-1][:nchar])
        else:
            spi = int(l[nchar+1])
            ppi = int(l[nchar])
            if ppi == root_posti:
                pp_nl[spi] = (p_nl[ppi].take(range(ci[ppi])+range(ci[ppi]+1,p_nl[ppi].shape[0]),0) * root_equil)
            else:
                np.dot(p[ppi].T, pp_nl[ppi], out=temp_dotprod)
                pp_nl[spi] = (p_nl[ppi].take(range(ci[ppi])+range(ci[ppi]+1,p_nl[ppi].shape[0]),0) * temp_dotprod)
            l[:nchar] = np.dot(p[spi].T, pp_nl[spi])

            ci[ppi] += 1

            m_nl[i][:nchar] = l[:nchar] * d_nl[l[nchar+1]][:nchar]

    return m_nl
