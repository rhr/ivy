import numpy as np

cdef extern:
    void f_dexpm(int nstates, double* H, double t, double* expH)
    void f_dexpm_wsp(int nstates, double* H, double t, int i, double* wsp,
                     double* expH)

def dexpm(q, double t):
    """
    Compute exp(q*t) and return the exponentiated array.

    q must be a C-contiguous numpy square array
    """
    assert len(q.shape)==2 and q.shape[0]==q.shape[1], 'q must be square'
    p = np.empty(q.shape, dtype=np.double, order='C')
    cdef double[:,::1] qview = q
    cdef double[:,::1] pview = p
    f_dexpm(q.shape[0], &qview[0,0], t, &pview[0,0])
    return p

def test_dexpm():
    q = np.array([[-1,1,0,0],
                  [0,-1,1,0],
                  [0,0,-1,1],
                  [0,0,0,0]], dtype=np.double, order='C')
    print 'q is:'
    print q
    print
    cdef double t = 1.0
    cdef int n = 4

    p = np.zeros((4,4), dtype=np.double, order='C')

    cdef double[:,::1] qview = q
    cdef double[:,::1] pview = p
    f_dexpm(n, &qview[0,0], t, &pview[0,0])
    print 'p is:'
    print p
    print

def test_dexpm_wsp():
    q = np.array([[-1,1,0,0],
                  [0,-1,1,0],
                  [0,0,-1,1],
                  [0,0,0,0]], dtype=np.double, order='C')
    print 'q is:'
    print q
    print
    cdef double t = 1.0
    cdef int n = 4
    ideg = 6
    wsp = np.empty(4*n*n + ideg + 1)

    p = np.zeros((4,4), dtype=np.double, order='C')

    cdef double[:,::1] qview = q
    cdef double[:,::1] pview = p
    cdef double[:] wspview = wsp
    f_dexpm_wsp(n, &qview[0,0], t, ideg, &wspview[0], &pview[0,0])
    print 'p is:'
    print p
    print

def test_dexpm_slice():
    q = np.array([
        [[-1,1,0,0],
         [0,-1,1,0],
         [0,0,-1,1],
         [0,0,0,0]],
        [[-2,2,0,0],
         [0,-2,2,0],
         [0,0,-2,2],
         [0,0,0,0]]], dtype=np.double, order='C')
    p = np.empty(q.shape, dtype=np.double, order='C')
    cdef double[:,:,:] qview = q
    cdef double[:,:,:] pview = p
    cdef double t = 1.0
    cdef int n = 4
    f_dexpm(n, &qview[0,0,0], t, &pview[0,0,0])
    return p
    
