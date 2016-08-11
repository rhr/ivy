"""
Numerical integration using the GSL: https://goo.gl/ZOF5K3
Requires CythonGSL: https://github.com/twiecki/CythonGSL
"""
from __future__ import print_function
from cython_gsl cimport *
import numpy as np
cimport numpy as np


cdef int bisse(double t, double y[], double f[], void *params) nogil:
    cdef double * P = <double *>params
    cdef double lambda0 = P[0]
    cdef double lambda1 = P[1]
    cdef double mu0 = P[2]
    cdef double mu1 = P[3]
    cdef double q01 = P[4]
    cdef double q10 = P[5]

    cdef double D0 = y[0], D1 = y[1], E0 = y[2], E1 = y[3]
    f[0] = -(lambda0+mu0+q01)*D0 + q01*D1 + 2*lambda0*E0*D0
    f[1] = -(lambda1+mu1+q10)*D1 + q10*D0 + 2*lambda1*E1*D1
    f[2] = mu0-(mu0+q01+lambda0)*E0 + q01*E1 + lambda0*E0**2
    f[3] = mu1-(mu1+q10+lambda1)*E1 + q10*E0 + lambda1*E1**2
    return GSL_SUCCESS

cdef int classe(double t, double y[], double f[], void *params) nogil:
    # Classe integration based on formulae from Goldberg & Igic 2012
    # doi:10.1111/j.1558-5646.2012.01730.x
    cdef double * P = <double *>params
    cdef int nstate = <int> P[0]
    cdef int i,j,k

    cdef float lambda_ijk
    cdef float q_ij
    cdef float q_ij_dj
    cdef float q_ij_ej
    cdef float lambda_ijk_dj_ek
    cdef float lambda_ijk_ej_ek

    cdef float q,l,temp1,temp2

    for i in range(nstate):
        lambda_ijk = 0
        q_ij = 0
        q_ij_dj = 0
        q_ij_ej = 0
        lambda_ijk_dj_ek = 0
        lambda_ijk_ej_ek = 0
        for j in range(nstate):
            q = get_qij(P,i,j,nstate)
            q_ij += q
            q_ij_dj += q*y[j]
            q_ij_ej += q*y[nstate+j]
            for k in range(nstate):
                l = get_lambda(P,i,j,k,nstate)
                lambda_ijk += l
                lambda_ijk_dj_ek += l*(y[j]*y[nstate+k] + y[k]*y[nstate+j])
                lambda_ijk_ej_ek += l*y[nstate+j]*y[nstate+k]

        f[i] = -(lambda_ijk + q_ij + get_mu(P,i,nstate))*y[i] + q_ij_dj + lambda_ijk_dj_ek
        f[nstate+i] = -(lambda_ijk + q_ij + get_mu(P,i,nstate))*y[nstate+i] + q_ij_ej + get_mu(P,i,nstate) + lambda_ijk_ej_ek
    return GSL_SUCCESS


cdef float get_lambda(double * params, int i, int j, int k, int nstate) nogil:
    # unpack lambda param from parameter array
    if j>k:
        return 0
    else:
        return(params[1+i*sum_range_int(nstate+1) + j*nstate + k - sum_range_int(j+1)])

cdef float get_lambda_gil(double[:] params,int i, int j, int k, int nstate):
    if j>k:
        return 0
    else:
        return(params[1+i*sum(range(nstate+1)) + j*nstate + k - sum(range(j+1))])

cdef float get_mu(double * params, int i, int nstate) nogil:
    # unpack mu param from parameter array
    cdef int start = sum_range_int(nstate+1)*nstate
    return(params[1+start+i])


cdef float get_qij(double * params,int i,int j,int nstate) nogil:
    # unpack q param from parameter array
    cdef int start = sum_range_int(nstate+1)*nstate+nstate
    if i==j:
        return 0
    elif i > j:
        return(params[1+start+(i*nstate)-i + j])
    else:
        return(params[1+start+(i*nstate)-i + j - 1])

cdef int sum_range_int(long n) nogil:
    cdef int out = 0
    cdef int i
    for i in range(n):
        out += i
    return out


def integrate_bisse(double[:] params, double t1, double[:] li, double[:] E):
    cdef int ndim = 4
    cdef gsl_odeiv2_system sys

    sys.function = bisse
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-6 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-15 # absolute error
    cdef double epsrel = 0.0  # relative error

    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rkf45, hstart, epsabs, epsrel)
    cdef double t
    cdef double[:] y = np.empty(ndim, dtype=np.double)

    t = 0.0
    y[0] = li[0];y[1] = li[1];y[2] = E[0];y[3] = E[1]
    cdef int status
    status = gsl_odeiv2_driver_apply(d, &t, t1, &y[0])
    gsl_odeiv2_driver_free(d)
    return y

def integrate_classe(double[:] params, double t1, double[:] li, double[:] E):
    cdef int nstate = int(params[0])
    cdef int ndim = nstate*2
    cdef gsl_odeiv2_system sys

    sys.function = classe
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-8 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-12 # absolute error
    cdef double epsrel = 0.0  # relative error

    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rkf45, hstart, epsabs, epsrel)
    cdef int i
    cdef double t
    cdef double[:] y = np.empty(ndim, dtype=np.double)
    t = 0.0
    for i in range(nstate):
        y[i] = li[i]
        y[nstate + i] = E[i]
    cdef int status
    status = gsl_odeiv2_driver_apply(d, &t, t1, &y[0])
    gsl_odeiv2_driver_free(d)
    return y

def bisse_odeiv(root,data,params,condition_on_surv=True):
    cdef double lambda0 = params[0]
    cdef double lambda1 = params[1]
    cdef double mu0 = params[2]
    cdef double mu1 = params[3]
    cdef double q01 = params[4]
    cdef double q10 = params[5]
    cdef int k = 2
    rootp = np.array([0.5,0.5])
    nnode = len(root)
    fraclnl = np.zeros([nnode,k])
    for node in root.leaves():
        fraclnl[node.ni,data[node.label]] = 1.0
    E = np.zeros([nnode,k])
    for node in root.postiter():
        if not node.isleaf:
            childN = node.children[0]
            childM = node.children[1]
            DN = integrate_bisse(params,childN.length,fraclnl[childN.ni],E[childN.ni])
            DM = integrate_bisse(params,childM.length,fraclnl[childM.ni],E[childM.ni])
            fraclnl[node.ni,0] = DN[0] * DM[0] * lambda0
            fraclnl[node.ni,1] = DN[1] * DM[1] * lambda1
            E[node.ni,0] = DN[2]
            E[node.ni,1] = DN[3]
    if condition_on_surv:
        fraclnl[0] = fraclnl[0] / sum(rootp * np.array([lambda0,lambda1]) * (1-E[0])**2)
    return np.log(np.sum(fraclnl[0])/2)

def classe_likelihood(root,data,nstate,double [:] params,condition_on_surv=True):
    """
    The likelihood function for a classe model. This is where the main calculations
    for classe and all of its derivative models take place.
    """
    cdef int k = int(nstate)
    rootp = np.array([1.0/k]*k)
    nnode = len(root)
    fraclnl = np.zeros([nnode,k])
    for node in root.leaves():
        fraclnl[node.ni,data[node.label]] = 1.0
    E = np.zeros([nnode,k])
    for node in root.postiter():
        if not node.isleaf:
            childN = node.children[0]
            childM = node.children[1]
            DN = integrate_classe(params,childN.length,fraclnl[childN.ni],E[childN.ni])
            DM = integrate_classe(params,childM.length,fraclnl[childM.ni],E[childM.ni])
            for i in range(k):
                fraclnl[node.ni,i] = classe_node_calculation(params,i,DN,DM,k)
                E[node.ni,i] = DN[k+i]
    if condition_on_surv:
        surv = np.empty(k)
        for i in range(k):
            surv[i] = np.sum(params[1+i*sum(range(k+1)):1+(i+1)*sum(range(k+1))])
        fraclnl[0] = fraclnl[0] / sum(rootp * surv * (1-E[0])**2)
    return np.log(np.sum(fraclnl[0]*rootp))


cdef float classe_node_calculation(params,i,DN,DM,nstate):
    cdef float lsum = 0.0
    for j in range(nstate):
        for k in range(nstate):
            lsum += get_lambda_gil(params,i,j,k,nstate)*(DN[j]*DM[k] + DN[k]*DM[j])
    return 0.5*lsum
