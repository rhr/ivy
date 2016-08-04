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
    cdef double * P = <double *>params
    cdef int nstate = sizeof(y)/2

    for i in range(nstate):
        pass

cdef float get_lambda(double * params, int i, int j, int k, int nstate) nogil:
    if j>k:
        return 0
    else:
        

cdef float get_mu(double * params, int i, int nstate) nogil:
    cdef int start = 3*nstate-1
    return(params[start+i])

cdef float get_qij(params,i,j,nstate) nogil:
    cdef int start = 4*nstate-1
    if i != j:
        return(params[start+(i*nstate+j)])
    else:
        return 0



def integrate_bisse(double[:] params, double t1, double[:] li, double[:] E):
    cdef int ndim = 4
    cdef gsl_odeiv2_system sys

    sys.function = bisse
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-6 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-8 # absolute error
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
    cdef int ndim=4
    cdef gsl_odeiv2_system sys

    sys.function = classe
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-6 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-8 # absolute error
    cdef double epsrel = 0.0  # relative error

    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rkf45, hstart, epsabs, epsrel)
    cdef int i
    cdef double t
    cdef double[:] y = np.empty(ndim, dtype=np.double)
    t = 0.0
    y[0] = li[0]
    y[1] = li[1]
    y[2] = E[0]
    y[3] = E[1]
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
    print(np.asarray(fraclnl[0]))
    return np.log(np.sum(fraclnl[0])/2)


def classe_odeiv_2state(root,data,params,condition_on_surv=True):
    pars_in_order = ["lambda000","lambda001","lambda011","lambda111","lambda101","lambda100",
                     "mu0","mu1","q01","q10"]
    anagenic_pars = np.array([params[x] for x in pars_in_order])
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
            DN = integrate_classe(anagenic_pars,childN.length,fraclnl[childN.ni],E[childN.ni])
            DM = integrate_classe(anagenic_pars,childM.length,fraclnl[childM.ni],E[childM.ni])
            fraclnl[node.ni,0] = 0.5*(params["lambda000"]*(DN[0]*DM[0]+DN[0]*DM[0]) +
                                      params["lambda001"]*(DN[0]*DM[1]+DN[1]*DM[0]) +
                                      params["lambda011"]*(DN[1]*DM[1]+DN[1]*DM[1]))
            fraclnl[node.ni,1] = 0.5*(params["lambda100"]*(DN[0]*DM[0]+DN[0]*DM[0]) +
                                      params["lambda101"]*(DN[0]*DM[1]+DN[1]*DM[0]) +
                                      params["lambda111"]*(DN[1]*DM[1]+DN[1]*DM[1]))
            E[node.ni,0] = DN[2]
            E[node.ni,1] = DN[3]
    if condition_on_surv:
        fraclnl[0] = fraclnl[0] / sum(rootp * np.array([params["lambda000"]+params["lambda001"]+params["lambda011"],params["lambda111"]+params["lambda100"]+params["lambda101"]]) * (1-E[0])**2)
    return np.log(np.sum(fraclnl[0])/2)



def classe_lnl(root,data,anaparams,cladoparams):
    pass
