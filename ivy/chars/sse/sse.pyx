"""
Numerical integration using the GSL: https://goo.gl/ZOF5K3
Requires CythonGSL: https://github.com/twiecki/CythonGSL
"""
from __future__ import print_function
from cython_gsl cimport *
import numpy as np
cimport numpy as np
from libc.math cimport exp, log
import cython

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

@cython.boundscheck(False)
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

    cdef float q,l

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


@cython.boundscheck(False)
cdef float get_lambda(double * params, int i, int j, int k, int nstate) nogil:
    # unpack lambda param from parameter array
    if j>k:
        return 0
    else:
        return(params[1+i*sum_range_int(nstate+1) + j*nstate + k - sum_range_int(j+1)])
@cython.boundscheck(False)
cdef float get_lambda_gil(double[:] params,int i, int j, int k, int nstate):
    if j>k:
        return 0
    else:
        return(params[1+i*sum(range(nstate+1)) + j*nstate + k - sum(range(j+1))])
@cython.boundscheck(False)
cdef float get_mu(double * params, int i, int nstate) nogil:
    # unpack mu param from parameter array
    cdef int start = sum_range_int(nstate+1)*nstate
    return(params[1+start+i])

@cython.boundscheck(False)
cdef float get_qij(double * params,int i,int j,int nstate) nogil:
    # unpack q param from parameter array
    cdef int start = sum_range_int(nstate+1)*nstate+nstate
    if i==j:
        return 0
    elif i > j:
        return(params[1+start+(i*nstate)-i + j])
    else:
        return(params[1+start+(i*nstate)-i + j - 1])
@cython.boundscheck(False)
cdef int sum_range_int(long n) nogil:
    cdef int out = 0
    cdef int i
    for i in range(n):
        out += i
    return out


cdef integrate_bisse(double[:] params, double t1, double[:] li, double[:] E):
    cdef int ndim = 4
    cdef gsl_odeiv2_system sys

    sys.function = bisse
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-6 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-12 # absolute error
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

@cython.boundscheck(False)
cdef void integrate_classe(double[:] params, double t1, double[:] li, double[:] E, double[:] y):
    cdef int nstate = int(params[0])
    cdef int ndim = nstate*2
    cdef gsl_odeiv2_system sys

    sys.function = classe
    sys.dimension = ndim
    sys.params = <void *>&params[0]

    cdef double hstart = 1e-8 # initial step size
    # keep the local error on each step within:
    cdef double epsabs = 1e-10 # absolute error
    cdef double epsrel = 0.0  # relative error

    cdef gsl_odeiv2_driver * d = gsl_odeiv2_driver_alloc_y_new(
        &sys, gsl_odeiv2_step_rkf45, hstart, epsabs, epsrel)
    cdef int i
    cdef double t
    t = 0.0
    for i in range(nstate):
        y[i] = li[i]
        y[nstate + i] = E[i]
    cdef int status
    status = gsl_odeiv2_driver_apply(d, &t, t1, &y[0])
    gsl_odeiv2_driver_free(d)
    # Results stored in y array

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
    return log(np.sum(fraclnl[0])/2)

@cython.boundscheck(False)
def make_classe(root,data,nstate=None,condition_on_surv=True,pi="Equal",pi_given=None):
    """
    Create a classe likelihood function to be optimized with nlopt or used
    in a Bayesian analysis

    Args:
        root (Node): Root of tree to perform the analysis on. MUST BE BIFURCATING
        data (dict): Dictionary mapping node tip labels to character states.
          Character states must be represented by ints starting at index 0.
        nstate (int): Number of states. Can be more than the number of
          states indicated by data. Defaults to number of states in data.
        condition_on_surv (bool): Whether to condition the likelihood on the
          survival of the clades and speciation of subtending clade. See Nee et al. 1994.
        pi (str): Behavior at the root. Defaults to "Equal". Possible values:
          Equal: Flat prior weighting all states equally
          Equilibrium: Weight by equilibrium distribution of model (Maddison et al. 2007)
          Fitzjohn: Weight states by relative probability of observing data (Fitzjohn et al 2009)
          Given: Weight by given pi values.
        pi_given (np.array): If pi = "Given", use this as the weighting at the root.
          Must sum to 1.
    Returns:
        function: Function that takes an array of parameters as input.
    """
    cdef int k # Number of states
    if nstate is None:
        k = len(set(data.values))
    else:
        k = int(nstate)
    cdef int nnode = len(root) # Number of nodes
    cdef list nodes = list(root.iternodes()) # List of nodes
    cdef double[:,:] D_lnl = np.zeros([nnode,k]) # Likelihood array of diversification rates
    cdef double[:,:] E_lnl = np.zeros([nnode,k]) # Likelihood array of extinction rates
    for leaf in root.leaves():
        D_lnl[leaf.ni,data[leaf.label]] = 1.0
    cdef double[:] logcomp = np.zeros([nnode]) # Log-compensation values to prevent underflow
    cdef double[:,:] D_lnl_copy = D_lnl.copy() # Copy to refresh values between calls

    cdef Py_ssize_t [:] postorder = np.array([n.ni for n in root.postiter() if not n.isleaf],dtype=np.intp) # Array of node indices in postorder sequence
    cdef double [:] t = np.array([n.length for n in root],dtype=np.double) # branch lengths

    cdef Py_ssize_t[:,:] children = np.zeros([nnode,2],dtype=np.intp)
    for i in range(len(postorder)):
        for j, child in enumerate(nodes[postorder[i]].children):
            children[i,j] = nodes.index(child)


    cdef int nparam = sum_range_int(k+1)*k + k**2
    cdef double [:] ode_params = np.zeros([nparam+1]) # The parameter array given to the integration function also has to contain the number of states, so it's one longer.
    ode_params[0] = np.double(k)
    cdef rootp = np.zeros([k])
    if pi == "Given":
        rootp[:] = pi_given[:]
    elif pi == "Equal":
        rootp[:] = 1.0/k
    # TODO: other root likelihoods
    cdef double[:] tmp = np.zeros([k])
    cdef double[:] surv = np.empty(k) # store calculations for conditioning on survival
    cdef double[:] y = np.empty([k*2])
    cdef double[:] DN = np.zeros([k*2])
    cdef double[:] DM = np.zeros([k*2])
    def f(np.ndarray[dtype=np.double_t,ndim=1] params, grad=None):
        """
        params takes on a very specific form: first all lambda values are listed
        in order of 000, 001, 011, 100, 101, 111 etc, then all mu values are listed,
        then all q values are listed in the form 01, 10, etc.

        grad=None exists for compatibility with nlopt.
        """
        cdef Py_ssize_t i,j
        cdef int nlam = sum_range_int(k+1)

        D_lnl[:] = D_lnl_copy[:]
        for i in range(1,ode_params.shape[0]):
            ode_params[i] = params[i-1] # Fill in parameter array

        # Perform the likelihood calculation
        classe_likelihood(D_lnl,E_lnl,t,postorder,children,k,logcomp,ode_params,y,DN,DM)


        if condition_on_surv:
            for i in range(k):
                surv[i] = c_sum(ode_params[1+i*nlam:1+(i+1)*nlam],nlam)
                tmp[i] = (1.0-E_lnl[0,i])**2
            for i in range(k):
                D_lnl[0,i] = D_lnl[0,i] / c_sum(rootp * surv * tmp, k)
        return log(c_sum(D_lnl[0]*rootp,k)) + c_sum(logcomp,nnode)
    f.D_lnl = D_lnl
    f.E_lnl = E_lnl
    f.D_lnl_copy = D_lnl_copy
    f.postorder = postorder
    f.children = children
    f.t = t
    f.logcomp = logcomp
    f.surv = surv
    f.tmp = tmp
    f.y = y
    f.DN = DN
    f.DM = DM
    return f
@cython.boundscheck(False)
cdef double c_sum(double[:] x, Py_ssize_t l) nogil:
    cdef Py_ssize_t i
    cdef double out = 0.0
    for i in range(l):
        out += x[i]
    return(out)

@cython.boundscheck(False)
cdef void classe_likelihood(double[:,:] D_lnl,
                      double[:,:] E_lnl,
                      double[:] t,
                      Py_ssize_t[:] postorder,
                      Py_ssize_t[:,:] children,
                      Py_ssize_t k,
                      double[:] logcomp,
                      double [:] params,
                      double [:] y,
                      double [:] DN,
                      double [:] DM,
                      ):
    """
    The likelihood function for a classe model. This is where the main calculations
    for classe and all of its derivative models take place.
    """
    cdef Py_ssize_t i,j,childN, childM, state, parent
    cdef double z


    for i in range(postorder.shape[0]):
        parent = postorder[i]

        childN = children[i,0]
        childM = children[i,1]
        integrate_classe(params,t[childN],D_lnl[childN],E_lnl[childN],y)
        DN[:] = y[:]
        integrate_classe(params,t[childM],D_lnl[childM],E_lnl[childM],y)
        DM[:] = y[:]
        for state in range(k):
            D_lnl[parent,state] = classe_node_calculation(params,state,DN,DM,k)
            E_lnl[parent,state] = DN[k+state]
        z = c_sum(D_lnl[parent],k)
        logcomp[parent] = log(z)
        for state in range(k):
            D_lnl[parent,state] /= z


@cython.boundscheck(False)
cdef float classe_node_calculation(double [:] params,Py_ssize_t i,double[:] DN,double[:] DM, Py_ssize_t nstate):
    cdef float lsum = 0.0
    cdef Py_ssize_t j,k
    for j in range(nstate):
        for k in range(nstate):
            lsum += get_lambda_gil(params,i,j,k,nstate)*(DN[j]*DM[k] + DN[k]*DM[j])
    return 0.5*lsum



def param_dict_to_list(paramdict):
    """
    Take human-readable lambda, mu, and q params and translate them into param
    array for likelihood function

    Args:
      paramdict (dict): dict containing the following arrays:
        "lambda": A k x k x k array. The first dimension corresponds
          to the state of the parent, the second dimension corresponds to
          the state of the first child, and the third dimension corresponds to the
          state of the second child. Any value where the index of the first child
          is higher than that of the second is ignored.
          Example:lambda0 = np.array([[0.3,0.1],
                                      [0.0,0.01]])
                  lambda1 = np.array([[0.01,0.1],
                                      [0.0,0.2]])
                  paramdict["lambda"] = np.array([lambda0,lambda1])
        "mu": A 1 dimensional array of length k. Contains the extinction rates
          for each state.
          Example: paramdict["mu"] = np.array([0.01,0.01])
        "q": A k x k array. The row number corresponds to the rootward state
          and the column number corresponds to the tipward state. Diagonal
          values are ignored.
          Example: paramdict["q"] = np.array([[0,0.2],
                                             [0.1,0]])
          (0.2 is the rate from 0 to 1)

    """
    lambda_ar = paramdict["lambda"]
    mu_ar = paramdict["mu"]
    q_ar = paramdict["q"]

    k = mu_ar.shape[0]
    nparam = sum(range(k+1))*k + k**2
    params = np.zeros([nparam])

    # Lambda params
    count = 0
    while count < sum(range(k+1))*k:
        for i in range(k):
            for j in range(k):
                for l in range(k):
                    if not j>l:
                        params[count] = lambda_ar[i,j,l]
                        count+=1
    # Mu params
    params[count:count+k] = mu_ar
    count += k
    # Q params
    while count < nparam:
        for i in range(k):
            for j in range(k):
                if i!=j:
                    params[count] = q_ar[i,j]
                    count += 1
    return params

def param_list_to_dict(params,k):
    """
    Take param array and translate it to human-readable format
    Args:
       params (np.array): Array of parameters as fitted by fit_classe
       k (int): Number of character states
    Returns:
       dict: parameters sorted into a human-readable format. See param_dict_to_list
       for details.
    """
    lambda_ar = np.zeros([k,k,k])
    mu_ar = np.zeros([k])
    q_ar = np.zeros([k,k])
    nparam = len(params)
    # Lambda params
    count = 0
    while count < sum(range(k+1))*k:
        for i in range(k):
            for j in range(k):
                for l in range(k):
                    if not j>l:
                        lambda_ar[i,j,l] = params[count]
                        count+=1
    # mu params
    mu_ar[:] = params[count:count+k]
    count += k

    # Q params
    while count < nparam:
        for i in range(k):
            for j in range(k):
                if i!=j:
                    q_ar[i,j] = params[count]
                    count += 1
    return({"q":q_ar,"mu":mu_ar,"lambda":lambda_ar})
