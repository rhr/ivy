"""
Categorical Markov models with k states.
"""
from __future__ import print_function
import numpy, scipy, random
import scipy.linalg
import scipy.optimize
from math import log, exp
rand = random.Random()
uniform = rand.uniform; expovariate = rand.expovariate

LARGE = 10e10 # large -lnL value used to bound parameter optimization

class Q:
    def __init__(self, k=2, layout=None):
        """
        Represents a square transition matrix with k states.
        
        'layout' is a square (k,k) array of integers that index free
        rate parameters (values on the diagonal are ignored).  Cells
        with value 0 will have the first rate parameter, 1 the
        second, etc.
        """
        self.k = k
        self.range = range(k)
        self.offdiag = array(numpy.eye(k)==0, dtype=numpy.int)
        if layout is None:
            layout = zeros((k,k), numpy.int)
        self.layout = layout*self.offdiag

    def fill(self, rates):
        m = numpy.take(rates, self.layout)*self.offdiag
        v = m.sum(1) * -1
        for i in self.range:
            m[i,i] = v[i]
        return m

    def default_priors(self):
        p = 1.0/self.k
        return [p]*self.k

def sample_weighted(weights):
    u = uniform(0, sum(weights))
    x = 0.0
    for i, w in enumerate(weights):
        x += w
        if u < x:
            break
    return i

def conditionals(root, data, Q):
    nstates = Q.shape[0]
    states = range(nstates)
    nodes = [ x for x in root.postiter() ]
    nnodes = len(nodes)
    v = zeros((nnodes,nstates))
    n2i = {}
    
    for i, n in enumerate(nodes):
        n2i[n] = i
        if n.isleaf:
            state = data[n.label]
            try:
                state = int(state)
                v[i,state] = 1.0
            except ValueError:
                if state == '?' or state == '-':
                    v[i,:] += 1/float(nstates)
        else:
            Pv = [ (expm(Q*child.length)*v[n2i[child]]).sum(1)
                   for child in n.children ]
            v[i] = numpy.multiply(*Pv)
            # fossils
            state = None
            if n.label in data:
                state = int(data[n.label])
            elif n in data:
                state = int(data[n])
            if state != None:
                for s in states:
                    if s != state: v[i,s] = 0.0
            
    return dict([ (n, v[i]) for n,i in n2i.items() ])

def contrasts(root, data, Q):
    cond = conditionals(root, data, Q)
    d = {}
    for n in root.postiter(lambda x:x.children):
        nc = cond[n]; nc /= sum(nc)
        diff = 0.0
        for child in n.children:
            cc = cond[child]; cc /= sum(cc)
            diff += numpy.sum(numpy.abs(nc-cc))
        d[n] = diff
    return d

def lnL(root, data, Q, priors):
    d = conditionals(root, data, Q)
    return numpy.log(sum(d[root]*priors))

def optimize(root, data, Q, priors=None):
    Qfill = Q.fill
    if priors is None: priors = Q.default_priors()
    def f(params):
        if (params<0).any(): return LARGE
        return -lnL(root, data, Qfill(params), priors)
        
    # initial parameter values
    p = [1.0]*len(set(Q.layout.flat))

    v = scipy.optimize.fmin_powell(
        f, p, full_output=True, disp=0, callback=None
        )
    params, neglnL = v[:2]
    if neglnL == LARGE:
        raise Exception("ConvergenceError")
    return params, neglnL

def sim(root, n2p, s0, d=None):
    if d is None:
        d = {root:s0}
    for n in root.children:
        v = n2p[n][s0]
        i = sample_weighted(v)
        d[n] = i
        sim(n, n2p, i, d)
    return d

def stmap(root, states, ancstates, Q, condition_on_success):
    """
    This and its dependent functions below need testing and
    optimization.
    """
    results = []
    for n in root.descendants():
        si = ancstates[n.parent]
        sj = ancstates[n]
        v = simulate_on_branch(states, si, sj, Q, n.length,
                               condition_on_success)
        print(n, si, sj)
        if v:
            results.append(v)
        else:
            return None
    return results

def simulate_on_branch(states, si, sj, Q, brlen, condition_on_success):
    point = 0.0
    history = [(si, point)]
    if si != sj:  # condition on one change occurring
        lambd = -(Q[si,si])
        U = uniform(0.0, 1.0)
        # see appendix of Nielsen 2001, Genetics
        t = brlen - point
        newpoint = -(1.0/lambd) * log(1.0 - U*(1.0 - exp(-lambd * t)))
        newstate = draw_new_state(states, Q, si)
        history.append((newstate, newpoint))
        si = newstate; point = newpoint
    while 1:
        lambd = -(Q[si,si])
        rv = expovariate(lambd)
        newpoint = point + rv

        if newpoint <= brlen:  # state change along branch
            newstate = draw_new_state(states, Q, si)
            history.append((newstate, newpoint))
            si = newstate; point = newpoint
        else:
            history.append((si, brlen))
            break
                
    if si == sj or (not condition_on_success): # success
        return history

    return None
        
def draw_new_state(states, Q, si):
    """
    Given a rate matrix Q, a starting state si, and an ordered
    sequence of states, eg (0, 1), draw a new state sj with
    probability -(qij/qii)
    """
    Qrow = Q[si]
    qii = Qrow[si]
    qij_probs = [ (x, -(Qrow[x]/qii)) for x in states if x != si ]
    uni = uniform(0.0, 1.0)
    val = 0.0
    for sj, prob in qij_probs:
        val += prob
        if uni < val:
            return sj
    
def sample_ancstates(node, states, conditionals, n2p, fixed={}):
    """
    Sample ancestral states from their conditional likelihoods
    """
    ancstates = {}
    for n in node.preiter():
        if n in fixed:
            state = fixed[n]
        else:
            cond = conditionals[n]

            if n.parent:
                P = n2p[n]
                ancst = ancstates[n.parent]
                newstate_Prow = P[ancst]
                cond *= newstate_Prow

            cond /= sum(cond)

            rv = uniform(0.0, 1.0)
            v = 0.0
            for state, c in zip(states, cond):
                v += c
                if rv < v:
                    break
        ancstates[n] = state

    return ancstates
