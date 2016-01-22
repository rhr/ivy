# Hidden-rates model
import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit



def anc_recon_py(tree, chars, Q, p=None, pi="Fitzjohn"):
    """
    - Pure python version of anc recon code

    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips (tips can be switched
    to their true values in post-processing)


    """
    chartree = tree.copy()
    chartree.char = None; chartree.downpass_likelihood={}
    t = [node.length for node in chartree.descendants()]
    t = np.array(t, dtype=np.double)
    nchar = Q.shape[0]

    # Generating probability matrix for each branch
    if p is None:
        p = np.empty([len(t), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    cyexpokit.dexpm_tree_preallocated_p(Q, t, p) # This changes p in place


    for i, nd in enumerate(chartree.descendants()):
        nd.pmat = p[i] # Assigning probability matrices for each branch
        nd.downpass_likelihood = {}
        nd.char = None

    for i, lf in enumerate(chartree.leaves()):
        lf.char = chars[i] # Assigning character states to tips

    # Performing the downpass
    for node in chartree.postiter():
        if node.char is not None: # For tip nodes, likelihoods are 1 for observed state and 0 for the rest
            for state in range(nchar):
                node.downpass_likelihood[state]=0.0
            node.downpass_likelihood[node.char]=1.0
        else:
            for state in range(nchar):
                likelihoodStateN = []
                for ch in node.children:
                    likelihoodStateNCh = []
                    for chState in range(nchar):
                        likelihoodStateNCh.append(ch.pmat[state, chState] * ch.downpass_likelihood[chState]) #Likelihood for a certain state = p(stateBegin, stateEnd * likelihood(stateEnd))
                    likelihoodStateN.append(sum(likelihoodStateNCh))
                node.downpass_likelihood[state]=np.product(likelihoodStateN)
    # Performing the uppass (skipping the root)
    # Iterate over nodes in pre-order sequence
    for node in chartree.descendants():
        # Marginal is equivalent to information coming UP from the root * information coming DOWN from the tips
        node.marginal_likelihood = {}

        ### Getting uppass information for node of interest
        ###(partial uppass likelihood of parent * partial downpass likelihood of parent)
        ## Calculating partial downpass likelihood vector for parent
        node.parent.partial_down_likelihood = {}
        sibs = node.get_siblings()
        for state in range(nchar):
            partial_likelihoodN = [1.0] * nchar
            # Sister to this node
            for chState in range(nchar):
                for sib in sibs:
                    partial_likelihoodN[chState]*=(sib.downpass_likelihood[chState] * sib.pmat[state, chState])
            node.parent.partial_down_likelihood[state] = sum(partial_likelihoodN)
        ## Calculating partial uppass likelihood vector for parent
        node.parent.partial_up_likelihood = {}
        # If the parent is the root, there is no up-likelihood because there is
        # nothing "upwards" of the root. Set all likelihoods to 1 for identity
        if node.parent.isroot:
            for state in range(nchar):
                node.parent.partial_up_likelihood[state] = 1.0
        # If the parent is not the root, the up-likelihood is equal to the up-likelihoods coming from the parent
        else:
            for state in range(nchar):
                node.parent.partial_up_likelihood[state] = 0.0
                partial_uplikelihoodN = [1.0] * nchar
                for pstate in range(nchar):
                    for sib in node.parent.get_siblings():
                        partial_uplikelihoodNP = [0.0] * nchar
                        for sibstate in range(nchar):
                            partial_uplikelihoodNP[pstate] += sib.downpass_likelihood[sibstate] * sib.pmat[pstate,sibstate]
                        partial_uplikelihoodN[pstate] *= partial_uplikelihoodNP[pstate]
                    node.parent.partial_up_likelihood[state] += partial_uplikelihoodN[pstate] * node.parent.pmat[pstate, state]
        ### Putting together the uppass information and the downpass information
        uppass_information = {}
        for state in range(nchar):
            uppass_information[state] = node.parent.partial_up_likelihood[state] * node.parent.partial_down_likelihood[state]
        downpass_information = node.downpass_likelihood

        for state in range(nchar):
            node.marginal_likelihood[state] = uppass_information[state] * downpass_information[state]
    return chartree



def anc_recon(tree, chars, Q, p=None, pi="Fitzjohn",
              preallocated_arrays=None):
    """
    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips (tips can be switched
    to their true values in post-processing)


    """
    nchar = Q.shape[0]
    if preallocated_arrays is None:
        # Creating arrays to be used later
        preallocated_arrays = {}
        preallocated_arrays["t"] = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
        preallocated_arrays["charlist"] = range(Q.shape[0])
    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place

    if len(preallocated_arrays.keys())==2:
        # Creating more arrays
        nnode = len(tree)
        preallocated_arrays["nodelist"] = np.zeros((nnode, nchar+1))
        preallocated_arrays["childlist"] = np.zeros(nnode, dtype=object)
        leafind = [ n.isleaf for n in tree.postiter()]
        # Reordering character states to be in postorder sequence
        preleaves = [ n for n in tree.preiter() if n.isleaf ]
        postleaves = [n for n in tree.postiter() if n.isleaf ]
        postnodes = list(tree.postiter());prenodes = list(tree.preiter())
        postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
        # Filling in the node list. It contains all of the information needed
        # to calculate the likelihoods at each node
        for k,ch in enumerate(postChars):
            [ n for i,n in enumerate(preallocated_arrays["nodelist"]) if leafind[i] ][k][ch] = 1.0
            for i,n in enumerate(preallocated_arrays["nodelist"][:-1]):
                n[nchar] = postnodes.index(postnodes[i].parent)
                preallocated_arrays["childlist"][i] = [ nod.pi for nod in postnodes[i].children ]
        preallocated_arrays["childlist"][i+1] = [ nod.pi for nod in postnodes[i+1].children ]

        # Setting initial node likelihoods to 1.0 for calculations
        preallocated_arrays["nodelist"][[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0
        # Empty array to store root priors
        preallocated_arrays["root_priors"] = np.empty([nchar], dtype=np.double)
        preallocated_arrays["nodelist-up"] = preallocated_arrays["nodelist"].copy()
    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.cy_mk(preallocated_arrays["nodelist"], p, preallocated_arrays["charlist"])
    # "nodelist" contains the downpass likelihood vectors for each node in postorder sequence

    # Now that the downpass has been performed, we must perform the up-pass

    # The root will be assigned an up-pass likelihood vector equal to the
    # stationary distribution of the Q matrix
    root_marginal =  ivy.chars.mk.qsd(Q)

    # temp_array stores temporary values that are not reported in
    # the final reconstruction
    # The temp_array corresponds to nodes in REVERSE post-order sequence
    temp_array = np.zeros([len(tree), nchar+1])
    # The last column is an edgelist; each value corresponds to the index
    # of the parent of the node in the original nodelist. The root's
    # parent is set at 0.
    temp_array[:,-1] = preallocated_arrays["nodelist"][:,-1][::-1]

    # Skip the root and start at the first child
    ni = len(tree)-2
