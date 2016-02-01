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
            uppass_information[state] = node.parent.partial_down_likelihood[state] * node.parent.partial_up_likelihood[state]
        downpass_information = node.downpass_likelihood

        for state in range(nchar):
            node.marginal_likelihood[state] = 0
            for pstate in range(nchar):
                node.marginal_likelihood[state] += uppass_information[pstate] * node.pmat[pstate, state]
            node.marginal_likelihood[state] *= downpass_information[state]
    return chartree



def anc_recon(tree, chars, Q, p=None, pi="Fitzjohn",
              preallocated_arrays=None):
    """
    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips

    """
    nchar = Q.shape[0]
    if preallocated_arrays is None:
        # Creating arrays to be used later
        objs = ["down_nodelist","up_nodelist","partial_nodelist","childlist","t"]
        preallocated_arrays = {k:v for k,v in zip(objs, _create_ancrecon_nodelist(tree, chars))}
        preallocated_arrays["charlist"] = range(Q.shape[0])
        # Empty array to store root priors
        preallocated_arrays["root_priors"] = np.empty([nchar], dtype=np.double)
    # Calculating the likelihoods for each node in post-order sequence
    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place
    # ------------------- Performing the down-pass -----------------------------
    for intnode in map(int, sorted(set(preallocated_arrays["down_nodelist"][:-1,nchar]))):
        nextli = preallocated_arrays["down_nodelist"][intnode]
        for chi, child in enumerate(preallocated_arrays["childlist"][intnode]):
            li = preallocated_arrays["down_nodelist"][child]
            p_li = preallocated_arrays["partial_nodelist"][intnode][chi]
            for ch in preallocated_arrays["charlist"]:
                p_li[ch] = sum([ p[child][ch,st] for st in preallocated_arrays["charlist"] ] * li[:nchar])
                nextli[ch] *= p_li[ch]
    # "downpass_likelihood" contains the downpass likelihood vectors for each node in postorder sequence
    # Now that the downpass has been performed, we must perform the up-pass
    # ------------------- Performing the up-pass -------------------------------
    # The up-pass likelihood at each node is equivalent to information coming
    # up from the root * information coming down from the tips

    # Each node has the following:
    # Uppass_likelihood (set to the marginal for the root)
    # Partial likelihood for each child node
    # The final two columns of up_nodelist point to the
    # postorder index numbers of the parent and self node, respectively

    # child_masks containsan array of the children to use for calculating
    # partial likelihood of the next child of that node. All parents
    # start out with excluding the first child that appears (for calculating
    # marginal likelihood of that child)
    child_masks = np.ones(preallocated_arrays["partial_nodelist"].shape, dtype=bool)
    child_masks[:,0,:].fill(False)

    root_posti = preallocated_arrays["up_nodelist"].shape[0] - 1

    # The parent's partial likelihood without current node
    partial_parent_likelihoods = np.zeros([preallocated_arrays["up_nodelist"].shape[0],nchar])

    for i,l in enumerate(preallocated_arrays["up_nodelist"]):
        # Uppass information for node
        if i == 0:
            # Set root node to be equal to the marginal
            l[:nchar] = ivy.chars.discrete.qsd(Q)
        else:
            parent = int(l[nchar]) # the parent's POSTORDER index
            parent_partial_up = preallocated_arrays["partial_nodelist"][parent][child_masks[parent]]
            child_masks[parent] = np.roll(child_masks[parent], 1, 0) # Roll child masks so that next likelihood calculated for this parent uses correct children
            if parent == root_posti:
                parent_partial_down = np.ones(nchar)
            else:
                parent_partial_down = np.dot(p[parent].T, partial_parent_likelihoods[parent])

            partial_parent_likelihoods[int(l[nchar+1])] =  parent_partial_up * parent_partial_down
            uppass_information = np.dot(p[l[nchar+1]], parent_partial_up * parent_partial_down)
            # Downpass information for node
            downpass_information = preallocated_arrays["down_nodelist"][l[nchar+1]][:nchar]

            l[:nchar] = uppass_information * downpass_information
    return preallocated_arrays["up_nodelist"]

def _create_ancrecon_nodelist(tree, chars):
    """
    Create nodelist. For use in anc_recon function

    Returns edgelist of nodes in postorder sequence, edgelist of nodes in preorder sequence,
    partial likelihood vector for each node's children, childlist, and branch lengths.
    """
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nchar = len(set(chars))
    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    prenodes = list(tree.preiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    down_nodelist = np.zeros([nnode, nchar+1])
    up_nodelist = np.zeros([nnode, nchar+2])
    childlist = np.zeros(nnode, dtype=object)
    leafind = [ n.isleaf for n in tree.postiter()]

    # --------- Down nodelist
    for k,ch in enumerate(postChars):
        [ n for i,n in enumerate(down_nodelist) if leafind[i] ][k][ch] = 1.0
    for i,n in enumerate(down_nodelist[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)
        childlist[i] = [ nod.pi for nod in postnodes[i].children ]
    childlist[i+1] = [ nod.pi for nod in postnodes[i+1].children ]
    # Setting initial node likelihoods to one for calculations
    down_nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

    # -------- Up nodelist
    up_nodelist[:,:-1].fill(1)
    for i,n in enumerate(up_nodelist[1:]):
        n[nchar] = postnodes.index(prenodes[1:][i].parent)
        n[nchar+1] = postnodes.index(prenodes[1:][i])

    # ------- Partial likelihood vectors
    partial_nodelist = np.zeros([nnode,nchar,max([n.nchildren for n in tree])])

    return down_nodelist,up_nodelist,partial_nodelist,childlist,t
