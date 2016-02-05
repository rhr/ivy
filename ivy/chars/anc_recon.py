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

def anc_recon_purepy(tree, chars, Q, p=None, pi="Fitzjohn", ars=None):
    """
    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips

    """
    nchar = Q.shape[0]
    if ars is None:
        # Creating arrays to be used later
        ars = create_ancrecon_ars(tree, chars)
    # Calculating the likelihoods for each node in post-order sequence
    if p is None: # Instantiating empty array
        p = np.empty([len(ars["t"]), Q.shape[0],
                     Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, ars["t"], p) # This changes p in place
    np.copyto(ars["down_nl_w"], ars["down_nl_r"]) # Copy original values if they have been changed
    ars["child_inds"].fill(0)
    root_equil = ivy.chars.mk.qsd(Q)
    # ------------------- Performing the down-pass -----------------------------
    for intnode in map(int, sorted(set(ars["down_nl_w"][:-1,nchar]))):
        nextli = ars["down_nl_w"][intnode]
        for chi, child in enumerate(ars["childlist"][intnode]):
            li = ars["down_nl_w"][child]
            p_li = ars["partial_nl"][intnode][chi]
            for ch in ars["charlist"]:
                p_li[ch] = sum([ p[child][ch,st] for st in ars["charlist"] ]
                               * li[:nchar])
                nextli[ch] *= p_li[ch]

    # "downpass_likelihood" contains the downpass likelihood vectors for each node in postorder sequence
    # Now that the downpass has been performed, we must perform the up-pass
    # ------------------- Performing the up-pass -------------------------------
    # The up-pass likelihood at each node is equivalent to information coming
    # up from the root * information coming down from the tips

    # Each node has the following:
    # Uppass_likelihood (set to the equilibrium frequency for the root)
    # Marginal_likelihood (product of uppass_likelihood and downpass likelihood)
    # Partial likelihood for each child node
    # The final two columns of up_nl point to the
    # postorder index numbers of the parent and self node, respectively

    # child_masks contains an array of the children to use for calculating
    # partial likelihood of the next child of that node. All parents
    # start out with excluding the first child that appears (for calculating
    # marginal likelihood of that child)


    # The parent's partial likelihood without current node
    # partial_parent_likelihoods = np.zeros([ars["up_nl"].shape[0],nchar])
    root_posti = ars["up_nl"].shape[0] - 1
    for i,l in enumerate(ars["up_nl"]):
        # Uppass information for node
        if i == 0:
            # Set root node uppass to be equivalent to the root equilibrium

            # Set the marginal to be equivalent to the root equilibrium times
            # the root downpass
            l[:nchar] = root_equil
            ars["marginal_nl"][i][:nchar] = (l[:nchar] *
                                               ars["down_nl_w"][-1][:nchar])
        else:
            spi = int(l[nchar+1]) #self's postorder index
            ppi = int(l[nchar]) # the parent's postorder index
            if ppi == root_posti:
                # If parent is the root, the parent's partial likelihood is
                # equivalent to the partial downpass (downpass likelihoods of
                # the node's siblings, indexed using 'child_masks') and
                # the equilibrium frequency of the Q matrix.
                ars["partial_parent_nl"][spi] = (ars["partial_nl"][ppi].take(range(ars["child_inds"][ppi])+range(ars["child_inds"][ppi]+1,ars["partial_nl"][ppi].shape[0]),0) *
                                                               root_equil)
            else:
                # If parent is not the root, the parent's partial likelihood is
                # the partial downpass * the partial uppass, which is calculated
                # as the parent of the parent's partial likelihood times
                # the transition probability
                np.dot(p[ppi].T, ars["partial_parent_nl"][ppi], out=ars["temp_dotprod"])
                ars["partial_parent_nl"][spi] = (ars["partial_nl"][ppi].take(range(ars["child_inds"][ppi])+range(ars["child_inds"][ppi]+1,ars["partial_nl"][ppi].shape[0]),0) *
                                                ars["temp_dotprod"])
            # The up-pass likelihood is equivalent to the parent's partial
            # likelihood times the transition probability
            np.dot(p[spi].T, ars["partial_parent_nl"][spi], out = l[:nchar])
            # Roll child masks so that next likelihood calculated for this
            # parent uses siblings of next node
            ars["child_inds"][ppi]  += 1

            # Marginal = Uppass * downpass
            ars["marginal_nl"][i][:nchar] = l[:nchar] * ars["down_nl_w"][l[nchar+1]][:nchar]

    return ars["marginal_nl"]

def anc_recon(tree, chars, Q, p=None, pi="Fitzjohn", ars=None):
    """
    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Q (np.array): Instantaneous rate matrix
        p (np.array): 3-D array of dimensions branch_number * nchar * nchar.
            Optional. Pre-allocated space for efficient calculations
        pi (str or np.array): Option to weight the root node by given values.
           Either a string containing the method or an array
           of weights. Weights should be given in order.

           Accepted methods of weighting root:

           Equal: flat prior
           Equilibrium: Prior equal to stationary distribution
             of Q matrix
           Fitzjohn: Root states weighted by how well they
             explain the data at the tips.
        ars (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays. Can be
          created with create_ancrecon_ars function.
    Returns:
        np.array: Array of nodes in preorder sequence containing marginal
          likelihoods.
    """
    nchar = Q.shape[0]
    if ars is None:
        # Creating arrays to be used later
        ars = create_ancrecon_ars(tree, chars)
    # Calculating the likelihoods for each node in post-order sequence
    if p is None: # Instantiating empty array
        p = np.empty([len(ars["t"]), Q.shape[0],
                     Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, ars["t"], p) # This changes p in place
    np.copyto(ars["down_nl_w"], ars["down_nl_r"]) # Copy original values if they have been changed
    ars["child_inds"].fill(0)
    root_equil = ivy.chars.mk.qsd(Q)

    cyexpokit.cy_anc_recon(p, ars["down_nl_w"], ars["charlist"], ars["childlist"],
                        ars["up_nl"], ars["marginal_nl"], ars["partial_parent_nl"],
                        ars["partial_nl"], ars["child_inds"], root_equil,ars["temp_dotprod"])



    return ars["marginal_nl"]


def create_ancrecon_ars(tree, chars):
    """
    Create nodelists. For use in anc_recon function

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
    down_nl = np.zeros([nnode, nchar+1])
    up_nl = np.zeros([nnode, nchar+2])
    childlist = np.zeros(nnode, dtype=object)
    leafind = [ n.isleaf for n in tree.postiter()]

    # --------- Down nl, postorder sequence
    for k,ch in enumerate(postChars):
        [ n for i,n in enumerate(down_nl) if leafind[i] ][k][ch] = 1.0
    for i,n in enumerate(down_nl[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)
        childlist[i] = [ nod.pi for nod in postnodes[i].children ]
    childlist[i+1] = [ nod.pi for nod in postnodes[i+1].children ]
    # Setting initial node likelihoods to one for calculations
    down_nl[[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

    # -------- Up nl, preorder sequence
    up_nl[:,:-1].fill(1)
    for i,n in enumerate(up_nl[1:]):
        n[nchar] = postnodes.index(prenodes[1:][i].parent)
        n[nchar+1] = postnodes.index(prenodes[1:][i])
    # -------- Marginal nl, also preorder sequence
    marginal_nl = up_nl.copy()

    # ------- Partial likelihood vectors
    partial_nl = np.zeros([nnode,nchar,max([n.nchildren for n in tree])])

    # ------- Parent partial likelihood vectors
    partial_parent_nl = np.zeros([up_nl.shape[0],nchar])

    # ------- Child masking arrays
    child_inds = np.zeros(partial_nl.shape[0], dtype=int)

    # ------- Character list
    charlist = range(len(set(chars)))

    # ------- Empty array to store root priors
    root_priors = np.empty([nchar], dtype=np.double)

    # ------- Temporary array to store dot products
    temp_dotprod = np.empty([nchar], dtype=np.double)


    names = ["down_nl_r","down_nl_w","up_nl","marginal_nl",
            "partial_nl","partial_parent_nl","child_inds",
            "childlist","t","charlist","root_priors","temp_dotprod"]
    ar_list = [down_nl,down_nl.copy(),up_nl,marginal_nl,partial_nl,partial_parent_nl,child_inds,
           childlist,t,charlist,root_priors,temp_dotprod]

    ar_dict = {name:ar for name,ar in zip(names, ar_list)}

    return ar_dict
