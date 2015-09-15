"""
Functions that interface with r using rpy2
"""
import rpy2
from rpy2.robjects.packages import importr
import numpy as np


def phylorate(eventfile, treefile, spex):
    """
    Use BAMMtools to get rate data for characters or speciation/extinction
    rates along the branches of tree

    (http://bamm-project.org/introduction.html)

    Args:
        eventfile (str): Path to event data output from BAMM
        treefile (str): Path to tree
        spex (str): "s", "e", or "netdiv". Whether to get speciation,
          extinction, or net diversification rates.
    Returns:
        Tuple of rates and node indices associated with the rates
    """

    ape = importr('ape')
    bamm = importr('BAMMtools')

    tree = ape.read_tree(treefile)
    edata = bamm.getEventData(tree, eventdata=eventfile, burnin=0.2)
    dtrates = bamm.dtRates(edata, 0.01, tmat=True).rx2('dtrates')
    nodeidx = np.array(dtrates.rx2('tmat').rx(True, 1), dtype=int)
    rates = np.array(dtrates.rx2('rates'))

    if spex == "s":
        out = rates[0]
    elif spex == "e":
        out = rates[1]
    elif spex == "netdiv":
        out = rates[0]-rates[1]

    return (out, nodeidx)
