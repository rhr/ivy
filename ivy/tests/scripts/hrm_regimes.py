import ivy
import numpy as np
import scipy
import math

import itertools

regimes = {"IRR_01_FAST", "IRR_10_FAST", "IRR_01_SLOW", "IRR_10_SLOW",
           "ASYM_01_FAST", "ASYM_10_FAST", "ASYM_01_SLOW", "ASYM_10_SLOW",
           "SYM_FAST", "SYM_SLOW"}
regime_combinations = list(itertools.combinations(regimes,2))



sts = [ (n.parent.sim_char["sim_state"], n.sim_char["sim_state"]) for n in simtree.descendants()]

two_three_trans = [s for s in sts if s[0] == 2 and s[1]==3]


float(len(two_three_trans))/len([s for s in sts if s[0] == 2])


three_two_trans = [s for s in sts if s[0] == 3 and s[1]==2]
float(len(three_two_trans))/len([s for s in sts if s[0] == 3])
