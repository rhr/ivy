import ivy
import numpy as np
import scipy
import math

import itertools

regimes = {"IRR_01_FAST", "IRR_10_FAST", "IRR_01_SLOW", "IRR_10_SLOW",
           "ASYM_01_FAST", "ASYM_10_FAST", "ASYM_01_SLOW", "ASYM_10_SLOW",
           "SYM_FAST", "SYM_SLOW"}
regime_combinations = list(itertools.combinations(regimes,2))
