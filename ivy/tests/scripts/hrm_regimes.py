import ivy
import numpy as np
import scipy
import math

import itertools

regimes = {"IRR_01_FAST", "IRR_10_FAST", "IRR_01_SLOW", "IRR_10_SLOW",
           "ASYM_01_FAST", "ASYM_10_FAST", "ASYM_01_SLOW", "ASYM_10_SLOW",
           "SYM_FAST", "SYM_SLOW", "NO_CHANGE"}
regime_combinations = list(itertools.combinations(regimes,2))

invalid_combinations = [("IRR_01_FAST", "IRR_01_SLOW"),
                        ("IRR_10_FAST", "IRR_10_SLOW"),
                        ("ASYM_01_FAST","ASYM_01_SLOW"),
                        ("ASYM_10_FAST","ASYM_10_SLOW")]

regime_combinations = [ i for i in regime_combinations if not i in invalid_combinations]


regimetypes = ('ASYM_01_SLOW', 'SYM_SLOW')
