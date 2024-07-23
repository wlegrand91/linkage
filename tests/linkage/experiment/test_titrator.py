import pytest
import numpy as np
import pandas as pd

from linkage.experiment.titrator import titrator

import copy

def test_titrator():
    
    base_kwargs = {"cell_contents":{"X":0,"Y":10},
                   "syringe_contents":{"X":10,"Y":10},
                   "cell_volume":100,
                   "injection_array":np.ones(10,dtype=float),
                   "constant_volume":False}
    
    kwargs = copy.deepcopy(base_kwargs)
    df = titrator(**kwargs)
    assert issubclass(type(df),pd.DataFrame)


