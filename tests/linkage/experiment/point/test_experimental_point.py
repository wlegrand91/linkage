
from linkage.experiment.point.experimental_point import ExperimentalPoint

import numpy as np

def test_ExperimentalPoint():

    # Test properties and basic setter/getter
    idx = 0
    expt_idx = 1
    obs_key = "test"
    micro_array = np.ones(10)
    macro_array = np.ones(3)
    e = ExperimentalPoint(idx=idx,
                          expt_idx=expt_idx,
                          obs_key=obs_key,
                          micro_array=micro_array,
                          macro_array=macro_array)
    
    assert e.idx == 0
    assert e.expt_idx == 1
    assert e.obs_key == "test"

    assert e._micro_array is micro_array
    assert e._macro_array is macro_array
    
