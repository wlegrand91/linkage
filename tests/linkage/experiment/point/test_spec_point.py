
from linkage.experiment.point.spec_point import SpecPoint

import numpy as np

def test_SpecPoint():

    # Test properties and basic setter/getter
    idx = 0
    expt_idx = 1
    obs_key = "test"
    micro_array = np.ones((50,10),dtype=float)
    macro_array = np.ones((50,3),dtype=float)
    del_macro_array = np.zeros((50,3),dtype=float)
    
    obs_mask = np.zeros(10,dtype=bool)
    obs_mask[1] = True
    denom = 2

    e = SpecPoint(idx=idx,
                  expt_idx=expt_idx,
                  obs_key=obs_key,
                  micro_array=micro_array,
                  macro_array=macro_array,
                  del_macro_array=del_macro_array,
                  obs_mask=obs_mask,
                  denom=denom)
  
    assert e.idx == 0
    assert e.expt_idx == 1
    assert e.obs_key == "test"

    assert e._micro_array is micro_array
    assert e._macro_array is macro_array
    assert e._del_macro_array is del_macro_array

    assert e._obs_mask is obs_mask
    assert e._denom == denom

def test_SpecPoint_calc_value():


    expt_idx = 1
    obs_key = "test"

    micro_array = np.ones((50,10),dtype=float)
    macro_array = np.ones((50,3),dtype=float)
    del_macro_array = np.zeros((50,3),dtype=float)
    
    # Point 0 and 1 in micro array
    micro_array[0,1] = 50
    micro_array[1,1] = 100

    # Point 0 and 1 in macro array
    macro_array[0,2] = 50
    macro_array[1,2] = 150

    # look at species 1 in micro array
    obs_mask = np.zeros(10,dtype=bool)
    obs_mask[1] = True

    # Look at species 2 in macro array
    denom = 2

    e = SpecPoint(idx=0,
                  expt_idx=expt_idx,
                  obs_key=obs_key,
                  micro_array=micro_array,
                  macro_array=macro_array,
                  del_macro_array=del_macro_array,
                  obs_mask=obs_mask,
                  denom=denom)
    
    assert e.calc_value() == 50/50
  

    e = SpecPoint(idx=1,
                  expt_idx=expt_idx,
                  obs_key=obs_key,
                  micro_array=micro_array,
                  macro_array=macro_array,
                  del_macro_array=del_macro_array,
                  obs_mask=obs_mask,
                  denom=denom)
    
    assert e.calc_value() == 100/150
  

    # Now look at points 1 and 2
    micro_array[0,2] = 50
    micro_array[1,2] = 50

    obs_mask = np.zeros(10,dtype=bool)
    obs_mask[1] = True
    obs_mask[2] = True

    e = SpecPoint(idx=0,
                  expt_idx=expt_idx,
                  obs_key=obs_key,
                  micro_array=micro_array,
                  macro_array=macro_array,
                  del_macro_array=del_macro_array,
                  obs_mask=obs_mask,
                  denom=denom)
    
    assert e.calc_value() == (50 + 50)/50
  

    e = SpecPoint(idx=1,
                  expt_idx=expt_idx,
                  obs_key=obs_key,
                  micro_array=micro_array,
                  macro_array=macro_array,
                  del_macro_array=del_macro_array,
                  obs_mask=obs_mask,
                  denom=denom)
    
    assert e.calc_value() == (100 + 50)/150