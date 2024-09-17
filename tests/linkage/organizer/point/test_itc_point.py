
from linkage.organizer.point.itc_point import ITCPoint

import numpy as np

def test_ITCPoint():

    # First, test __init__ and setter/getters

    idx = 0
    expt_idx = 1
    obs_key = "test"
    micro_array = np.ones((50,10),dtype=float)
    macro_array = np.ones((50,3),dtype=float)
    del_macro_array = 0.1*np.ones((50,3),dtype=float)
    total_volume = 200e-6
    injection_volume = 1e-6

    dh_param_start_idx = 1
    dh_param_end_idx = 4
    dh_sign = np.array([1,-1])

    m0 = np.zeros(10,dtype=bool)
    m1 = np.zeros(10,dtype=bool)
    m0[0] = True
    m1[1] = True
    dh_product_mask = [m0,m1]

    dh_dilution_mask = np.array([True,False,False],dtype=bool)

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 total_volume=total_volume,
                 injection_volume=injection_volume,
                 dh_param_start_idx=dh_param_start_idx,
                 dh_param_end_idx=dh_param_end_idx,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_mask=dh_dilution_mask)
    
    assert e.idx == 0
    assert e.expt_idx == 1
    assert e.obs_key == "test"
    assert e._micro_array is micro_array
    assert e._macro_array is macro_array
    assert e._del_macro_array is del_macro_array
    assert e._total_volume == total_volume
    assert e._injection_volume == injection_volume

    assert e._dh_param_start_idx == dh_param_start_idx
    assert e._dh_param_end_idx == dh_param_end_idx
    assert e._dh_sign is dh_sign
    assert e._dh_product_mask is dh_product_mask
    assert e._dh_dilution_mask is dh_dilution_mask

    expected_vol_dilution = (1 - injection_volume/total_volume)
    assert np.isclose(e._meas_vol_dilution,expected_vol_dilution)

    # Now test calculator. Silly test given everyone is zero, but make sure 
    # it runs. 
    parameters = np.zeros(5)

    calc_value = e.calc_value(parameters=parameters)
    assert calc_value == 0

def test_ITCPoints_calc_value():

    # first stop (for ITC experiment)
    idx = 1

    # ignored in this test
    expt_idx = 0
    obs_key = "ignored"
    macro_array = np.zeros((2,3),dtype=float)

    # Make a micro array that has two rows and five microspecies
    micro_array = np.zeros((2,5),dtype=float)

    # del macro array
    del_macro_array = np.zeros((2,3),dtype=float) + 0.1

    # Species 1 changes from 1.0 -> 1.2 over this step
    micro_array[0,1] = 1.0
    micro_array[1,1] = 1.2

    # Species 2 changes from 1.0 -> 0.8 over this step
    micro_array[0,2] = 1.0
    micro_array[1,2] = 0.8

    total_volume = 280e-6
    injection_volume = 2e-6

    # Parameters in array are 3 and 4
    dh_param_start_idx = 3
    dh_param_end_idx = 8

    # negative and positive sign for dH
    dh_sign = [-1,1]

    # Look at species 1 and 2
    dh_product_mask = [np.array([False,True,False,False,False],dtype=bool),
                       np.array([False,False,True,False,False],dtype=bool)]
    
    dh_dilution_mask = np.array([False,False,True],dtype=bool)


    # dH first is 1, dH second is 10, heat of dilution for first species is 1
    parameters = np.zeros(10)
    parameters[3] = 1
    parameters[4] = 10
    parameters[5] = 1

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 total_volume=total_volume,
                 injection_volume=injection_volume,
                 dh_param_start_idx=dh_param_start_idx,
                 dh_param_end_idx=dh_param_end_idx,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_mask=dh_dilution_mask)
    
    calculated_value = e.calc_value(parameters=parameters)

    meas_vol_dilution = (1 - injection_volume/total_volume)

    # Manually calculate what we expect this to do
    expected_value = 0
    for i in [1,2]:
        rxn = (micro_array[1,i] - micro_array[0,i]*meas_vol_dilution)
        rxn *= dh_sign[i-1]*parameters[i+2]*total_volume
        
        expected_value += rxn
    
    # heat of dilution is 1; volume; molar change of 0.1 (del_macro_array)
    expected_value += 1*injection_volume*0.1

    assert np.isclose(calculated_value,expected_value)


    




