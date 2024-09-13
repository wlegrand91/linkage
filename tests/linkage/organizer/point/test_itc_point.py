
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
    meas_vol_dilution = 1.0
    dh_param_start_idx = 1
    dh_param_end_idx = 4
    dh_sign = np.array([1,-1])
    m0 = np.zeros(10,dtype=bool)
    m1 = np.zeros(10,dtype=bool)
    m0[0] = True
    m1[1] = True
    dh_product_mask = [m0,m1]

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 meas_vol_dilution=meas_vol_dilution,
                 dh_param_start_idx=dh_param_start_idx,
                 dh_param_end_idx=dh_param_end_idx,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask)
    
    assert e.idx == 0
    assert e.expt_idx == 1
    assert e.obs_key == "test"
    assert e._micro_array is micro_array
    assert e._macro_array is macro_array
    
    assert e._meas_vol_dilution == meas_vol_dilution
    assert e._dh_param_start_idx == dh_param_start_idx
    assert e._dh_param_end_idx == dh_param_end_idx
    assert e._dh_sign is dh_sign
    assert e._dh_product_mask is dh_product_mask

    # Now test calculator. Silly test given everyone is zero, but make sure 
    # it runs. 
    parameters = np.zeros(5)
    parameters[1] = 1
    parameters[2] = 4

    calc_value = e.calc_value(parameters=parameters)
    assert calc_value == 0

def test_ITCPoints_get_value():

    # first stop (for ITC experiment)
    idx = 1

    # ignored in this test
    expt_idx = 0
    obs_key = "ignored"
    macro_array = np.zeros((2,3),dtype=float)

    # Make a micro array that has two rows and five microspecies
    micro_array = np.zeros((2,5),dtype=float)

    # del macro array
    del_macro_array = np.zeros((2,3),dtype=float) - 0.1

    # Species 1 changes from 1.0 -> 1.2 over this step
    micro_array[0,1] = 1.0
    micro_array[1,1] = 1.2

    # Species 2 changes from 1.0 -> 0.8 over this step
    micro_array[0,2] = 1.0
    micro_array[1,2] = 0.8

    # dilution
    meas_vol_dilution = (1 - 2/280)

    # Parameters in array are 3 and 4
    dh_param_start_idx = 3
    dh_param_end_idx = 8

    # negative and positive sign for dH
    dh_sign = [-1,1]

    # Look at species 1 and 2
    dh_product_mask = [np.array([False,True,False,False,False],dtype=bool),
                       np.array([False,False,True,False,False],dtype=bool)]
    

    # dH first is 1, dH second is 10
    parameters = np.zeros(10)
    parameters[3] = 1
    parameters[4] = 10

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 meas_vol_dilution=meas_vol_dilution,
                 dh_param_start_idx=dh_param_start_idx,
                 dh_param_end_idx=dh_param_end_idx,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask)
    
    calculated_value = e.calc_value(parameters=parameters)

    # Manually calculate what we expect this to do
    rxn0 = (micro_array[1,1] - micro_array[0,1]*meas_vol_dilution)*dh_sign[0]*parameters[3]
    rxn1 = (micro_array[1,2] - micro_array[0,2]*meas_vol_dilution)*dh_sign[1]*parameters[4]
    expected_value = rxn0 + rxn1

    assert np.isclose(calculated_value,expected_value)


    




