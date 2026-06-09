
from linkage.global_model.point.itc_point import ITCPoint

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

    dh_dilution_idx = [5]
    titrating_species_mask = np.array([True,False,False],dtype=bool)

    # Identity extent matrix for two independent reactions
    extent_matrix = np.eye(2)

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 total_volume=total_volume,
                 injection_volume=injection_volume,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_idx=dh_dilution_idx,
                 titrating_species_mask=titrating_species_mask,
                 extent_matrix=extent_matrix)
    
    assert e.idx == 0
    assert e.expt_idx == 1
    assert e.obs_key == "test"
    assert e._micro_array is micro_array
    assert e._macro_array is macro_array
    assert e._del_macro_array is del_macro_array
    assert e._total_volume == total_volume
    assert e._injection_volume == injection_volume

    assert e._dh_sign is dh_sign
    assert e._dh_product_mask is dh_product_mask
    assert e._dh_dilution_idx is dh_dilution_idx
    assert e._titrating_species_mask is titrating_species_mask

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
    
    # heat of dilution for first species is 1
    dh_dilution_idx = [5]
    titrating_species_mask = np.array([True,False,False],dtype=bool)

    # Independent reactions => identity extent matrix
    extent_matrix = np.eye(2)

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
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_idx=dh_dilution_idx,
                 titrating_species_mask=titrating_species_mask,
                 extent_matrix=extent_matrix)
    
    calculated_value = e.calc_value(parameters=parameters, full_dh_array=parameters[dh_param_start_idx:dh_param_end_idx])

    meas_vol_dilution = (1 - injection_volume/total_volume)

    # Manually calculate what we expect this to do
    expected_value = 0
    for i in [1,2]:
        rxn = (micro_array[1,i] - micro_array[0,i]*meas_vol_dilution)
        rxn *= dh_sign[i-1]*parameters[i+2]*total_volume
        
        expected_value += rxn
    
    # heat of dilution is 1; volume; molar change of 0.1 (del_macro_array)
    # The first species (index 0) is being titrated (mask=[True,False,False])
    # del_macro_array has 0.1 for everything. 
    # dil_heats = parameters[5] = 1. molar_change = 0.1.
    expected_value += 1*injection_volume*0.1

    assert np.isclose(calculated_value,expected_value)


def test_ITCPoint_cascade():
    """
    Test the extent-recovery fix for sequential binding cascades.

    Scenario: 2-reaction chain
        rxn 1:  A + C  ->  AC1   (product: species 0 = AC1)
        rxn 2:  AC1 + C -> AC2   (product: species 1 = AC2)

    Stoichiometry:
        delta[AC1] = xi_1 - xi_2
        delta[AC2] = xi_2

    Choose concentrations so that delta[AC1] = 0 and delta[AC2] = x.
    That means xi_2 = x and xi_1 = xi_2 = x  (both reactions fired equally).

    With the OLD (broken) code, heat_1 = dH1 * delta[AC1] = 0, missing rxn 1.
    With the NEW code (extent matrix), xi_1 = x = xi_2, so both contribute.
    """
    idx = 1
    expt_idx = 0
    obs_key = "heat"

    # 2-point time series, 2 microspecies (AC1=0, AC2=1)
    # delta[AC1] = 0 (no net change), delta[AC2] = +x
    x = 5e-6
    micro_array = np.zeros((2, 2), dtype=float)
    micro_array[0, 0] = 10e-6   # AC1 before injection
    micro_array[1, 0] = 10e-6   # AC1 after  (net zero change)
    micro_array[0, 1] = 0.0     # AC2 before
    micro_array[1, 1] = x       # AC2 after  (+x change)

    macro_array = np.zeros((2, 1), dtype=float)
    del_macro_array = np.zeros((2, 1), dtype=float)

    total_volume = 200e-6
    injection_volume = 0.0      # zero injection so meas_vol_dilution = 1

    # Product masks: rxn1 -> AC1 (idx 0), rxn2 -> AC2 (idx 1)
    mask_ac1 = np.array([True, False], dtype=bool)
    mask_ac2 = np.array([False, True], dtype=bool)
    dh_product_mask = [mask_ac1, mask_ac2]

    dh_sign = [1.0, 1.0]
    dh_dilution_idx = []
    titrating_species_mask = np.array([False], dtype=bool)

    # Correct extent matrix for this 2-step chain:
    # N = [[1, -1],   (AC1 produced by rxn1, consumed by rxn2)
    #      [0,  1]]   (AC2 produced by rxn2 only)
    # pinv(N) = [[1, 1],
    #            [0, 1]]
    extent_matrix = np.array([[1.0, 1.0],
                               [0.0, 1.0]])

    dH1 = 1000.0   # cal/mol
    dH2 = 2000.0   # cal/mol
    full_dh_array = np.array([dH1, dH2])

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 total_volume=total_volume,
                 injection_volume=injection_volume,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_idx=dh_dilution_idx,
                 titrating_species_mask=titrating_species_mask,
                 extent_matrix=extent_matrix)

    calc = e.calc_value(parameters=np.array([]), full_dh_array=full_dh_array)

    # delta_c = [delta[AC1], delta[AC2]] = [0, x]
    # extents  = M @ [0, x] = [x, x]
    # heat = (dH1*1*x + dH2*1*x) * V = (dH1+dH2)*x*V
    expected = (dH1 + dH2) * x * total_volume
    assert np.isclose(calc, expected), f"Got {calc}, expected {expected}"

    # Sanity check: with identity matrix (old behaviour) the heat would be
    # WRONG — only rxn2 contributes since delta[AC1]=0.
    e_old = ITCPoint(idx=idx,
                     expt_idx=expt_idx,
                     obs_key=obs_key,
                     micro_array=micro_array,
                     macro_array=macro_array,
                     del_macro_array=del_macro_array,
                     total_volume=total_volume,
                     injection_volume=injection_volume,
                     dh_sign=dh_sign,
                     dh_product_mask=dh_product_mask,
                     dh_dilution_idx=dh_dilution_idx,
                     titrating_species_mask=titrating_species_mask,
                     extent_matrix=np.eye(2))   # old identity = wrong for cascade

    calc_old = e_old.calc_value(parameters=np.array([]), full_dh_array=full_dh_array)
    wrong_expected = dH2 * x * total_volume
    assert np.isclose(calc_old, wrong_expected), "Sanity check: old code should miss rxn1"
    assert not np.isclose(calc, calc_old), "New and old results must differ for cascade"


def test_ITCPoint_stat_factor():
    """
    Test statistical-factor (stoich_weight) handling for a 2-reaction cascade.

    Uses the E0A2_sf stoichiometry:
        AT = A + 2*AC1 + AC2   →  stoich_weight = [2, 1]

    Scenario (same cascade as test_ITCPoint_cascade but with weights):
        After injection: Δ[AC1] = 30 (per microstate), Δ[AC2] = 10.
        Weighted delta_c = [2*30, 10] = [60, 10].
        Extents = M @ [60, 10] = [70, 10].
        Heat = (dH_1 * 70 + dH_2 * 10) * V.
    """
    idx = 1
    expt_idx = 0
    obs_key = "heat"

    micro_array = np.zeros((2, 2), dtype=float)
    micro_array[0, 0] = 0.0    # AC1 before
    micro_array[1, 0] = 30e-6  # AC1 after  (Δ = 30e-6 per microstate)
    micro_array[0, 1] = 0.0    # AC2 before
    micro_array[1, 1] = 10e-6  # AC2 after  (Δ = 10e-6)

    macro_array = np.zeros((2, 1), dtype=float)
    del_macro_array = np.zeros((2, 1), dtype=float)

    total_volume = 200e-6
    injection_volume = 0.0  # zero so dilution factor = 1

    mask_ac1 = np.array([True, False], dtype=bool)
    mask_ac2 = np.array([False, True], dtype=bool)
    dh_product_mask = [mask_ac1, mask_ac2]

    dh_sign = [1.0, 1.0]
    dh_dilution_idx = []
    titrating_species_mask = np.array([False], dtype=bool)

    # Cascade extent matrix  M = pinv([[1,-1],[0,1]]) = [[1,1],[0,1]]
    extent_matrix = np.array([[1.0, 1.0],
                               [0.0, 1.0]])

    dH1 = 1000.0
    dH2 = 2000.0
    full_dh_array = np.array([dH1, dH2])

    stoich_weight = [2.0, 1.0]  # statistical factor from AT = A + 2*AC1 + AC2

    e = ITCPoint(idx=idx,
                 expt_idx=expt_idx,
                 obs_key=obs_key,
                 micro_array=micro_array,
                 macro_array=macro_array,
                 del_macro_array=del_macro_array,
                 total_volume=total_volume,
                 injection_volume=injection_volume,
                 dh_sign=dh_sign,
                 dh_product_mask=dh_product_mask,
                 dh_dilution_idx=dh_dilution_idx,
                 titrating_species_mask=titrating_species_mask,
                 extent_matrix=extent_matrix,
                 stoich_weight=stoich_weight)

    calc = e.calc_value(parameters=np.array([]), full_dh_array=full_dh_array)

    # weighted delta_c = [2*30e-6, 10e-6]
    # extents = M @ [60e-6, 10e-6] = [70e-6, 10e-6]
    # heat = (dH1*70e-6 + dH2*10e-6) * V
    expected = (dH1 * 70e-6 + dH2 * 10e-6) * total_volume
    assert np.isclose(calc, expected), f"Got {calc}, expected {expected}"

    # Without stoich_weight (default = [1,1]) the heat would be wrong
    e_no_sf = ITCPoint(idx=idx,
                       expt_idx=expt_idx,
                       obs_key=obs_key,
                       micro_array=micro_array,
                       macro_array=macro_array,
                       del_macro_array=del_macro_array,
                       total_volume=total_volume,
                       injection_volume=injection_volume,
                       dh_sign=dh_sign,
                       dh_product_mask=dh_product_mask,
                       dh_dilution_idx=dh_dilution_idx,
                       titrating_species_mask=titrating_species_mask,
                       extent_matrix=extent_matrix)

    calc_no_sf = e_no_sf.calc_value(parameters=np.array([]), full_dh_array=full_dh_array)
    wrong_expected = (dH1 * 40e-6 + dH2 * 10e-6) * total_volume  # unweighted
    assert np.isclose(calc_no_sf, wrong_expected), f"Sanity: got {calc_no_sf}, expected {wrong_expected}"
    assert not np.isclose(calc, calc_no_sf), "Stat-factor weighting must change the result"
