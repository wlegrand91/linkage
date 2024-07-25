import pytest

from linkage.models.ca_edta import CaEDTA

import numpy as np

def test_CaEDTA_integrated():

    def _check_with_log(bm,param_array,macro_array,expected_values):

        concs = bm.get_concs(param_array=param_array,
                             macro_array=macro_array)

        log_round_concs = np.round(np.log(concs),3)
        assert np.allclose(log_round_concs,expected_values)


    bm = CaEDTA()

    assert np.array_equal(bm.macro_species,np.array(["CT","ET"]))
    assert np.array_equal(bm.micro_species,np.array(["C","E","EC"]))

    print("basic check")
    _check_with_log(bm=bm,
                    param_array=np.array([1e7]),
                    macro_array=np.array([1e-7,1e-10]),
                    expected_values=np.array([-16.119, -23.719, -23.719]))


    print("another arbitrary conc check")
    _check_with_log(bm=bm,
                    param_array=np.array([1e7]),
                    macro_array=np.array([1e-3,1e-3]),
                    expected_values=np.array([-11.518, -11.518,  -6.918]))

    print("zero checks")
    concs = bm.get_concs(param_array=np.array([1e7]),
                         macro_array=np.array([0,1e-10]))
    assert np.allclose(concs,np.array([0,1e-10,0]))

    concs = bm.get_concs(param_array=np.array([1e7]),
                         macro_array=np.array([1e-7,0]))
    assert np.allclose(concs,np.array([1e-7,0,0]))

    concs = bm.get_concs(param_array=np.array([1e7]),
                         macro_array=np.array([0,0]))
    assert np.allclose(concs,np.array([0,0,0]))

    print("graceful na checks")

    with pytest.warns():
        concs = bm.get_concs(param_array=np.array([np.nan]),
                             macro_array=np.array([1e-7,1e-10]))
        assert np.array_equal(concs,np.nan*np.ones(3,dtype=float),equal_nan=True)

    with pytest.warns():
        concs = bm.get_concs(param_array=np.array([1e7]),
                             macro_array=np.array([np.nan,1e-10]))
        assert np.array_equal(concs,np.nan*np.ones(3,dtype=float),equal_nan=True)

    with pytest.warns():
        concs = bm.get_concs(param_array=np.array([np.nan]),
                            macro_array=np.array([np.nan,np.nan]))
        assert np.array_equal(concs,np.nan*np.ones(3,dtype=float),equal_nan=True)