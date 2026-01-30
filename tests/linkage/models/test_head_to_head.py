import pytest
from linkage.models.head_to_head import HeadToHead
import numpy as np

def test_HeadToHead_properties():
    bm = HeadToHead()
    
    # Check simple properties
    assert np.array_equal(bm.param_names, ["K1", "K2"])
    assert np.array_equal(bm.macro_species, ["mt", "st", "at"])
    assert np.array_equal(bm.micro_species, ["m", "ma", "s", "sa", "a"])
    
    # Check parsed properties
    assert len(bm.species) == 3
    assert len(bm.equilibria) == 2

def test_HeadToHead_get_concs():
    bm = HeadToHead()
    
    # Case 1: All zero concentrations
    # mt=0, st=0, at=0 -> all 0
    concs = bm.get_concs(param_array=np.array([1.0, 1.0]), 
                         macro_array=np.array([0, 0, 0]))
    assert np.allclose(concs, np.zeros(5))

    # Case 2: No ligand (at=0)
    # mt=1, st=1, at=0 -> m=1, ma=0, s=1, sa=0, a=0
    concs = bm.get_concs(param_array=np.array([1.0, 1.0]), 
                         macro_array=np.array([1.0, 1.0, 0]))
    # species order: m, ma, s, sa, a
    expected = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(concs, expected)

    # Case 3: No receptor (mt=0, st=0)
    # mt=0, st=0, at=1 -> m=0, ma=0, s=0, sa=0, a=1
    concs = bm.get_concs(param_array=np.array([1.0, 1.0]), 
                         macro_array=np.array([0, 0, 1.0]))
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert np.allclose(concs, expected)

    # Case 4: General calculation (verify non-nan and mass conservation)
    mt, st, at = 1.0, 1.0, 1.0
    concs = bm.get_concs(param_array=np.array([1.0, 1.0]), 
                         macro_array=np.array([mt, st, at]))
    
    # Check shape
    assert concs.shape == (5,)
    # Check finite
    assert np.all(np.isfinite(concs))
    # Check non-negative
    assert np.all(concs >= 0)
    
    # Check mass conservation
    # m_total = m + ma
    # s_total = s + sa
    # a_total = a + ma + sa
    # indices: m=0, ma=1, s=2, sa=3, a=4
    m, ma, s, sa, a = concs
    
    assert np.isclose(m + ma, mt)
    assert np.isclose(s + sa, st)
    assert np.isclose(a + ma + sa, at)
