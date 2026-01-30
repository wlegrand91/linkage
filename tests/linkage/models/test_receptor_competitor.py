import pytest
from linkage.models.receptor_competitor import ReceptorCompetitor
import numpy as np

def test_ReceptorCompetitor_properties():
    bm = ReceptorCompetitor()
    
    # Check simple properties
    assert np.array_equal(bm.param_names, ["K", "KA", "KI"])
    assert np.array_equal(bm.macro_species, ["mt", "st", "at", "rt"])
    assert np.array_equal(bm.micro_species, ["m", "s", "a", "r", "ma", "rma", "rs"])
    
    # Check parsed properties
    assert len(bm.species) == 4
    assert len(bm.equilibria) == 3

def test_ReceptorCompetitor_get_concs():
    bm = ReceptorCompetitor()
    
    # Case 1: All zero concentrations
    concs = bm.get_concs(param_array=np.array([1.0, 1.0, 1.0]), 
                         macro_array=np.array([0, 0, 0, 0]))
    assert np.allclose(concs, np.zeros(7))
    
    # Case 2: Verification of warning when rt is large
    # rt > 0.01 * min(mt, st, at)
    with pytest.warns(UserWarning, match="we assumed rt was small"):
        bm.get_concs(param_array=np.array([1.0, 1.0, 1.0]),
                     macro_array=np.array([1.0, 1.0, 1.0, 1.0]))

    # Case 3: Calculation checks
    # mt=1, st=2, at=1, rt=0.001 (small enough to avoid warning)
    concs = bm.get_concs(param_array=np.array([1.0, 1.0, 1.0]), 
                         macro_array=np.array([1.0, 2.0, 1.0, 0.001]))
    
    assert concs.shape == (7,)
    assert np.all(np.isfinite(concs))
    assert np.all(concs >= 0)
    
    # Mass conservation checks
    # species: m, s, a, r, ma, rma, rs
    m, s, a, r, ma, rma, rs = concs
    mt, st, at, rt = 1.0, 2.0, 1.0, 0.001
    
    # Note: The model derivation assumes rt is small and seems to calculate m/s/a 
    # without subtracting the r-bound fractions first, then calculates r derivatives.
    # Therefore, exact mass conservation for m, s, a might be slightly off if rt is non-zero,
    # as evidenced by:
    # m = mt - ma
    # at = a + ma
    # but actual at = a + ma + rma
    # So a + ma should equal at - rma. The code does a = at - ma.
    # This implies the code assumes rma is negligible for mass balance of A.
    
    assert np.isclose(m + ma + rma, mt, atol=1e-2) # Loose tolerance due to approximation
    assert np.isclose(s + rs, st, atol=1e-2) # Code sets s = st, so rs is neglected
    assert np.isclose(a + ma + rma, at, atol=1e-2)
    assert np.isclose(r + rma + rs, rt) # The R conservation should be exact
