
import pytest
import numpy as np
from linkage.symbolic.model import SymbolicBindingModel
from linkage.symbolic.generic_binding_model import GenericBindingModel

def test_SymbolicBindingModel_init():
    # Simple model: A + B <-> AB
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
        BT = B + AB
    """
    bm = SymbolicBindingModel(model_spec)
    assert "K1" in bm.equilibrium_constants
    assert "AB" in bm.physical_poly._micro_species
    assert "AT" in bm.physical_poly._macro_species

    # Check regression params
    assert "K1" in bm.regression_params
    
def test_SymbolicBindingModel_solve():
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
        BT = B + AB
    """
    bm = SymbolicBindingModel(model_spec)
    
    # Solve for K1=1e6 (LogK = 13.8), AT=1e-6, BT=1e-6
    # Param dict expects LogK if not dH? No, symbolic model expects raw values in reg_dict?
    # SymbolicBindingModel.solve_concentrations takes reg_param_values.
    # In GenericBindingModel wrapper we do exp().
    # Let's check SymbolicBindingModel behavior directly.
    # It passes values to `get_physical_params`.
    # If standard map, K1_fit -> K1_phys is exp().
    # Wait, SymbolicBindingModel usually sets up LogK mapping by default.
    # Let's verify defaults.
    
    # SymbolicBindingModel via Generic maps K1 and dH_1
    reg_params = {"K1": np.log(1e6), "dH_1": 0.0}
    macro_concs = {"AT": 1e-5, "BT": 1e-5}
    
    res = bm.solve_concentrations(reg_params, macro_concs)
    
    # Strong binding, 1:1. AT=BT=10uM. Kd = 1uM.
    # Should result in significant AB.
    # A + B <-> AB. K=1e6. AB = 1e6 * A * B.
    # T = A + AB.
    
    assert res["A"] < 1e-5
    assert res["AB"] > 0
    assert np.isclose(res["A"] + res["AB"], 1e-5)

def test_GenericBindingModel_init():
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
    """
    gbm = GenericBindingModel(model_spec)
    # GenericBindingModel automatically adds dH parameters
    assert "K1" in gbm.param_names
    assert "dH_1" in gbm.param_names
    assert np.array_equal(gbm.macro_species, ["AT"])

def test_GenericBindingModel_get_concs():
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
        BT = B + AB
    """
    gbm = GenericBindingModel(model_spec)
    
    # Pass params as ARRAY (LogK, dH)
    params = np.array([np.log(1e6), 0.0]) # K1 = 1e6, dH = 0
    macros = np.array([1e-5, 1e-5])  # AT, BT
    
    concs = gbm.get_concs(params, macros)
    # Returns [A, B, AB] array
    
    micro_names = gbm.micro_species
    c_dict = dict(zip(micro_names, concs))
    
    assert c_dict["AB"] > 0
    assert np.isclose(c_dict["A"] + c_dict["AB"], 1e-5)

