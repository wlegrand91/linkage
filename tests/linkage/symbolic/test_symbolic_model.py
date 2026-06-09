
import pytest
import numpy as np
from linkage.symbolic.model import SymbolicBindingModel
from linkage.symbolic.binding_model import BindingModel

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
    # In BindingModel wrapper we do exp().
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

def test_BindingModel_init():
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
    """
    gbm = BindingModel(model_spec)
    # BindingModel automatically adds dH parameters
    assert "K1" in gbm.param_names
    assert "dH_1" in gbm.param_names
    assert np.array_equal(gbm.macro_species, ["AT"])

def test_BindingModel_get_concs():
    model_spec = """
    equilibria:
        A + B -> AB; K1
    species:
        AT = A + AB
        BT = B + AB
    """
    gbm = BindingModel(model_spec)
    
    # Pass params as ARRAY (LogK, dH)
    params = np.array([np.log(1e6), 0.0]) # K1 = 1e6, dH = 0
    macros = np.array([1e-5, 1e-5])  # AT, BT
    
    concs = gbm.get_concs(params, macros)
    # Returns [A, B, AB] array
    
    micro_names = gbm.micro_species
    c_dict = dict(zip(micro_names, concs))
    
    assert c_dict["AB"] > 0
    assert np.isclose(c_dict["A"] + c_dict["AB"], 1e-5)


def test_root_selection_continuity():
    """
    Verify continuity-tracking root selection for a 2-site symmetric model.

    As AT increases (more ligand titrated into fixed ET), the free ligand [A]
    should be monotonically non-decreasing. Mass balance should hold at every
    injection point.
    """
    model_spec = """
    equilibria:
        E + A -> EA; K1
        EA + A -> EA2; K2
    species:
        ET = E + EA + EA2
        AT = A + EA + 2*EA2
    reparameterize:
        K2 = K1
    """
    bm = SymbolicBindingModel(model_spec)

    ET = 1e-5
    AT_values = np.linspace(0, 4e-5, 20)
    reg_params = {"K1": np.log(1e6), "dH_1": 0.0, "dH_2": 0.0}

    prev_c = None
    free_A_values = []

    for at in AT_values:
        result = bm.solve_concentrations(reg_params, {"ET": ET, "AT": at}, prev_c=prev_c)
        c_key = str(bm.c_symbol)
        prev_c = result.get(c_key)
        free_A_values.append(prev_c)

        # Mass balance: A + EA + 2*EA2 == AT
        mass_balance = result["A"] + result["EA"] + 2 * result["EA2"]
        assert np.isclose(mass_balance, at, atol=1e-15, rtol=1e-6), (
            f"Mass balance failed at AT={at:.2e}: got {mass_balance:.4e}"
        )

    # Free [A] should be non-decreasing across the titration
    free_A = np.array(free_A_values)
    diffs = np.diff(free_A)
    assert np.all(diffs >= -1e-18), (
        f"Free [A] is not monotonically non-decreasing — possible wrong-root selection.\n"
        f"Diffs: {diffs}"
    )
