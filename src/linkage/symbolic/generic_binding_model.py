import numpy as np
import pandas as pd
import warnings
from .model import SymbolicBindingModel

class GenericBindingModel:
    """
    Wrapper for SymbolicBindingModel to interface with Linkage/DataProb.
    """

    def __init__(self, model_spec, debug=False):
        if model_spec is None:
            raise ValueError("No model specification provided")

        self.model_spec = model_spec
        self._debug = debug

        # Initialize the new Symbolic implementation
        self._bm = SymbolicBindingModel(model_spec, debug=debug)

        # Expose properties expected by GlobalModel
        self._equilibria = self._bm.physical_poly._equilibria
        self._constants = self._bm.equilibrium_constants # Original Ks
        self._micro_species = self._bm.physical_poly._micro_species
        self._macro_species = self._bm.physical_poly._macro_species
        
        # Parameters to fit
        # Note: regression_params includes Ks, dHs, and new params (alpha, etc.)
        self._fit_params = self._bm.regression_params
        
        # dH handling:
        # SymbolicBindingModel includes dH in regression_params if they are relevant.
        # We need to expose this so GlobalModel knows about them.
        
        # Internal storage
        self._concentrations_df = pd.DataFrame(columns=self._micro_species, dtype=float)
        self._last_result = None

        # Expose properties for external access
        self.simplified_eqs = self._bm.physical_poly.simplified_eqs
        self.solved_vars = self._bm.physical_poly.solved_vars
        self.final_ct = self._bm.physical_poly.binding_polynomial

        # Check if Jacobian is available
        if hasattr(self._bm, "get_conc_jacobian_vs_regression"):
            self.jacobian_function = True # Flag for GlobalModel check
        else:
            self.jacobian_function = None

    @property
    def param_names(self):
        return np.array(self._fit_params)

    @property
    def macro_species(self):
        return np.array(self._macro_species)

    @property
    def micro_species(self):
        return np.array(self._micro_species)
        
    @property
    def equilibria(self):
        return self._equilibria

    @property
    def concentrations_df(self):
        return self._concentrations_df
        
    @property
    def reparam_rules(self):
        return self._bm.reparam_rules

    def get_physical_params(self, reg_params_dict):
        return self._bm.get_physical_params(reg_params_dict)

    def get_physical_jacobian(self, reg_params_dict):
        return self._bm.get_physical_jacobian(reg_params_dict)

    @property
    def physical_param_names(self):
        return self._bm.mapper.physical_params

    def get_concs(self, param_array, macro_array):
        """
        Calculate concentrations.
        param_array: values of self.param_names.
                     NOTE: DataProb/GlobalModel usually passes LOG(K) but LINEAR(dH).
                     We need to handle this transform here.
        macro_array: values of self.macro_species
        """
        
        # 1. Prepare Regression Dictionary
        reg_dict = {}
        for i, p_name in enumerate(self._fit_params):
            val = param_array[i]
            
            # Application of Log/Linear transform assumption
            # Heuristic: If it starts with 'dH' or 'nuisance', it's Linear.
            # Else (K, alpha, etc) it's Log (passed as log, needs exp).
            if p_name.startswith('dH') or "nuisance" in p_name:
                reg_dict[p_name] = val
            else:
                reg_dict[p_name] = np.exp(val)
                
        # 2. Prepare Macro Dictionary
        macro_dict = dict(zip(self._macro_species, macro_array))
        
        # 3. Solve
        try:
            result = self._bm.solve_concentrations(reg_dict, macro_dict)
            self._last_result = result
            
            # Store concentrations
            concs_dict = {s: result[s] for s in self._micro_species}
            
            # Record to DF
            df_row = pd.DataFrame([concs_dict], columns=self._micro_species)
            self._concentrations_df = pd.concat([self._concentrations_df, df_row], ignore_index=True)
            
            return np.array([concs_dict[s] for s in self._micro_species])
            
        except Exception as e:
            if self._debug: print(f"Solving failed: {e}")
            self._last_result = None
            return np.full(len(self._micro_species), np.nan)

    def get_numerical_jacobian(self, concs_dict):
        """
        Returns d[MicroSpecies]/d[FitParams].
        Uses concs_dict (params and concentrations) to calculate Jacobian at a specific point.
        """
        if concs_dict is None:
            return None
            
        try:
            # 1. Map Regression Params (in concs_dict) to Physical Params
            # concs_dict contains 'K1' (value), 'dH' (value), etc. derived from GlobalModel.
            phys_params = self._bm.mapper.get_physical_params(concs_dict)
            
            # 2. Prepare Solver Result Dict (needs Reg Params, Phys Params, Concs)
            # concs_dict actually contains Reg Params and Species Concs.
            solver_result = {**concs_dict, **phys_params}
            
            # 3. Get Jacobian dC/dP_lin
            J_lin = self._bm.get_conc_jacobian_vs_regression(solver_result)
            
            # 4. Apply Chain Rule for Log params
            # If P_lin = exp(P_fit) -> dP_lin/dP_fit = exp(P_fit) = P_lin
            # If P_lin = P_fit -> dP_lin/dP_fit = 1
            
            J_fit = J_lin.copy()
            
            for i, p_name in enumerate(self._fit_params):
                if not (p_name.startswith('dH') or "nuisance" in p_name):
                    # It was a Log parameter
                    # Scale column by P_lin value
                    p_val_lin = concs_dict.get(p_name, 1.0) # Should be in dict
                    J_fit[:, i] *= p_val_lin
            
            return J_fit
            
        except Exception as e:
            if self._debug: print(f"Jacobian failed: {e}")
            return None