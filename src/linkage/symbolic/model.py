import numpy as np
import pandas as pd
from sympy import symbols, diff, Matrix, lambdify, Poly, simplify, sympify
import warnings
from .polynomial import BindingPolynomial
from .parameter_map import ParameterMapper

class SymbolicBindingModel:
    """
    A high-level class that wraps BindingPolynomial to provide specific functionality
    for regression, including proper handling of reparameterization and Jacobians.

    Attributes:
        physical_poly (BindingPolynomial): The polynomial derived from the physical model (constants un-substituted).
        param_mapper (ParameterMapper): Handles mapping from Regression Params -> Physical Params.
        regression_params (list): Names of parameters to be optimized.
        physical_params (list): Names of meaningful physical parameters (Ks, dHs).
    """

    def __init__(self, model_spec, debug=False):
        self._model_spec = model_spec
        self._debug = debug
        
        # 1. Parse Pre-processing
        # We want to strip the reparameterize block before passing to BindingPolynomial
        # so we can handle it explicitly.
        self._clean_spec, self._reparam_block = self._split_reparam_section(model_spec)
        
        # 2. Get Physical Model (Standard BindingPolynomial)
        # This gives us P(c, Ks) where Ks are the original physical constants.
        self.physical_poly = BindingPolynomial(self._clean_spec, debug=debug)
        
        # 3. Identify all Physical Parameters
        # These are the Ks from the polynomial + any dHs or other parameters we need.
        # Since BindingPolynomial only finds Ks, we need to parse dHs from the reparam block or context.
        # But wait, dHs are usually implicit until we do ITC.
        # However, for reparameterization, we need to know they exist.
        # We will scan the reparam block for any LHS variables that aren't Ks.
        
        self.equilibrium_constants = self.physical_poly._constants # ['K1', 'K2', ...]
        
        # Auto-generate dH parameters for every K
        self.enthalpy_params = [self._derive_dH_name(k) for k in self.equilibrium_constants]
        
        # Identify any other physical parameters from reparam block (e.g. if they introduce new things not tied to Ks, though rare)
        # We pass both Ks and dHs as "existing" so we only find truly new stuff or mapped vars
        self.existing_physical = self.equilibrium_constants + self.enthalpy_params
        self.other_physical_params = self._identify_all_reparam_symbols(self._reparam_block, self.existing_physical)
        
        self.all_physical_params = sorted(list(set(self.existing_physical + self.other_physical_params)))
        
        # 4. Process Reparameterization Rules
        self.reparam_rules_dict = self._parse_reparam_rules(self._reparam_block)
        
        # 5. Create Mapper
        self.mapper = ParameterMapper(self.all_physical_params, self.reparam_rules_dict)
        self.regression_params = self.mapper.regression_params
        
        # 6. Setup Symbolic Jacobian (d[Species]/d[RegressionParams])
        # We will use the Chain Rule:
        # J_reg = J_phys @ J_map
        # where J_phys = d[Species]/d[Physical]
        #       J_map  = d[Physical]/d[Regression] (from Mapper)
        
        self._setup_symbolic_jacobian_components()

    @property
    def reparam_rules(self):
        return self.mapper.rules_sympy

    def get_physical_params(self, reg_params_dict):
        return self.mapper.get_physical_params(reg_params_dict)

    def get_physical_jacobian(self, reg_params_dict):
        return self.mapper.get_jacobian(reg_params_dict)


    def _derive_dH_name(self, k_name):
        # User convention: dH_x for Kx
        if k_name.startswith('K'):
            suffix = k_name[1:]
            return f"dH_{suffix}"
        else:
            # Fallback
            return f"dH_{k_name}"

    def _log(self, msg):
        if self._debug: print(f"[SymbolicBindingModel] {msg}")

    def _split_reparam_section(self, spec):
        lines = spec.split('\n')
        clean_lines = []
        reparam_lines = []
        in_reparam = False
        
        for line in lines:
            if 'reparameterize:' in line:
                in_reparam = True
                continue
            if in_reparam:
                # Check if we hit another section
                if line.strip().endswith(':') and not '=' in line and line.strip() != "reparameterize:":
                    in_reparam = False
                    clean_lines.append(line)
                else:
                    reparam_lines.append(line)
            else:
                clean_lines.append(line)
                
        return "\n".join(clean_lines), "\n".join(reparam_lines)

    def _parse_reparam_rules(self, reparam_str):
        rules = {}
        for line in reparam_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                rules[lhs.strip()] = rhs.strip()
        return rules

    def _identify_all_reparam_symbols(self, reparam_str, existing_ks):
        """
        Identify all symbols used in the reparameterization block.
        This includes LHS (dependent) and RHS (independent) variables.
        """
        found_symbols = set()
        
        # Use sympify to parse RHS to find hidden variables (e.g. alpha in K2 = K1 * alpha)
        lines = reparam_str.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                found_symbols.add(lhs.strip())
                
                try:
                    # sympify might use existing Ks if we don't be careful, but we just want symbol names
                    # We assume anything that isn't a math function is a symbol
                    expr = sympify(rhs.strip())
                    for sym in expr.free_symbols:
                        found_symbols.add(str(sym))
                except Exception as e:
                    if self._debug: print(f"Warning: could not parse RHS of {line}: {e}")
                    pass
        
        # Return symbols that are NOT in existing_ks (to avoid duplicates in the 'other' list)
        return sorted(list(found_symbols - set(existing_ks)))

    def _setup_symbolic_jacobian_components(self):
        # We borrow logic from the existing GenericBindingModel to get J_phys
        # But we implement it here cleanly.
        
        self._log("Deriving Jacobians...")
        
        # A. J_phys: d[Species]/d[Physical_K]
        # Only relevant for the K parameters. dH parameters don't affect species concentrations directly.
        # So we compute d[S]/d[K] for K in self.equilibrium_constants.
        
        # 1. Variables
        self.c_symbol = self.physical_poly._c_symbol
        self.species_symbols = [self.physical_poly.symbols[s] for s in self.physical_poly._micro_species]
        self.k_symbols = [self.physical_poly.symbols[k] for k in self.equilibrium_constants]
        
        # 2. Polynomial P(c, K)
        P = self.physical_poly.binding_polynomial
        
        # 3. Implicit Theorem: dc/dK = - (dP/dK) / (dP/dc)
        dP_dc = diff(P, self.c_symbol)
        dP_dKs = [diff(P, k) for k in self.k_symbols]
        
        # dc_dKs is a list of expressions
        dc_dKs = [-dpk / dP_dc for dpk in dP_dKs]
        
        # 4. For each species S(c, K):
        # dS/dK = (dS/dc)*(dc/dK) + (dS/dK)_partial
        
        # We need the expressions for species.
        # Solved vars (Base Macros):
        solved_vars = self.physical_poly.solved_vars
        # Simplified Eqs (Micros):
        simplified_eqs = self.physical_poly.simplified_eqs
        
        self.species_exprs = {}
        
        # Ensure that ALL species are expressed in terms of C, Ks, and Totals (no inter-species dependencies).
        # solved_vars expresses Base Vars in terms of (Total, C, Ks).
        # simplified_eqs expresses Complex Species in terms of (C, Ks, Base Vars).
        # We must substitute Base Vars into simplified_eqs.
        base_subs = {s: expr for s, expr in solved_vars.items()}
        
        for s_name in self.physical_poly._micro_species:
            s_sym = self.physical_poly.symbols[s_name]
            if s_sym == self.c_symbol:
                self.species_exprs[s_name] = self.c_symbol
            elif s_sym in solved_vars:
                self.species_exprs[s_name] = solved_vars[s_sym]
            elif s_sym in simplified_eqs:
                # Substitute base vars
                expr = simplified_eqs[s_sym]
                self.species_exprs[s_name] = expr.subs(base_subs)
            else:
                if self._debug: print(f"DEBUG: {s_name} NOT found in solved/simplified. Mapping to self.")
                self.species_exprs[s_name] = s_sym 
        
        rows = []
        for s_name in self.physical_poly._micro_species:
            expr = self.species_exprs[s_name]
            row = []
            
            # Derivatives w.r.t Ks
            d_expr_dc = diff(expr, self.c_symbol)
            
            for i, k_sym in enumerate(self.k_symbols):
                # Chain rule
                term1 = d_expr_dc * dc_dKs[i]
                term2 = diff(expr, k_sym)
                row.append(term1 + term2)
            rows.append(row)
            
        self.J_phys_symbolic = Matrix(rows) # Shape: (N_species, N_Ks)
        
        # Use lambdify for J_phys
        # INPUTS must include C, Ks, AND Totals because species expressions now depend on Totals.
        # Previously we assumed only C and Ks, but Base Vars introduce Totals.
        # We must detect args dynamically.
        
        self.input_syms_J_phys = sorted(list(self.J_phys_symbolic.free_symbols), key=lambda s: s.name)
        self.J_phys_func = lambdify(self.input_syms_J_phys, self.J_phys_symbolic, modules="numpy")
        
    def get_conc_jacobian_vs_regression(self, concentrations_dict, calibration_dict={}):
        """
        Returns d[Species]/d[RegressionParams].
        
        Args:
            concentrations_dict: dict with 'C', 'K1', 'K2'... values.
            calibration_dict: currently unused, but good for future.
            
        Returns:
            J (ndarray): (N_species x N_regression)
        """
        # 1. Calculate J_phys (N_species x N_Ks)
        # We only need the K subset of physical parameters for this part.
        
        try:
            # We must use proper arguments matching input_syms_J_phys
            # which might include C, Ks, and Totals.
            args = []
            for sym in self.input_syms_J_phys:
                args.append(concentrations_dict[sym.name])
            
            J_phys_val = np.array(self.J_phys_func(*args)) # (N_spec, N_K)
            
        except KeyError as e:
            # Missing parameter value
            raise ValueError(f"Missing value for {e} in concentrations_dict. Required: {[s.name for s in self.input_syms_J_phys]}")
            
        # 2. Calculate J_map (N_K x N_Reg)
        # We need the full Jacobian from the mapper d[Physical]/d[Regression]
        # Then slice it to get only d[Ks]/d[Reg]
        
        # We presume the 'concentrations_dict' contains the Physical values (calculated by Forward Map).
        # Wait, get_jacobian needs the Regression values to evaluate derivatives.
        # So we need to ensure we have access to the original regression inputs.
        # This might be tricky if not passed in.
        # For now, let's assume the user passes a dict that "has everything" or we update the API.
        
        # Better: The user calls 'get_regression_jac(reg_values)' -> we convert reg to phys internally?
        # But concentrations depend on solve.
        
        # Let's assume the user is calling this after solving.
        # We need regression param values.
        
        # If they are not in the dict, we fail.
        reg_vals = {p: concentrations_dict[p] for p in self.regression_params if p in concentrations_dict}
        if len(reg_vals) != len(self.regression_params):
             # Try to see if we can get them or if we proceed with partials?
             pass

        J_map_full = self.mapper.get_jacobian(concentrations_dict) # (N_Phys_Total x N_Reg)
        
        # indices of Ks in all_physical_params
        k_indices = [self.all_physical_params.index(k) for k in self.equilibrium_constants]
        
        J_map_Ks = J_map_full[k_indices, :] # (N_K x N_Reg)
        
        # 3. Multiply
        # J_reg = J_phys @ J_map_Ks
        # (N_spec, N_K) @ (N_K, N_Reg) -> (N_spec, N_Reg)
        
        J_reg = J_phys_val @ J_map_Ks
        
        return J_reg

    def solve_concentrations(self, regression_params_dict, macro_concentrations_dict):
        """
        High level solver.
        1. Map Reg -> Physical
        2. Solve P(c, K_phys) for c
        3. Back substitute for species
        4. Return full dict including derived physical parameters
        """
        # 1. Map
        phys_vals = self.mapper.get_physical_params(regression_params_dict)
        
        # 2. Create param dict for solver
        # Solver needs Ks and Total Macros
        solver_input = {**phys_vals, **macro_concentrations_dict}
        
        coeffs_sym = self.physical_poly.get_polynomial_coefficients() # [An, An-1, ..., A0]
        
        # Lambdify coeffs once for speed?
        if not hasattr(self, '_coeff_funcs'):
            self._coeff_funcs = []
            self._coeff_args_maps = [] # List of symbols for each coeff
            
            for c_expr in coeffs_sym:
                # Find all free symbols in this coefficient
                # These could be Ks or Total Macro concentrations
                free_syms = sorted(list(c_expr.free_symbols), key=lambda s: s.name)
                self._coeff_args_maps.append(free_syms)
                self._coeff_funcs.append(lambdify(free_syms, c_expr, modules='numpy'))
        
        coeffs_num = []
        for i, func in enumerate(self._coeff_funcs):
            # Prepare args for this specific coefficient function
            needed_syms = self._coeff_args_maps[i]
            # Map symbol names to values from solver_input
            # solver_input keys are strings. symbol.name is string.
            try:
                func_args = [solver_input[s.name] for s in needed_syms]
            except KeyError as e:
                # Provide a better error message
                missing = [s.name for s in needed_syms if s.name not in solver_input]
                raise KeyError(f"Missing value for symbols {missing} required to evaluate polynomial coefficients. Solver input keys: {list(solver_input.keys())}")
                
            val = func(*func_args)
            coeffs_num.append(val)
        
        # Solve
        roots = np.roots(coeffs_num)
        
        # Filter roots (real, positive, < CT)
        # Find CT name
        ct_name = self.physical_poly._ct_macrospecies_name
        ct_val = solver_input[ct_name]
        
        real_roots = roots[np.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= 0) & (real_roots <= ct_val)]
        
        if len(valid_roots) == 0:
             # Relax constraint slightly for numerical noise
             valid_roots = real_roots[(real_roots >= -1e-12) & (real_roots <= ct_val * 1.00001)]
             if len(valid_roots) == 0:
                 c_sol = 0.0 # Error state?
             else:
                 c_sol = max(0.0, np.min(valid_roots)) # Usual binding logic: smallest valid root? Or largest?
                 # Actually for binding poly C + ... = CT, usually unique positive root.
                 # Let's take the one that makes sense. Usually there is only 1 in range.
                 # If multiple, take the relevant one.
                 pass
        
        if len(valid_roots) > 0:
            c_sol = max(0.0, np.min(valid_roots)) 
        else:
             c_sol = 0.0 
             
        # 3. Species
        result = {**regression_params_dict, **phys_vals, **macro_concentrations_dict}
        result[str(self.c_symbol)] = c_sol
        
        # Eval species
        if not hasattr(self, '_species_funcs'):
            self._species_funcs = {}
            self._species_args_maps = {}
            
            for s_name, expr in self.species_exprs.items():
                free_syms = sorted(list(expr.free_symbols), key=lambda s: s.name)
                # Ensure C is in the list if not already (it should be)
                # Actually free_symbols includes it.
                self._species_args_maps[s_name] = free_syms
                self._species_funcs[s_name] = lambdify(free_syms, expr, modules='numpy')
            
        for s_name, func in self._species_funcs.items():
            needed_syms = self._species_args_maps[s_name]
            # solver_input plus 'C'
            full_input = {**solver_input, str(self.c_symbol): c_sol}
            
            try:
                func_args = [full_input[s.name] for s in needed_syms]
            except KeyError:
                missing = [s.name for s in needed_syms if s.name not in full_input]
                raise KeyError(f"Missing value for symbols {missing} required to evaluate species {s_name}.")
                
            result[s_name] = func(*func_args)
            
        return result
