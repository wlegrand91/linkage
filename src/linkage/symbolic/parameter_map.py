import numpy as np
from sympy import symbols, sympify, Matrix, diff, lambdify

class ParameterMapper:
    """
    Handles certain parameters being defined as functions of other parameters.
    Maintains the relationship between 'Physical Parameters' (used in the chemical model)
    and 'Regression Parameters' (coefficients that are actually optimized).
    """
    def __init__(self, physical_params, reparam_rules_dict):
        """
        Args:
            physical_params (list of str): The full list of physical parameters (e.g. ['K1', 'K2', 'dH_1']).
            reparam_rules_dict (dict): Dictionary mapping {dependent_var_name: expression_string}.
                Example: {'K2': 'K1 * alpha', 'dH_2': 'dH_1'}
        """
        self.physical_params = sorted(physical_params)
        self.reparam_rules_str = reparam_rules_dict
        
        self.regression_params = []
        self.rules_sympy = {}
        self.mapping_funcs = {}
        self.jacobian_func = None
        
        self._parse_rules()
        
    def _parse_rules(self):
        # 1. We need a consistent set of Symbol objects.
        # We will create a map {name: Symbol(name)} and ALWAYS use these objects.
        
        # Start with physical params
        self.symbols_map = {p: symbols(p) for p in self.physical_params}
        
        # Helper to get or create symbol
        def get_sym(name):
            if name not in self.symbols_map:
                self.symbols_map[name] = symbols(name)
            return self.symbols_map[name]

        # 2. Parse rules to find dependencies and NEW variables
        dependent_vars = set(self.reparam_rules_str.keys())
        potential_independent = set(self.physical_params) - dependent_vars
        
        new_symbols_found = set()
        
        self.rules_sympy = {}
        
        for dep_name, expr_str in self.reparam_rules_str.items():
            # We must parse using OUR symbols map to ensure object identity
            # AND we must catch new symbols that appear in the string.
            
            # Since we don't know the new symbols yet, we can't pre-populate locals completely.
            # But sympify can parse and we can swap symbols, or we can use custom dict.
            # A robust way: parse, find symbols by name, replace with our canonical symbols.
            
            try:
                # First parse generally
                temp_expr = sympify(expr_str)
            except Exception as e:
                raise ValueError(f"Failed to parse rule {dep_name} = {expr_str}: {e}")
            
            # Now traverse free symbols, ensure they are in our map
            final_expr = temp_expr
            for sym in temp_expr.free_symbols:
                s_name = str(sym)
                canonical_sym = get_sym(s_name)
                # Replace in expression if it's a different object (it likely is)
                final_expr = final_expr.subs(sym, canonical_sym)
                
                # Check if it's a new parameter
                if s_name not in self.physical_params and s_name not in dependent_vars:
                    new_symbols_found.add(s_name)

            self.rules_sympy[get_sym(dep_name)] = final_expr

        # Regression parameters are:
        # (PhysicalParams - DependentParams) + NewParams
        self.regression_params = sorted(list(potential_independent) + list(new_symbols_found))
        
        # 3. Build Full Mapping (Physical -> Expression of Regression)
        self.full_mapping_sympy = {}
        
        for p_name in self.physical_params:
            p_sym = get_sym(p_name)
            if p_name in self.reparam_rules_str:
                self.full_mapping_sympy[p_name] = self.rules_sympy[p_sym]
            else:
                self.full_mapping_sympy[p_name] = p_sym

        # 4. Iterative Substitution
        # We need to substitute until only regression parameters remain.
        reg_param_symbols = {get_sym(s) for s in self.regression_params}
        reg_param_names = set(self.regression_params)
        
        for _ in range(len(self.physical_params) + 5):
            dirty = False
            for p_name in self.physical_params:
                expr = self.full_mapping_sympy[p_name]
                
                # Check if we have symbols that are NOT in regression params
                # (and thus must be dependent variables)
                
                # We iterate free_symbols
                # If we find a symbol that IS NOT a regression param, we look for a rule.
                
                current_syms = expr.free_symbols
                should_sub = False
                for s in current_syms:
                    if str(s) not in reg_param_names:
                        should_sub = True
                        break
                
                if should_sub:
                    # Substitute using known rules
                    # We can bulk subs
                    # We need to be careful: subs(dict) is unordered?
                    # Generally safely done iteratively.
                    
                    # We want to sub 'rule' for 'dependent_var'.
                    # Which rule? The one from rules_sympy.
                    # Or better: the one from full_mapping_sympy (current state)?
                    # Using full_mapping_sympy allows chaining to propagate faster.
                    
                    # Create subs dict from full_mapping (but only for things that are keys there)
                    # We only want to sub things that ARE dependent variables.
                    
                    subs_dict = {}
                    for dep_name in dependent_vars:
                        dep_sym = get_sym(dep_name)
                        if dep_sym in current_syms:
                             # Use the rule for this dependent var
                             # Use the LATEST version from full_mapping?
                             # Or the raw rule? 
                             # If we use full_mapping, we get the benefit of previous work.
                             subs_dict[dep_sym] = self.full_mapping_sympy[dep_name]
                    
                    if subs_dict:
                        new_expr = expr.subs(subs_dict)
                        if new_expr != expr:
                            self.full_mapping_sympy[p_name] = new_expr
                            dirty = True
                            
            if not dirty:
                break
        
        # 5. Lambdify
        # Use canonical symbols for args
        self.reg_syms = [get_sym(p) for p in self.regression_params]
        self._setup_numerical_functions()

    def _setup_numerical_functions(self):
        # 1. Forward Map: Regression -> Physical
        # We'll make a single function that returns an array or specific dict.
        # Actually a dict is best for clarity.
        
        self.forward_funcs = {}
        for p_name, expr in self.full_mapping_sympy.items():
            # Lambdify
            # Ensure we only pass the args available in regression params
            self.forward_funcs[p_name] = lambdify(self.reg_syms, expr, modules="numpy")
            
        # 2. Jacobian: d(Physical)/d(Regression)
        # Matrix of shape (N_physical, N_regression)
        # J_ij = d(Physical_i) / d(Regression_j)
        
        rows = []
        for p_name in self.physical_params:
            expr = self.full_mapping_sympy[p_name]
            row_diffs = [diff(expr, r_sym) for r_sym in self.reg_syms]
            rows.append(row_diffs)
            
        self.jacobian_matrix = Matrix(rows)
        self.jacobian_lam = lambdify(self.reg_syms, self.jacobian_matrix, modules="numpy")

    def get_physical_params(self, regression_params_dict):
        """
        Convert regression parameters (dict) to physical parameters (dict).
        """
        # Ensure we have all args
        try:
            args = [regression_params_dict[p] for p in self.regression_params]
        except KeyError as e:
            raise KeyError(f"Missing regression parameter: {e}. Available: {list(regression_params_dict.keys())}")

        phys_vals = {}
        for p_name in self.physical_params:
            try:
                # Evaluate
                func = self.forward_funcs[p_name]
                val = func(*args)
                
                # Check if result is symbolic (sympy expression) instead of float
                # This happens if substitution wasn't complete or lambdify kept it symbolic
                if hasattr(val, 'free_symbols') and val.free_symbols:
                     # This should not happen if mapped correctly to regression params
                     raise ValueError(f"Result for {p_name} is still symbolic: {val}")
                     
                phys_vals[p_name] = float(val)
            except Exception as e:
                # Fallback or error
                print(f"Error evaluating {p_name}: {e}")
                print(f"  Expression: {self.full_mapping_sympy[p_name]}")
                print(f"  Args: {args}")
                print(f"  Regression Params: {self.regression_params}")
                
                # Check what symbols are in the expression
                free = self.full_mapping_sympy[p_name].free_symbols
                print(f"  Free symbols in expr: {free}")
                
                phys_vals[p_name] = np.nan
                raise e
        return phys_vals

    def get_jacobian(self, regression_params_dict):
        """
        Returns matrix d[Physical]/d[Regression]
        Rows: Physical Params (in order of self.physical_params)
        Cols: Regression Params (in order of self.regression_params)
        """
        args = [regression_params_dict[p] for p in self.regression_params]
        return np.array(self.jacobian_lam(*args))
