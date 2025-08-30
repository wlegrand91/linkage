import numpy as np
import pandas as pd
from sympy import symbols, expand, simplify, collect, prod, Poly, lambdify
from scipy.optimize import root_scalar # Still needed as a fallback or if not a polynomial
import warnings

class GenericBindingModel():
    """
    Base class for generating binding models from specifications.
    
    This class bypasses the docstring parsing from the BindingModel base class
    and instead processes a model specification provided as a constructor parameter.
    """

    def __init__(self, model_spec, debug=False):
        """
        Initialize binding model from specification.
        
        Parameters
        ----------
        model_spec : str
            Model specification string containing equilibria and species definitions
        debug : bool, optional
            If True, outputs detailed diagnostic information
        """
        if model_spec is None:
            raise ValueError("No model specification provided")

        self._model_spec = model_spec
        self._debug = debug
        
        # Parse the model specification
        self._equilibria, self._constants, self._species, self._micro_species, self._macro_species = self._parse_model_spec()
        
        # Set up symbolic representation (this now includes lambdification)
        self._setup_symbolic_model() # This will now pre-calculate more
        
        # Initialize concentrations DataFrame
        self._concentrations_df = pd.DataFrame(columns=self._micro_species, dtype=float)
    
    def print_summary(self):
        """Print a summary of the model's key properties."""
        print("\n===== GENERIC BINDING MODEL SUMMARY =====")
        print(f"Constants (parameters to fit as ln(K)): {self._constants}")
        print(f"Microspecies: {self._micro_species}")
        print(f"Macrospecies (total concentrations): {self._macro_species}")
        print(f"Equilibria:")
        for k, (reactants, products) in self._equilibria.items():
            print(f"  {' + '.join(reactants)} -> {' + '.join(products)}; {k}")
        
        print(f"\nSymbolic final conservation equation (set to 0): {self.final_ct_for_C_poly_extraction}")
        
        if hasattr(self, '_lambdified_final_ct'):
            print(f"Lambdified function for final conservation equation created for fallback root finding.")
            print(f"  Expected non-C args for lambdified func: {[s.name for s in self._non_C_params_for_lambdify_final_ct]}")

        if hasattr(self, '_is_final_ct_polynomial_in_C'):
            if self._is_final_ct_polynomial_in_C:
                print("Final conservation equation IS a polynomial in C (after other substitutions).")
                print(f"  Polynomial C symbol: {self._c_symbol.name}")
                print(f"  Polynomial coefficient symbols (excluding C): {[s.name for s in self._poly_coeff_symbols_ordered]}")
            else:
                print("Final conservation equation IS NOT a simple polynomial in C (after other substitutions). Will use numerical root finding.")
        print("===== END SUMMARY =====\n")

    def _log(self, message):
        if self._debug:
            print(f"DEBUG: {message}")

    def _parse_model_spec(self):
        # ... (same as your original _parse_model_spec, _parse_equilibria_section, _parse_species_section, _validate_and_extract_species)
        self._log("Parsing model specification")
        equilibria, constants = self._parse_equilibria_section()
        species = self._parse_species_section()
        micro_species, macro_species = self._validate_and_extract_species(equilibria, species)
        return equilibria, constants, species, micro_species, macro_species

    def _parse_equilibria_section(self):
        # ... (same as your original)
        equilibria = {}
        constants = []
        in_equilibria = False
        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'equilibria:' in line:
                in_equilibria = True
                continue
            elif 'species:' in line or not line:
                in_equilibria = False # Corrected from 'True' to 'False'
                if 'species:' in line: continue # If it's the species line, don't skip parsing it later
            if in_equilibria and line:
                if ';' not in line or '->' not in line:
                    self._log(f"Skipping malformed equilibria line: '{line}'")
                    continue
                reaction, K = line.split(';')
                K = K.strip()
                if not K or not K.startswith('K'):
                    self._log(f"Skipping equilibria with invalid constant: '{line}'")
                    continue
                reactants_str, products_str = reaction.split('->')
                reactants = [r.strip() for r in reactants_str.split('+') if r.strip()]
                products = [p.strip() for p in products_str.split('+') if p.strip()]
                equilibria[K] = [reactants, products]
                if K not in constants:
                    constants.append(K)
                self._log(f"Parsed equilibrium: {reactants} -> {products}; {K}")
        if not equilibria:
            raise ValueError("No valid equilibria found in model specification")
        return equilibria, constants

    def _parse_species_section(self):
        # ... (same as your original)
        species = {}
        in_species = False
        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'species:' in line:
                in_species = True
                continue
            # elif not line: # This was causing species section to be skipped if a blank line appeared before it
            #     continue
            if in_species and '=' in line: # Make sure we are in species section AND line is an equation
                lhs, rhs = line.split('=')
                macro = lhs.strip()
                micro_species_list = []
                stoichiometries = []
                for item in rhs.strip().split('+'):
                    item = item.strip()
                    if '*' in item:
                        coef, species_name_str = item.split('*')
                        try:
                            stoich = int(coef.strip())
                            species_name_str = species_name_str.strip()
                            micro_species_list.append(species_name_str)
                            stoichiometries.append(stoich)
                        except ValueError:
                            self._log(f"Skipping malformed stoichiometry: '{item}'")
                            continue
                    else:
                        species_name_str = item.strip()
                        if species_name_str:
                            micro_species_list.append(species_name_str)
                            stoichiometries.append(1)
                species[macro] = (micro_species_list, stoichiometries)
                self._log(f"Parsed species: {macro} = {micro_species_list} with stoichiometries {stoichiometries}")
            elif in_species and line and 'equilibria:' in self._model_spec.split(line)[0]: # End of species if new section starts
                 in_species = False

        if not species:
            raise ValueError("No valid species definitions found in model specification")
        return species

    def _validate_and_extract_species(self, equilibria, species):
        # ... (same as your original)
        micro_in_equilibria = set()
        for K_val, (reactants, products) in equilibria.items():
            micro_in_equilibria.update(reactants)
            micro_in_equilibria.update(products)
        micro_in_species = set()
        for macro_val, (micros, _) in species.items():
            micro_in_species.update(micros)
        
        all_micro_species_set = micro_in_equilibria.union(micro_in_species)
        if micro_in_equilibria != micro_in_species:
            only_in_eq = micro_in_equilibria - micro_in_species
            only_in_sp = micro_in_species - micro_in_equilibria
            warning_msg = "WARNING: Mismatch between microspecies in equilibria and species definitions"
            if only_in_eq: warning_msg += f"\n  Species only in equilibria: {', '.join(only_in_eq)}"
            if only_in_sp: warning_msg += f"\n  Species only in species definitions: {', '.join(only_in_sp)}"
            print(warning_msg)
        
        micro_species_list = sorted(list(all_micro_species_set))
        macro_species_list = sorted(list(species.keys()))

        # Define C as the primary free species we solve for (convention)
        self._c_species_name = "C" # Assuming 'C' is always the species name for free calcium/ligand
        if self._c_species_name not in micro_species_list:
             warnings.warn(f"WARNING: Default free species '{self._c_species_name}' not found in microspecies list: {micro_species_list}. Root finding might be problematic.")
        
        # Ensure CT (or equivalent) is present for upper bound in root finding
        self._ct_macrospecies_name = None
        for m_name in macro_species_list:
            if m_name.endswith('T') and self._c_species_name in m_name : # e.g., CT, CaT, LT
                self._ct_macrospecies_name = m_name
                break
        if not self._ct_macrospecies_name:
             # Fallback: try to find any macro species containing 'C' and ending in 'T'
            for m_name in macro_species_list:
                if 'C' in m_name and m_name.endswith('T'):
                    self._ct_macrospecies_name = m_name
                    warnings.warn(f"WARNING: No direct '{self._c_species_name}T' macrospecies found. Using '{m_name}' as CT for bounds.")
                    break
            if not self._ct_macrospecies_name:
                 # Last fallback: use the first macrospecies if only one is defined
                if len(macro_species_list) == 1:
                    self._ct_macrospecies_name = macro_species_list[0]
                    warnings.warn(f"WARNING: No CT-like species. Using '{self._ct_macrospecies_name}' for root-finding bounds.")
                else:
                    warnings.warn(f"WARNING: Cannot identify a CT-like macrospecies for '{self._c_species_name}'. Root finding bounds might be incorrect.")


        self._log(f"Validated microspecies: {micro_species_list}")
        self._log(f"Validated macrospecies: {macro_species_list}")
        return micro_species_list, macro_species_list


    def _setup_symbolic_model(self):
        self._log("Setting up symbolic model")
        self.symbols_dict = {name: symbols(name) for name in 
                             self._micro_species + self._macro_species + self._constants}
        
        # Ensure 'C' (or the designated c_species_name) is a symbol
        self._c_symbol = self.symbols_dict.get(self._c_species_name)
        if self._c_symbol is None:
            # This case should ideally be caught by _validate_and_extract_species, but as a safeguard:
            raise ValueError(f"Symbol for free species '{self._c_species_name}' not created. Check model spec.")

        self.equilibrium_eqs_dict = self._create_equilibrium_equations_dict() # Returns dict: {product_sym: rhs_expr}
        self.simplified_eqs = self._simplify_equilibrium_equations(self.equilibrium_eqs_dict)
        self.solved_vars, self.final_ct_for_C_poly_extraction = self._solve_conservation_equations(self.simplified_eqs)

        # Attempt to extract polynomial coefficients for C
        try:
            # Substitute all non-C symbols that will be numerical values later (Ks, other totals)
            # Keep C symbolic for poly extraction.
            # Create a list of symbols that are NOT C and will be parameters to the coeff function
            
            symbols_to_remain_for_poly_coeffs = []
            # These are Ks and Total concentrations (AT, ET, etc., excluding CT if CT is implicit in final_ct_for_C_poly_extraction)
            # CT (total C) is usually part of the final_ct_for_C_poly_extraction and should not be substituted yet if we are making a polynomial in free C.
            # The symbols in final_ct_for_C_poly_extraction, excluding self._c_symbol, are the ones whose numerical values will define the polynomial coeffs.
            
            # Identify symbols in final_ct_for_C_poly_extraction that are parameters for polynomial coefficients
            self._poly_coeff_parameters = [s for s in self.final_ct_for_C_poly_extraction.free_symbols if s != self._c_symbol]
            self.final_ct_poly_in_C = Poly(self.final_ct_for_C_poly_extraction, self._c_symbol)
            
            # Check if it's truly a polynomial in C (i.e., no C in denominators of coefficients)
            # This is implicitly handled if Poly() succeeds without raising an error for non-polynomial expressions.
            self._is_final_ct_polynomial_in_C = True
            self._log(f"Successfully created Poly object for C: {self.final_ct_poly_in_C.expr}")

            # Lambdify the coefficients of the polynomial in C
            # The coefficients will be functions of the *other* parameters (Ks, total concs)
            coeffs_sym = self.final_ct_poly_in_C.all_coeffs() # List of symbolic coefficients
            
            # Order of symbols for lambdifying coefficients must match the order in _poly_coeff_parameters
            self._poly_coeff_symbols_ordered = sorted(self._poly_coeff_parameters, key=lambda s: s.name)

            self._lambdified_coeffs_funcs = []
            for coeff_expr_sym in coeffs_sym:
                if not coeff_expr_sym.free_symbols: # If coefficient is a constant number
                    # Lambdify still works, or you can store the number directly
                     self._lambdified_coeffs_funcs.append(lambdify([], coeff_expr_sym, "numpy"))
                else:
                    # Ensure symbols in coeff_expr_sym are all in _poly_coeff_symbols_ordered
                    # Order for lambdify must match the arguments it will receive
                    args_for_this_coeff_lambdify = [s for s in self._poly_coeff_symbols_ordered if s in coeff_expr_sym.free_symbols]
                    if not args_for_this_coeff_lambdify and coeff_expr_sym.is_constant(): # handles numerical coeffs
                         self._lambdified_coeffs_funcs.append(lambda *args, val=float(coeff_expr_sym): val) # Returns the constant
                    elif args_for_this_coeff_lambdify :
                        self._lambdified_coeffs_funcs.append(lambdify(args_for_this_coeff_lambdify, coeff_expr_sym, "numpy"))
                    else: # Should not happen if coeff_expr_sym has free_symbols and they are not in _poly_coeff_symbols_ordered
                        self._log(f"Warning: Coeff expr {coeff_expr_sym} has free symbols not in ordered list. Treating as constant 0 for safety.")
                        self._lambdified_coeffs_funcs.append(lambda *args: 0.0)


            self._log(f"Lambdified polynomial coefficients for C. Number of coeffs: {len(self._lambdified_coeffs_funcs)}")

        except Exception as e: # E.g., if not a polynomial in C (sympy.PolynomialError:NotPolynomial)
            self._log(f"Failed to treat final_ct as polynomial in C or lambdify coeffs: {e}. Will use numerical root finding for final_ct.")
            self._is_final_ct_polynomial_in_C = False
            # Prepare for numerical root finding using lambdify on the whole final_ct_for_C_poly_extraction
            all_free_symbols_in_final_ct = list(self.final_ct_for_C_poly_extraction.free_symbols)
            if self._c_symbol not in all_free_symbols_in_final_ct:
                 # This should ideally not happen if C is part of the equation.
                 # If it does, it means C might have been eliminated or the equation is trivial.
                 self._log(f"Warning: C symbol '{self._c_symbol.name}' not found in final_ct_for_C_poly_extraction's free symbols. Fallback root finding may fail.")
                 # Setup a dummy lambdified function to avoid crashes, though it won't work.
                 self._lambdified_final_ct = lambda *args: np.nan 
                 self._non_C_params_for_lambdify_final_ct = []
            else:
                ordered_symbols_for_lambdify_final_ct = [self._c_symbol]
                other_symbols_final_ct = sorted([s for s in all_free_symbols_in_final_ct if s != self._c_symbol], key=lambda s: s.name)
                ordered_symbols_for_lambdify_final_ct.extend(other_symbols_final_ct)
                self._lambdified_final_ct = lambdify(ordered_symbols_for_lambdify_final_ct, self.final_ct_for_C_poly_extraction, "numpy")
                self._non_C_params_for_lambdify_final_ct = other_symbols_final_ct

        self._log("Symbolic model setup complete")

    def _create_equilibrium_equations_dict(self):
        # ... (Modified to return a dict directly for simplified_eqs)
        # Similar to your _create_equilibrium_equations but returns {product_sym: rhs_expr}
        eq_dict = {}
        for K, (reactants, products) in self._equilibria.items():
            if not products: continue
            for product_name in products:
                product_sym = self.symbols_dict[product_name]
                reactant_syms = [self.symbols_dict[r] for r in reactants]
                rhs = self.symbols_dict[K] * prod(reactant_syms)
                eq_dict[product_sym] = rhs
                self._log(f"Created eq_dict entry: {product_sym} = {rhs}")
        return eq_dict
        
    def _simplify_equilibrium_equations(self, eq_dict_from_create): # Takes dict
        # ... (same as your original, but operates on the dict)
        # eq_dict_from_create maps {product_sym: symbolic_rhs_expr}
        
        # Identify base variable symbols (these should not be substituted away)
        # Assuming self.base_vars contains strings like "A", "E" (from "AT", "ET")
        # and self._c_species_name is "C"
        base_var_strings = [macro[:-1] for macro in self._macro_species if macro.endswith('T')]
        # Add self._c_species_name if it's not already covered (e.g. if C is free but no CT)
        if self._c_species_name not in base_var_strings:
            base_var_strings.append(self._c_species_name)

        base_symbols_to_preserve = {self.symbols_dict[bvs] for bvs in base_var_strings if bvs in self.symbols_dict}
        self._log(f"Base symbols to preserve during simplification: {[s.name for s in base_symbols_to_preserve]}")

        simplified_expressions = {}
        for product_sym_being_defined, rhs_expr_to_simplify in eq_dict_from_create.items():
            current_expr = rhs_expr_to_simplify
            
            # Iteratively substitute until no more changes or max iterations
            # This loop substitutes intermediate complex species with their definitions in terms of simpler species
            # until RHS is in terms of base variables and constants.
            for _iteration in range(len(eq_dict_from_create) + 1): # Max iterations to prevent infinite loops
                made_change_in_iteration = False
                # Iterate over all possible substitutions defined in eq_dict_from_create
                for intermediate_complex_sym, its_definition_expr in eq_dict_from_create.items():
                    # Don't substitute the product we are currently defining with itself in its own definition.
                    if intermediate_complex_sym == product_sym_being_defined:
                        continue 
                    # Don't substitute away base variables if they appear on RHS of other definitions
                    if intermediate_complex_sym in base_symbols_to_preserve:
                        continue

                    if intermediate_complex_sym in current_expr.free_symbols:
                        current_expr = current_expr.subs(intermediate_complex_sym, its_definition_expr)
                        current_expr = expand(current_expr) # Expand after each substitution
                        made_change_in_iteration = True
                
                if not made_change_in_iteration:
                    break # No more substitutions can be made for this product_sym_being_defined
            
            # Collect terms with respect to the main free species (e.g., C) if desired, or just simplify
            # For now, just simplify the final expression for this product.
            simplified_expressions[product_sym_being_defined] = simplify(current_expr)
            self._log(f"Simplified equation: {product_sym_being_defined} = {simplified_expressions[product_sym_being_defined]}")

        return simplified_expressions

    def _solve_conservation_equations(self, simplified_equilibrium_dict): # Takes simplified_eqs
        # ... ( Largely similar to your original, ensure it uses self._ct_macrospecies_name and self._c_symbol correctly)
        # simplified_equilibrium_dict maps {micro_species_symbol: its_expr_in_terms_of_base_vars_and_Ks}
        
        solved_vars = {} # Will map {base_var_sym (e.g. A_sym): its_expr_in_terms_of_TotalMacro_and_C_and_Ks}
        
        # Identify base variable symbols (e.g. A from AT, E from ET)
        # These are the variables we want to solve for from their respective total equations,
        # to eventually substitute into the CT equation.
        # Exclude self._c_symbol because we are solving *for* C at the very end.
        base_vars_to_solve_for_syms = {self.symbols_dict[macro[:-1]] for macro in self._macro_species 
                                     if macro.endswith('T') and macro != self._ct_macrospecies_name 
                                     and macro[:-1] in self.symbols_dict}
        
        self._log(f"Base variables to solve from their Total equations: {[s.name for s in base_vars_to_solve_for_syms]}")

        # Solve for each base variable (like A from AT, E from ET) in terms of its Total, C, and Ks
        for total_macro_name_str, (micro_species_list_in_total, stoich_list_in_total) in self._species.items():
            base_var_of_this_total_str = total_macro_name_str[:-1] # e.g. "A" from "AT"
            base_var_of_this_total_sym = self.symbols_dict.get(base_var_of_this_total_str)

            if base_var_of_this_total_sym not in base_vars_to_solve_for_syms:
                continue # Skip if this is CT or not a base variable we're solving at this stage

            # Construct the RHS of TotalMacro = sum(stoich * micro_species_expr)
            # where micro_species_expr is already in terms of base variables (A, E, C...) and Ks
            rhs_sum_expr = 0
            for micro_str, stoich_val in zip(micro_species_list_in_total, stoich_list_in_total):
                micro_sym = self.symbols_dict.get(micro_str)
                if micro_sym is None: continue

                # Get the expression for this micro_species from simplified_equilibrium_dict
                # If micro_sym is a base variable itself (A, E, C), its "simplified_expr" is just the symbol itself.
                # simplified_equilibrium_dict contains expressions for *complexes* primarily.
                # Base free species like A, E, C will typically not be keys in simplified_equilibrium_dict
                # unless they are also products of some "identity" equilibrium (A -> A; KA=1), which is unusual.
                # So, if micro_sym is a base var, use micro_sym. If it's a complex, use its simplified expr.
                
                expr_for_this_micro = simplified_equilibrium_dict.get(micro_sym, micro_sym) # Default to micro_sym if not in dict
                rhs_sum_expr += stoich_val * expr_for_this_micro
            
            rhs_sum_expr = expand(rhs_sum_expr)
            
            # Now, rhs_sum_expr is in terms of base_var_of_this_total_sym, self._c_symbol, other base vars, and Ks.
            # We want to solve: TotalMacro_sym = rhs_sum_expr for base_var_of_this_total_sym
            # Example: AT_sym = A_sym * (coeff_of_A) + terms_without_A
            # So, A_sym = (AT_sym - terms_without_A) / coeff_of_A
            
            collected_expr = collect(rhs_sum_expr, base_var_of_this_total_sym)
            
            # Coefficient of base_var_of_this_total_sym in the collected expression
            # This coefficient should be an expression in terms of C_sym, other base_vars, and Ks
            coeff_of_base_var = collected_expr.coeff(base_var_of_this_total_sym, 1) # Power 1
            
            # Terms not containing base_var_of_this_total_sym
            terms_without_base_var = collected_expr.coeff(base_var_of_this_total_sym, 0) # Power 0 (constant term wrt base_var)
            
            if coeff_of_base_var == 0:
                self._log(f"Warning: Coefficient of {base_var_of_this_total_sym.name} is zero in its total equation. Cannot solve for it.")
                continue

            total_macro_sym = self.symbols_dict[total_macro_name_str]
            solution_for_base_var = (total_macro_sym - terms_without_base_var) / coeff_of_base_var
            solved_vars[base_var_of_this_total_sym] = simplify(solution_for_base_var)
            self._log(f"Solved for {base_var_of_this_total_sym.name} = {solved_vars[base_var_of_this_total_sym]}")

        # Construct the final CT equation
        if not self._ct_macrospecies_name:
             raise ValueError("CT macrospecies name not identified. Cannot construct final conservation equation for C.")
        
        ct_macro_name_str, (ct_micro_list, ct_stoich_list) = self._ct_macrospecies_name, self._species[self._ct_macrospecies_name]
        
        final_ct_rhs_expr = 0
        for micro_str, stoich_val in zip(ct_micro_list, ct_stoich_list):
            micro_sym = self.symbols_dict.get(micro_str)
            if micro_sym is None: continue
            expr_for_this_micro_in_ct = simplified_equilibrium_dict.get(micro_sym, micro_sym)
            final_ct_rhs_expr += stoich_val * expr_for_this_micro_in_ct
        
        final_ct_rhs_expr = expand(final_ct_rhs_expr)

        # Substitute the solved expressions for other base variables (A, E, etc.) into final_ct_rhs_expr
        for base_var_sym_solved, its_solution_expr in solved_vars.items():
            if base_var_sym_solved in final_ct_rhs_expr.free_symbols:
                final_ct_rhs_expr = final_ct_rhs_expr.subs(base_var_sym_solved, its_solution_expr)
                final_ct_rhs_expr = expand(final_ct_rhs_expr)
        
        # The equation to solve is: final_ct_rhs_expr - TotalCT_sym = 0
        # final_ct_rhs_expr should now primarily be a function of C_sym, Ks, and Total concentrations.
        total_ct_sym = self.symbols_dict[ct_macro_name_str]
        final_conservation_eq_for_C = simplify(final_ct_rhs_expr - total_ct_sym)
        self._log(f"Final conservation equation for C: {final_conservation_eq_for_C} = 0")
        
        return solved_vars, final_conservation_eq_for_C


    def _get_free_c(self, **param_dict_num_values): # param_dict_num_values has K and Total numerical values
        # param_dict_num_values contains numerical values for Ks and Total concentrations (AT, ET, CT etc.)
        
        if self._ct_macrospecies_name not in param_dict_num_values:
             self._log(f"CT-like species '{self._ct_macrospecies_name}' not in param_dict for _get_free_c. Cannot determine bounds.")
             return np.nan # Or a default like 0.0, but NaN is more indicative of an issue
        
        CT_numerical_val = param_dict_num_values[self._ct_macrospecies_name]
        if CT_numerical_val == 0: return 0.0

        if self._is_final_ct_polynomial_in_C:
            # Prepare arguments for lambdified coefficient functions
            # These are the numerical values of Ks, AT, ET, etc. (excluding CT if it's part of the polynomial structure directly)
            # The order must match self._poly_coeff_symbols_ordered
            
            coeff_param_values = []
            for sym_param in self._poly_coeff_symbols_ordered:
                if sym_param.name not in param_dict_num_values:
                    self._log(f"Error: Symbol {sym_param.name} needed for polynomial coefficient calculation not found in param_dict.")
                    return np.nan # Critical error
                coeff_param_values.append(param_dict_num_values[sym_param.name])

            numerical_coeffs = []
            for i, lamb_func in enumerate(self._lambdified_coeffs_funcs):
                # Determine which parameters this specific coefficient's lambdified function needs
                # This requires knowing the arg signature of each lamb_func, which is complex to get robustly here.
                # Simpler: pass all coeff_param_values; lambdify uses what it needs if args match.
                # This assumes lambdify was created with ordered symbols that match coeff_param_values structure.
                # More robust: if lamb_func was created from a coeff_expr_sym, its args are coeff_expr_sym.free_symbols
                # For now, this relies on the careful ordering in _setup_symbolic_model
                
                # Let's refine: Each lamb_func for a coefficient was made with specific args.
                # We need to extract those specific args from coeff_param_values.
                # This is tricky without storing the arg specification for each lamb_func.

                # Simplification: if lamb_func takes no args (e.g. constant coefficient)
                if not lamb_func.__code__.co_argcount: # No arguments
                    numerical_coeffs.append(lamb_func())
                else:
                    # This assumes that if it takes args, it takes all of coeff_param_values in the correct order
                    # This is a strong assumption based on how _lambdified_coeffs_funcs are created.
                    # A more robust way would be to store the arg names for each lamb_func.
                    # For now, let's proceed with the assumption of consistent argument ordering.
                    
                    # A quick check: if a coeff_expr_sym was constant, its lamb_func might be `lambda: const_val`
                    # or `lambdify([], const_val)`. If it had symbols, it was `lambdify(symbols_list, expr)`.
                    # The current structure of _lambdified_coeffs_funcs creation needs to be robust here.
                    # The lambda *args approach in _setup_symbolic_model for constant coefficients handles this.
                    try:
                        numerical_coeffs.append(lamb_func(*coeff_param_values))
                    except TypeError as te: # Mismatch in number of arguments
                        # This means the assumption that all lamb_funcs take all coeff_param_values is wrong.
                        # Fallback: try to find which specific params are needed for this coeff_func.
                        # This is where storing the arg spec per lamb_func would be better.
                        # For now, let's assume if it errors, it's a setup issue or constant.
                        self._log(f"TypeError calling lambdified coeff func {i}: {te}. Coeff params passed: {len(coeff_param_values)}. This indicates an issue in _setup_symbolic_model or that the coefficient is constant and its lambda wrapper is incorrect.")
                        # Attempt to see if it's a 0-arg constant lambda due to earlier setup
                        try:
                            numerical_coeffs.append(lamb_func()) # Try calling with no args
                        except TypeError: # Still fails, then it's a problem
                             numerical_coeffs.append(np.nan) # Mark as problematic
            
            if np.any(np.isnan(numerical_coeffs)):
                self._log(f"NaN encountered in numerical_coeffs. Polynomial root finding will fail. Coeffs: {numerical_coeffs}")
                # Fallback to numerical root finding if possible, or return NaN
                return self._get_free_c_numerical_fallback(**param_dict_num_values)


            # polyroots wants coeffs from C^0 to C^N
            # self.final_ct_poly_in_C.all_coeffs() gives highest power to lowest.
            # So, numerical_coeffs is also highest to lowest. We need to reverse.
            # Also ensure they are floats
            try:
                coeffs_for_polyroots = [float(c) for c in numerical_coeffs[::-1]]
            except Exception as e_float:
                self._log(f"Could not convert all numerical_coeffs to float: {numerical_coeffs}. Error: {e_float}")
                return self._get_free_c_numerical_fallback(**param_dict_num_values)

            if not coeffs_for_polyroots: # Empty list
                self._log("No coefficients for polynomial root finding.")
                return np.nan

            try:
                roots = np.polynomial.polynomial.polyroots(coeffs_for_polyroots)
                return self._get_real_root(roots, upper_bounds=[CT_numerical_val])
            except Exception as e_polyroots:
                self._log(f"numpy.polyroots failed: {e_polyroots}. Coeffs: {coeffs_for_polyroots}. Falling back.")
                # Fallback to numerical root finding
                return self._get_free_c_numerical_fallback(**param_dict_num_values)

        else: # Not a polynomial, or polynomial extraction failed, use numerical root_scalar
            return self._get_free_c_numerical_fallback(**param_dict_num_values)

    def _get_free_c_numerical_fallback(self, **param_dict_num_values):
        if not hasattr(self, '_lambdified_final_ct'):
            self._log("Error: _lambdified_final_ct not found for numerical fallback.")
            return np.nan

        CT_numerical_val = param_dict_num_values.get(self._ct_macrospecies_name, 0.0) # Default to 0 if CT not found
        if CT_numerical_val == 0 and self._ct_macrospecies_name in param_dict_num_values : return 0.0 # Explicitly 0 CT

        # Prepare args for the lambdified G_final_ct function
        # These are the numerical values of Ks, AT, ET, etc., in the order defined by _non_C_params_for_lambdify_final_ct
        g_func_args_num = []
        for sym_param in self._non_C_params_for_lambdify_final_ct:
            if sym_param.name not in param_dict_num_values:
                self._log(f"Error: Symbol {sym_param.name} needed for G_final_ct not found in param_dict.")
                return np.nan
            g_func_args_num.append(param_dict_num_values[sym_param.name])
        
        def G_final_ct(c_scalar_val):
            try:
                if hasattr(c_scalar_val, 'item'): c_scalar_val = c_scalar_val.item()
                # Call the lambdified function: C value first, then other *args
                return self._lambdified_final_ct(c_scalar_val, *g_func_args_num)
            except Exception as e_lambdify_G:
                self._log(f"Error in lambdified G_final_ct({c_scalar_val=}): {e_lambdify_G}")
                return np.nan
        
        # Bracketing logic (simplified from your original, can be expanded if needed)
        lower_b, upper_b = 1e-15, CT_numerical_val
        if lower_b >= upper_b: # Handle CT_numerical_val being very small or zero
            if CT_numerical_val > 0 : lower_b = CT_numerical_val / 100.0 # Ensure lower < upper if CT is tiny
            else: # CT is zero or negative (invalid)
                 self._log(f"Warning: CT value {CT_numerical_val} is not positive for bracketing in fallback.")
                 return 0.0 # Or np.nan

        try:
            f_low = G_final_ct(lower_b)
            f_high = G_final_ct(upper_b)

            if np.isnan(f_low) or np.isnan(f_high):
                self._log("NaN at bracket boundaries for root_scalar fallback.")
                return np.nan

            if f_low * f_high > 0:
                # Try to find a better bracket if signs are the same
                # This part needs careful implementation if simple bracketing fails often
                # For now, if initial bracket fails, we might return NaN or try a wider search
                self._log(f"Initial bracket [{lower_b:.2e}, {upper_b:.2e}] for fallback has same sign: f_low={f_low:.2e}, f_high={f_high:.2e}. Root finding may fail.")
                # Attempt a wider search or a different method if brentq requires a sign change
                # For brentq, a sign change is essential.
                # Could try a small perturbation or a log-spaced search if this happens often.
                # If simple bracket fails, one option is to test C=0 if allowed.
                # If G_final_ct(0) has opposite sign to f_high, use [0, upper_b].
                # However, C=0 can be problematic if C is in denominators symbolically.
                # For now, we proceed, and brentq will error if no sign change.
            
            sol = root_scalar(G_final_ct, bracket=[lower_b, upper_b], method='brentq', xtol=1e-12, rtol=1e-10)
            if sol.converged:
                # Additional check: ensure root is physically plausible (e.g. non-negative)
                # _get_real_root already does this for polynomial roots.
                if sol.root >= 0 and sol.root <= CT_numerical_val * 1.001 : # Allow slight overshoot due to numerics
                     return sol.root
                else:
                    self._log(f"Fallback root {sol.root} out of physical bounds [0, {CT_numerical_val}].")
                    return np.nan # Or 0.0 if that's preferred for non-convergence
            else:
                self._log(f"Fallback root_scalar did not converge. {sol.flag}")
                return np.nan
        except ValueError as ve: # Often from bracket not having a sign change for brentq
            self._log(f"ValueError in fallback root_scalar (likely bracket issue): {ve}")
            return np.nan
        except Exception as e_rs_fallback:
            self._log(f"Exception in fallback root_scalar: {e_rs_fallback}")
            return np.nan

    def get_concs(self, param_array, macro_array):
        # param_array from GlobalModel contains the log(K) values
        # macro_array from GlobalModel contains the numerical total concentrations
        
        # Convert log(K) to K, overwriting param_array
        try:
            # Exponentiate in place (or re-assign to the same name)
            param_array = np.exp(param_array) 
            if np.any(np.isinf(param_array)) or np.any(np.isnan(param_array)):
                self._log("Warning: Inf or NaN after exp(param_array). Clamping values.")
                param_array = np.nan_to_num(param_array, nan=1.0, posinf=1e30, neginf=1e-30) # Avoid 0 if K must be positive
        except Exception as e_exp:
            self._log(f"Error exponentiating param_array: {e_exp}. Using raw values (potential error).")
            # If exp fails, param_array still holds log_values. Ensure it's float.
            param_array = np.asarray(param_array, dtype=float) 


        # --- Create param_dict with numerical K values and numerical Total concs ---
        param_dict_for_free_c = {}
        # Now param_array holds K_values
        if len(self._constants) != len(param_array): 
            self._log(f"Warning: Mismatch K param length. Expected {len(self._constants)}, got {len(param_array)}")
            return np.full(len(self._micro_species), np.nan)
        for k_name, k_val in zip(self._constants, param_array): # Use param_array directly
            param_dict_for_free_c[k_name] = k_val

        # Use macro_array directly
        if len(self._macro_species) != len(macro_array): 
            self._log(f"Warning: Mismatch macro_array length. Expected {len(self._macro_species)}, got {len(macro_array)}")
            return np.full(len(self._micro_species), np.nan)
        for m_name, m_val in zip(self._macro_species, macro_array): # Use macro_array directly
            param_dict_for_free_c[m_name] = m_val
        
        # --- Get free C ---
        C_free_val = self._get_free_c(**param_dict_for_free_c)
        if np.isnan(C_free_val):
            self._log("Failed to determine free C. Returning NaNs for all species.")
            return np.full(len(self._micro_species), np.nan)

        # --- Calculate all other species concentrations ---
        calculated_concs_dict = {self._c_species_name: C_free_val}

        # Calculate other base free species
        # self.solved_vars maps {A_sym: expr_for_A_in_terms_of_AT_C_Ks}
        for base_var_sym, expr_for_base_var in self.solved_vars.items():
            temp_expr = expr_for_base_var
            # Substitute K's (which are now in param_array)
            for k_name, k_val in zip(self._constants, param_array): 
                temp_expr = temp_expr.subs(self.symbols_dict[k_name], k_val)
            # Substitute Total concentrations (from macro_array)
            for m_name, m_val in zip(self._macro_species, macro_array):
                 if self.symbols_dict[m_name] in temp_expr.free_symbols:
                    temp_expr = temp_expr.subs(self.symbols_dict[m_name], m_val)
            # Substitute C_free_val
            temp_expr = temp_expr.subs(self._c_symbol, C_free_val)
            
            try:
                val = float(temp_expr)
                calculated_concs_dict[base_var_sym.name] = val if np.isfinite(val) else 0.0
            except Exception as e_basesolve:
                self._log(f"Error evaluating solved base var {base_var_sym.name}: {e_basesolve}. Expr: {temp_expr}")
                calculated_concs_dict[base_var_sym.name] = 0.0

        # Calculate complex species concentrations
        # self.simplified_eqs maps {Complex_sym: expr_for_Complex_in_terms_of_base_free_species_and_Ks}
        for complex_sym, expr_for_complex in self.simplified_eqs.items():
            temp_expr = expr_for_complex
            # Substitute K's (from param_array)
            for k_name, k_val in zip(self._constants, param_array):
                temp_expr = temp_expr.subs(self.symbols_dict[k_name], k_val)
            # Substitute already calculated free species concentrations (C_free, A_free, E_free etc.)
            for free_spec_name, free_spec_val in calculated_concs_dict.items():
                if self.symbols_dict[free_spec_name] in temp_expr.free_symbols:
                    temp_expr = temp_expr.subs(self.symbols_dict[free_spec_name], free_spec_val)
            
            try:
                val = float(temp_expr)
                calculated_concs_dict[complex_sym.name] = val if np.isfinite(val) else 0.0
            except Exception as e_complexsolve:
                self._log(f"Error evaluating complex {complex_sym.name}: {e_complexsolve}. Expr: {temp_expr}")
                calculated_concs_dict[complex_sym.name] = 0.0
        
        # Ensure all microspecies have a value
        for micro_name_str in self._micro_species:
            if micro_name_str not in calculated_concs_dict:
                if micro_name_str != self._c_species_name :
                     self._log(f"Warning: Micro-species {micro_name_str} not explicitly calculated. Defaulting to 0.")
                calculated_concs_dict.setdefault(micro_name_str, 0.0)

        final_concs_array = np.array([calculated_concs_dict.get(name, 0.0) for name in self._micro_species])
        
        if np.any(np.isnan(final_concs_array)) or np.any(np.isinf(final_concs_array)):
            self._log("NaN or Inf in final concentrations array. Clamping.")
            final_concs_array = np.nan_to_num(final_concs_array, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            self._concentrations_df = pd.concat([
                self._concentrations_df, 
                pd.DataFrame([calculated_concs_dict], columns=self._micro_species)
            ], ignore_index=True)
        except Exception: pass

        return final_concs_array

    @property
    def param_names(self):
        return np.array(self._constants)
    
    @property
    def macro_species(self):
        return np.array(self._macro_species)
    
    @property
    def micro_species(self):
        return np.array(self._micro_species)
        
    def _get_real_root(self, roots_complex, upper_bounds=[]):
        # Filter for real roots
        real_roots_mask = np.isreal(roots_complex)
        real_roots = np.real(roots_complex[real_roots_mask])

        if len(real_roots) == 0:
            self._log("No real roots found.")
            return np.nan

        # Filter for non-negative roots
        non_negative_mask = (real_roots >= -1e-14) # Allow for very small negative due to precision
        positive_roots = real_roots[non_negative_mask]
        positive_roots[positive_roots < 0] = 0 # Clamp tiny negatives to zero

        if len(positive_roots) == 0:
            self._log("No non-negative real roots found.")
            return np.nan

        # Filter by upper bounds
        valid_roots = positive_roots
        if upper_bounds:
            min_upper_bound = np.min(upper_bounds) # Should always be positive (e.g. CT)
            if min_upper_bound < 0: min_upper_bound = 0 # Safety for upper bound

            within_bounds_mask = (valid_roots <= min_upper_bound * 1.0001) # Allow slight overshoot
            valid_roots = valid_roots[within_bounds_mask]

        if len(valid_roots) == 0:
            self._log(f"No roots found within upper bounds (e.g., CT={min_upper_bound if upper_bounds else 'N/A'}). Positive roots found: {positive_roots}")
            # If no root is within bounds, but positive roots exist, maybe the smallest positive is best?
            # This can happen if CT is very small and all roots are slightly larger due to numerics.
            # Or if the model/params lead to no physical solution.
            # For now, strict: if none in bounds, return NaN.
            # Consider returning np.min(positive_roots) if it's "close" to bounds, or if this happens often.
            return np.nan
        
        if len(valid_roots) > 1:
            self._log(f"Multiple valid roots found: {valid_roots}. Returning the smallest positive one.")
            # Heuristic: often the smallest positive root is the physically relevant one.
            # This needs careful consideration based on the system.
            return np.min(valid_roots) 
            # Alternative: check which one best satisfies the original polynomial if there's doubt.
            # For now, min positive is a common choice.

        return valid_roots[0]
    
    @property
    def equilibria(self): return self._equilibria
    
    @property
    def species(self): return self._species
    
    @property
    def concentrations_df(self): return self._concentrations_df # Corrected property name
    
    @property
    def model_spec(self): return self._model_spec
    
    def set_debug(self, debug=True): self._debug = debug; return self
    
    def get_symbolic_equations(self):
        return {
            'equilibria_dict': self.equilibrium_eqs_dict, # Renamed for clarity
            'simplified_species_expressions': self.simplified_eqs, # Renamed
            'solved_base_variables': self.solved_vars, # Renamed
            'final_conservation_equation_for_C': self.final_ct_for_C_poly_extraction # Renamed
        }

    # (Original print_model_summary can be kept or adapted)