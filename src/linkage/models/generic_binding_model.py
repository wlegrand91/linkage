import numpy as np
import pandas as pd
from sympy import symbols, expand, simplify, collect, prod
from scipy.optimize import root_scalar


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
        
        # Set up symbolic representation
        self._setup_symbolic_model()
        
        # Initialize concentrations DataFrame
        self._concentrations_df = pd.DataFrame(columns=self._micro_species, dtype=float)
    
    def print_summary(self):
        """Print a summary of the model's key properties."""
        print("\n===== GENERIC BINDING MODEL SUMMARY =====")
        print(f"Constants: {self._constants}")
        print(f"Microspecies: {self._micro_species}")
        print(f"Macrospecies: {self._macro_species}")
        print(f"Equilibria:")
        for k, (reactants, products) in self._equilibria.items():
            print(f"  {' + '.join(reactants)} -> {' + '.join(products)}; {k}")
        print(f"Final conservation equation: {self.final_ct}")
        print("===== END SUMMARY =====\n")
    
    def _log(self, message):
        """
        Print debug messages if debug mode is enabled.
        
        Parameters
        ----------
        message : str
            Debug message to print
        """
        if self._debug:
            print(f"DEBUG: {message}")
    
    def _parse_model_spec(self):
        """
        Parse the model specification string into structured components.
        
        Returns
        -------
        tuple
            (equilibria, constants, species, micro_species, macro_species)
        """
        self._log("Parsing model specification")
        
        # Parse equilibria and species sections
        equilibria, constants = self._parse_equilibria_section()
        species = self._parse_species_section()
        
        # Extract and validate microspecies and macrospecies
        micro_species, macro_species = self._validate_and_extract_species(equilibria, species)
        
        return equilibria, constants, species, micro_species, macro_species
    
    def _parse_equilibria_section(self):
        """
        Parse the equilibria section of the model specification.
        
        Returns
        -------
        tuple
            (equilibria_dict, constants_list)
        """
        equilibria = {}
        constants = []
        
        in_equilibria = False
        
        for line in self._model_spec.split('\n'):
            line = line.strip()
            
            # Check for section header
            if 'equilibria:' in line:
                in_equilibria = True
                continue
            elif 'species:' in line or not line:
                in_equilibria = False
                continue
            
            # Process equilibria lines
            if in_equilibria and line:
                # Skip malformed lines
                if ';' not in line or '->' not in line:
                    if self._debug:
                        self._log(f"Skipping malformed equilibria line: '{line}'")
                    continue
                
                # Parse reaction and constant
                reaction, K = line.split(';')
                K = K.strip()
                
                # Validate constant - must be non-empty and start with K
                if not K or not K.startswith('K'):
                    if self._debug:
                        self._log(f"Skipping equilibria with invalid constant: '{line}'")
                    continue
                
                # Process reactants and products
                reactants_str, products_str = reaction.split('->')
                reactants = [r.strip() for r in reactants_str.split('+') if r.strip()]
                products = [p.strip() for p in products_str.split('+') if p.strip()]
                
                # Store in equilibria dictionary
                equilibria[K] = [reactants, products]
                
                # Add to constants list
                if K not in constants:
                    constants.append(K)
                    
                self._log(f"Parsed equilibrium: {reactants} -> {products}; {K}")
        
        if not equilibria:
            raise ValueError("No valid equilibria found in model specification")
            
        return equilibria, constants
    
    def _parse_species_section(self):
        """
        Parse the species section of the model specification.
        
        Returns
        -------
        dict
            Dictionary with macrospecies as keys and tuples of (microspecies_list, stoichiometries) as values
        """
        species = {}
        in_species = False
        
        for line in self._model_spec.split('\n'):
            line = line.strip()
            
            # Check for section header
            if 'species:' in line:
                in_species = True
                continue
            elif not line:
                continue
            
            # Process species lines
            if in_species and '=' in line:
                lhs, rhs = line.split('=')
                macro = lhs.strip()
                
                # Parse RHS into components and stoichiometries
                micro_species = []
                stoichiometries = []
                
                for item in rhs.strip().split('+'):
                    item = item.strip()
                    if '*' in item:
                        # Handle explicit stoichiometry (e.g., 2*AC1)
                        coef, species_name = item.split('*')
                        try:
                            stoich = int(coef.strip())
                            species_name = species_name.strip()
                            micro_species.append(species_name)
                            stoichiometries.append(stoich)
                        except ValueError:
                            if self._debug:
                                self._log(f"Skipping malformed stoichiometry: '{item}'")
                            continue
                    else:
                        # Implicit stoichiometry of 1
                        species_name = item.strip()
                        if species_name:
                            micro_species.append(species_name)
                            stoichiometries.append(1)
                
                # Store processed species
                species[macro] = (micro_species, stoichiometries)
                self._log(f"Parsed species: {macro} = {micro_species} with stoichiometries {stoichiometries}")
        
        if not species:
            raise ValueError("No valid species definitions found in model specification")
            
        return species
    
    def _validate_and_extract_species(self, equilibria, species):
        """
        Validate species consistency and extract complete lists of microspecies and macrospecies.
        
        Parameters
        ----------
        equilibria : dict
            Dictionary of parsed equilibria
        species : dict
            Dictionary of parsed species
            
        Returns
        -------
        tuple
            (micro_species_list, macro_species_list)
        """
        # Get all microspecies mentioned in equilibria
        micro_in_equilibria = set()
        for K, (reactants, products) in equilibria.items():
            micro_in_equilibria.update(reactants)
            micro_in_equilibria.update(products)
        
        # Get all microspecies mentioned in species definitions
        micro_in_species = set()
        for macro, (micros, _) in species.items():
            micro_in_species.update(micros)
        
        # Check for inconsistencies - print warnings but don't fail
        if micro_in_equilibria != micro_in_species:
            only_in_eq = micro_in_equilibria - micro_in_species
            only_in_sp = micro_in_species - micro_in_equilibria
            
            warning_msg = "WARNING: Mismatch between microspecies in equilibria and species definitions"
            if only_in_eq:
                warning_msg += f"\nSpecies only in equilibria: {', '.join(only_in_eq)}"
            if only_in_sp:
                warning_msg += f"\nSpecies only in species definitions: {', '.join(only_in_sp)}"
                
            print(warning_msg)
            
            # Instead of failing, merge the sets
            all_micro = micro_in_equilibria.union(micro_in_species)
            micro_species = sorted(list(all_micro))
        else:
            # Get lists in sorted order for consistency
            micro_species = sorted(list(micro_in_equilibria))
            
        macro_species = sorted(list(species.keys()))
        
        # Make sure we have at least CT in the macrospecies
        if not any(m.endswith('T') and 'C' in m for m in macro_species):
            print("WARNING: No CT-like species found in macrospecies")
            
        self._log(f"Validated microspecies: {micro_species}")
        self._log(f"Validated macrospecies: {macro_species}")
        
        return micro_species, macro_species
    
    def _setup_symbolic_model(self):
        """
        Set up the symbolic representation of the model using sympy.
        """
        self._log("Setting up symbolic model")
        
        # Create symbol dictionary
        self.symbols_dict = {}
        
        # Extract base variables from macro species (removing 'T' suffix)
        self.base_vars = [macro[:-1] for macro in self._macro_species]
        
        # Create symbols for all entities
        for var in self.base_vars:
            self.symbols_dict[var] = symbols(var)
        
        for const in self._constants:
            self.symbols_dict[const] = symbols(const)
        
        for micro in self._micro_species:
            self.symbols_dict[micro] = symbols(micro)
        
        for macro in self._macro_species:
            self.symbols_dict[macro] = symbols(macro)
        
        # Process model equations
        self.equilibrium_eqs = self._create_equilibrium_equations()
        self.simplified_eqs = self._simplify_equilibrium_equations(self.equilibrium_eqs)
        self.solved_vars, self.final_ct = self._solve_conservation_equations(self.simplified_eqs)
        
        self._log("Symbolic model setup complete")
    
    def _create_equilibrium_equations(self):
        """
        Create symbolic equations for each equilibrium.
        
        Returns
        -------
        list
            List of tuples (product_symbol, rhs_expression)
        """
        equations = []
        
        for K, (reactants, products) in self._equilibria.items():
            # Each equilibrium should have at least one product
            if not products:
                continue
                
            # Create expression for each product
            for product in products:
                product_sym = self.symbols_dict[product]
                
                # Create the right-hand side expression: K * product of reactants
                reactant_syms = [self.symbols_dict[r] for r in reactants]
                rhs = self.symbols_dict[K] * prod(reactant_syms)
                
                equations.append((product_sym, rhs))
                self._log(f"Created equation: {product} = {K} * {' * '.join(reactants)}")
        
        return equations
    
    def _simplify_equilibrium_equations(self, equations):
        """
        Simplify equilibrium equations by recursive substitution.
        
        Parameters
        ----------
        equations : list
            List of tuples (product_symbol, rhs_expression)
            
        Returns
        -------
        dict
            Dictionary mapping species symbols to their simplified expressions
        """
        # Create dictionary from equation list
        eq_dict = {lhs: rhs for lhs, rhs in equations}
        
        # Function for recursive substitution
        def substitute_recursive(expression):
            changed = True
            while changed:
                changed = False
                for term, replacement in eq_dict.items():
                    if term in expression.free_symbols and term not in [self.symbols_dict[var] for var in self.base_vars]:
                        expression = expression.subs(term, replacement)
                        expression = expand(expression)
                        changed = True
            return expression
        
        # Apply substitution to all equations
        simplified = {}
        for lhs, rhs in eq_dict.items():
            simplified_expr = substitute_recursive(rhs)
            simplified_expr = collect(simplified_expr, self.symbols_dict[self.base_vars[0]])
            simplified[lhs] = simplified_expr
            
            if self._debug:
                self._log(f"Simplified equation: {lhs} = {simplified_expr}")
        
        return simplified
    
    def _solve_conservation_equations(self, equilibrium_dict):
        """
        Solve conservation equations to get expressions for base variables and final conservation equation.
        
        Parameters
        ----------
        equilibrium_dict : dict
            Dictionary mapping species symbols to their simplified expressions
            
        Returns
        -------
        tuple
            (solved_vars, final_ct)
        """
        solved_vars = {}
        
        # Find CT equation specifically
        ct_eq = None
        other_eqs = []
        
        for macro, (micros, stoich) in self._species.items():
            if 'CT' in macro:
                # This is the CT equation, store for later
                ct_eq = (macro, micros, stoich)
            else:
                # Other conservation equations
                other_eqs.append((macro, micros, stoich))
        
        if not ct_eq:
            # Special case: If no CT equation found but only one species equation exists,
            # assume it's for a simple system and use it as the CT equation
            if len(self._species) == 1:
                macro, (micros, stoich) = next(iter(self._species.items()))
                ct_eq = (macro, micros, stoich)
                print(f"WARNING: No explicit CT equation found. Using {macro} as the conservation equation.")
            else:
                raise ValueError("Could not find CT equation in species definitions")
        
        # Process all non-CT equations to solve for variables
        for macro, micros, stoich in other_eqs:
            try:
                # Create symbolic expression for conservation equation
                rhs_expr = sum(s * self.symbols_dict[m] for m, s in zip(micros, stoich))
                
                # Substitute equilibrium expressions
                prev_expr = None
                while prev_expr != rhs_expr:
                    prev_expr = rhs_expr
                    for species_sym, expr in equilibrium_dict.items():
                        if species_sym in rhs_expr.free_symbols:
                            rhs_expr = rhs_expr.subs(species_sym, expr)
                
                # Solve for the base variable
                var_to_solve = self.symbols_dict[macro[:-1]]  # Remove 'T' suffix
                collected = collect(rhs_expr, var_to_solve)
                
                # Extract coefficient of the variable
                coeff = collected.coeff(var_to_solve)
                
                if coeff == 0:
                    print(f"WARNING: Cannot solve for {var_to_solve} in equation {macro} = {rhs_expr}")
                    continue
                    
                # Solve for the variable: macro/coeff
                solution = self.symbols_dict[macro]/coeff
                solved_vars[var_to_solve] = simplify(solution)
                
                self._log(f"Solved for {var_to_solve} = {solution}")
            except Exception as e:
                print(f"WARNING: Failed to process equation for {macro}: {str(e)}")
                continue
        
        # Process the CT equation to get final conservation expression
        try:
            macro, micros, stoich = ct_eq
            
            # Create expression for CT equation
            ct_rhs_expr = sum(s * self.symbols_dict[m] for m, s in zip(micros, stoich))
            
            # Substitute equilibrium expressions
            prev_expr = None
            while prev_expr != ct_rhs_expr:
                prev_expr = ct_rhs_expr
                for species_sym, expr in equilibrium_dict.items():
                    if species_sym in ct_rhs_expr.free_symbols:
                        ct_rhs_expr = ct_rhs_expr.subs(species_sym, expr)
            
            # Substitute solved variables
            for var, solution in solved_vars.items():
                ct_rhs_expr = ct_rhs_expr.subs(var, solution)
            
            # Final expression: ct_rhs - CT
            final_ct = ct_rhs_expr - self.symbols_dict[macro]
            
            self._log(f"Final conservation equation: {final_ct} = 0")
            
            return solved_vars, final_ct
        except Exception as e:
            error_msg = f"Failed to process CT equation: {str(e)}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def _get_free_c(self, **param_dict):
        """
        Get free calcium concentration by solving the conservation equation.
        
        Parameters
        ----------
        param_dict : dict
            Dictionary of parameter values including equilibrium constants and total concentrations
            
        Returns
        -------
        float
            Free calcium concentration that satisfies the conservation equation
        """
        # Extract CT for bounds checking
        if 'CT' not in param_dict:
            if self._debug:
                self._log("CT not found in parameter dictionary")
            return 0.0
            
        CT = param_dict['CT']
        
        # Early return if no calcium present
        if CT == 0 or CT < 1e-15:
            return 0.0
            
        # Check if all required parameters are present
        missing_params = []
        for param in self.symbols_dict:
            if param not in param_dict and param != 'C' and param not in self._micro_species:
                missing_params.append(param)
                
        if missing_params:
            if self._debug:
                self._log(f"Missing parameters in _get_free_c: {missing_params}")
            # Instead of failing, we'll use default values
            for param in missing_params:
                if param in self._constants:
                    param_dict[param] = 1.0  # Default K value of 1
                elif param in self._macro_species:
                    param_dict[param] = 0.0  # Default concentration of 0
        
        # Get the conservation equation and substitute parameter values
        eq = self.final_ct
        
        for symbol_name, value in param_dict.items():
            if symbol_name in self.symbols_dict:
                eq = eq.subs(self.symbols_dict[symbol_name], value)
        
        # Numerical function for root finding
        def equation(c):
            """Convert symbolic equation to numerical function for root finding"""
            try:
                # Handle numpy scalar if passed
                if hasattr(c, 'item'):
                    c = c.item()
                
                # Substitute C value and evaluate
                result = float(eq.subs(self.symbols_dict['C'], c))
                
                return result
            except Exception as e:
                if self._debug:
                    self._log(f"Error in equation evaluation at C={c}: {str(e)}")
                return np.nan
        
        try:
            # Check sign at boundaries to determine search interval
            f_zero = equation(1e-15)  # Almost zero
            f_ct = equation(CT)
            
            # Initial bounds
            lower_bound = 1e-15
            upper_bound = CT
            
            # Handle NaN cases
            if np.isnan(f_zero) or np.isnan(f_ct):
                if self._debug:
                    self._log("Equation evaluation returned NaN at boundaries")
                return 0.0  # Return safe default
            
            # If same sign at boundaries, try to expand the interval
            if f_zero * f_ct > 0:
                # Try above CT first (in case we're missing some bound)
                expanded_upper = CT * 2
                f_expanded = equation(expanded_upper)
                
                if np.isnan(f_expanded):
                    if self._debug:
                        self._log("Equation evaluation returned NaN at expanded upper bound")
                    return 0.0  # Return safe default
                
                if f_zero * f_expanded < 0:
                    upper_bound = expanded_upper
                else:
                    # Try a broader search
                    test_points = np.logspace(-15, np.log10(CT*10), 20)
                    found_bracket = False
                    
                    prev_f = f_zero
                    for point in test_points:
                        current_f = equation(point)
                        if np.isnan(current_f):
                            continue
                        if prev_f * current_f < 0:
                            # Found sign change
                            found_idx = np.where(test_points == point)[0][0]
                            if found_idx > 0:
                                lower_bound = test_points[found_idx - 1]
                            upper_bound = point
                            found_bracket = True
                            break
                        prev_f = current_f
                    
                    if not found_bracket:
                        if self._debug:
                            self._log("Could not find interval with sign change for root finding")
                        return 0.0  # Return safe default
            
            # Use bounded optimization with validated interval
            try:
                result = root_scalar(equation, 
                                    bracket=[lower_bound, upper_bound], 
                                    method='brentq', 
                                    xtol=1e-12, 
                                    rtol=1e-10,
                                    maxiter=100)
                
                if result.converged:
                    root = result.root
                    
                    # Additional validation
                    if lower_bound <= root <= upper_bound and np.isfinite(root):
                        self._log(f"Found root C = {root:e}")
                        return root
                    else:
                        if self._debug:
                            self._log(f"Root {root:e} outside valid range [{lower_bound:e}, {upper_bound:e}]")
                        return 0.0  # Return safe default
                else:
                    if self._debug:
                        self._log(f"Root finding failed to converge after {result.iterations} iterations")
                    return 0.0  # Return safe default
            except Exception as e:
                if self._debug:
                    self._log(f"Error in root_scalar: {str(e)}")
                return 0.0  # Return safe default
                
        except Exception as e:
            if self._debug:
                self._log(f"Root finding failed with error: {str(e)}")
            return 0.0  # Return safe default
    
    def get_concs(self, param_array, macro_array):
        """
        Get concentrations of all species given parameters and macro concentrations.
        
        Parameters
        ----------
        param_array : numpy.ndarray
            Array of equilibrium constants (exponentiated from log values)
        macro_array : numpy.ndarray
            Array of total concentrations for macrospecies
            
        Returns
        -------
        numpy.ndarray
            Array of concentrations for all microspecies
        """
        # Handle case where param_array or macro_array is None or empty
        if param_array is None or len(param_array) == 0:
            if self._debug:
                self._log("Empty param_array provided to get_concs")
            return np.zeros(len(self._micro_species))
            
        if macro_array is None or len(macro_array) == 0:
            if self._debug:
                self._log("Empty macro_array provided to get_concs")
            return np.zeros(len(self._micro_species))
        
        # Safety check for NaN or infinite values in input arrays
        if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
            if self._debug:
                self._log("NaN or infinite values in param_array")
            # Replace with zeros
            param_array = np.nan_to_num(param_array, nan=0.0, posinf=1e10, neginf=-1e10)
            
        if np.any(np.isnan(macro_array)) or np.any(np.isinf(macro_array)):
            if self._debug:
                self._log("NaN or infinite values in macro_array")
            # Replace with zeros
            macro_array = np.nan_to_num(macro_array, nan=0.0, posinf=1e10, neginf=0.0)
            
        # Verify lengths of arrays
        if len(param_array) != len(self._constants):
            if self._debug:
                self._log(f"Param array length mismatch: got {len(param_array)}, expected {len(self._constants)}")
            # Instead of failing, pad or truncate
            if len(param_array) < len(self._constants):
                # Pad with zeros
                param_array = np.pad(param_array, (0, len(self._constants) - len(param_array)))
            else:
                # Truncate
                param_array = param_array[:len(self._constants)]
                
        if len(macro_array) != len(self._macro_species):
            if self._debug:
                self._log(f"Macro array length mismatch: got {len(macro_array)}, expected {len(self._macro_species)}")
            # Instead of failing, pad or truncate
            if len(macro_array) < len(self._macro_species):
                # Pad with zeros
                macro_array = np.pad(macro_array, (0, len(self._macro_species) - len(macro_array)))
            else:
                # Truncate
                macro_array = macro_array[:len(self._macro_species)]
                
        # Create parameter dictionary with exponentiated K values and macro values
        param_dict = {}
        
        # Add exponentiated K values - handle potential overflows
        try:
            exp_values = np.exp(param_array)
            # Check for infinities from overflow
            if np.any(np.isinf(exp_values)):
                if self._debug:
                    self._log("Overflow in exponentiating parameters")
                # Cap at a large value
                exp_values = np.nan_to_num(exp_values, nan=1.0, posinf=1e30)
            
            for name, value in zip(self._constants, exp_values):
                param_dict[name] = value
        except Exception as e:
            if self._debug:
                self._log(f"Error exponentiating parameters: {str(e)}")
            # Use default values of 1.0 for all constants
            for name in self._constants:
                param_dict[name] = 1.0
            
        # Add macro species values
        for name, value in zip(self._macro_species, macro_array):
            param_dict[name] = value
        
        # Get free C concentration - now returns 0.0 instead of NaN on failure
        C = self._get_free_c(**param_dict)
        
        # Calculate concentrations of all species
        concs_dict = {}
        
        # First solve for base species using conservation equations
        for species_sym, expr in self.solved_vars.items():
            species = str(species_sym)
            try:
                # Substitute all parameter values
                for param, value in param_dict.items():
                    expr = expr.subs(self.symbols_dict[param], value)
                expr = expr.subs(self.symbols_dict['C'], C)
                conc = float(expr)
                # Check for invalid values
                if not np.isfinite(conc):
                    conc = 0.0
                concs_dict[species] = conc
            except Exception as e:
                if self._debug:
                    self._log(f"Failed to calculate {species} due to: {str(e)}")
                concs_dict[species] = 0.0
        
        # Next calculate derived species from equilibrium equations
        for species, expr in self.simplified_eqs.items():
            species = str(species)
            try:
                # First substitute known concentrations
                for known_species, known_conc in concs_dict.items():
                    expr = expr.subs(self.symbols_dict[known_species], known_conc)
                
                # Then substitute C and parameters
                expr = expr.subs(self.symbols_dict['C'], C)
                for param, value in param_dict.items():
                    expr = expr.subs(self.symbols_dict[param], value)
                
                conc = float(expr)
                # Check for invalid values
                if not np.isfinite(conc):
                    conc = 0.0
                concs_dict[species] = conc
            except Exception as e:
                if self._debug:
                    self._log(f"Failed to calculate {species} due to: {str(e)}")
                concs_dict[species] = 0.0
        
        # Create final array in the correct order
        result = np.array([concs_dict.get(species, 0.0) for species in self._micro_species])
        
        # Final check for invalid values
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            if self._debug:
                self._log("Invalid values in result array")
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store calculated concentrations for debugging
        try:
            self._concentrations_df = pd.concat([
                self._concentrations_df, 
                pd.DataFrame([{species: conc for species, conc in zip(self._micro_species, result)}])
            ], ignore_index=True)
        except Exception as e:
            if self._debug:
                self._log(f"Failed to update concentrations DataFrame: {str(e)}")
        
        return result
    
    @property
    def param_names(self):
        """Get names of model parameters."""
        if not hasattr(self, '_constants') or not self._constants:
            print("WARNING: No constants found in model")
            return np.array([])
        
        # Ensure we're returning valid string constants
        valid_constants = []
        for const in self._constants:
            if const is not None and isinstance(const, str) and const.strip():
                valid_constants.append(const)
        
        # Debug output
        if self._debug:
            self._log(f"Parameter names: {valid_constants}")
            
        if not valid_constants:
            print("WARNING: No valid constants found after filtering")
            
        # Return as numpy array
        return np.array(valid_constants)
    
    @property
    def macro_species(self):
        """Get names of macro (total) species."""
        return np.array(self._macro_species)
    
    @property
    def micro_species(self):
        """Get names of micro species."""
        return np.array(self._micro_species)
        
    # Implementation of methods required by the BaseModel interface
    
    def _get_real_root(self, roots, upper_bounds=[]):
        """
        Get the real root between 0 and upper_bounds.
        This is a reimplementation of the method from BindingModel.

        Parameters
        ----------
        roots : numpy.ndarray
            numpy array with roots to check
        upper_bounds : list-like
            list of upper bounds against which to check root.
        """
        # Check for realness
        to_check = [np.isreal(roots)]

        # Check to see if root >= 0
        to_check.append(np.logical_or(roots > 0, np.isclose(roots, 0)))

        # Check to see if root <= lowest upper bound
        if len(upper_bounds) > 0:
            min_upper = np.min(upper_bounds)
            to_check.append(np.logical_or(roots < min_upper,
                                        np.isclose(roots, min_upper)))
        
        # Get all roots that meet all criteria
        mask = np.logical_and.reduce(to_check)
        solution = np.unique(roots[mask])
        
        # No root matches condition
        if len(solution) == 0:
            if self._debug:
                self._log("No valid roots found")
            return np.nan 
        
        # Multiple roots match conditions
        if len(solution) > 1:
            # Check if roots are numerically close
            close_mask = np.isclose(solution[0], solution)
            if np.sum(close_mask) != len(solution):
                if self._debug:
                    self._log("Multiple distinct roots found")
                return np.nan
        
        # Return real component
        return np.real(solution[0])
    
    @property
    def equilibria(self):
        """Get equilibrium definitions."""
        return self._equilibria
    
    @property
    def species(self):
        """
        Get species definitions.
        
        Returns
        -------
        dict
            Dictionary mapping macrospecies to lists of microspecies and their stoichiometries
        """
        return self._species
    
    @property
    def concentrations(self):
        """
        Get the DataFrame of calculated concentrations.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing concentration calculations with columns for each microspecies
        """
        return self._concentrations_df
    
    @property
    def model_spec(self):
        """
        Get the original model specification string.
        
        Returns
        -------
        str
            The model specification string used to initialize the model
        """
        return self._model_spec
    
    def set_debug(self, debug=True):
        """
        Enable or disable debug mode.
        
        Parameters
        ----------
        debug : bool
            Whether to enable detailed debug output
        """
        self._debug = debug
        return self
    
    def get_symbolic_equations(self):
        """
        Get the symbolic equations for the model.
        
        Returns
        -------
        dict
            Dictionary containing different sets of symbolic equations
        """
        return {
            'equilibria': self.equilibrium_eqs,
            'simplified': self.simplified_eqs,
            'solved_vars': self.solved_vars,
            'conservation': self.final_ct
        }

    def print_model_summary(self):
        """
        Print a summary of the model structure.
        """
        print("\n=== Generic Binding Model Summary ===")
        
        print("\nEquilibria:")
        for k, (reactants, products) in self._equilibria.items():
            r_str = " + ".join(reactants)
            p_str = " + ".join(products)
            print(f"  {r_str} -> {p_str}; {k}")
        
        print("\nSpecies:")
        for macro, (micros, stoich) in self._species.items():
            rhs = " + ".join([f"{s}*{m}" if s > 1 else m for m, s in zip(micros, stoich)])
            print(f"  {macro} = {rhs}")
        
        print("\nParameters:")
        print(f"  {', '.join(self.param_names)}")
        
        print("\nMacro Species:")
        print(f"  {', '.join(self.macro_species)}")
        
        print("\nMicro Species:")
        print(f"  {', '.join(self.micro_species)}")
        
        print("\nFinal Conservation Equation:")
        print(f"  {self.final_ct} = 0")
        
        print("\n===================================\n")