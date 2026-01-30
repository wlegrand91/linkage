import numpy as np
from sympy import symbols, expand, simplify, collect, prod, Poly, fraction, sympify

class BindingPolynomial:
    """
    A class to derive a binding polynomial from a set of chemical equilibria and mass balance equations.

    This tool takes a model specification as a string, parses it, and uses symbolic mathematics
    to derive the polynomial equation for a specific free species (conventionally 'C').
    """

    def __init__(self, model_spec, debug=False):
        """
        Initializes the BindingPolynomial object.

        Parameters
        ----------
        model_spec : str
            A string containing the formatted chemical equilibria and mass balance equations.
        debug : bool, optional
            If True, prints detailed debugging information during processing.
        """
        if not model_spec:
            raise ValueError("The model specification is empty.")

        self._model_spec = model_spec
        self._debug = debug

        # These attributes are defined in _parse_model_spec but initialized here for clarity
        self.reparam_rules = {}
        self.new_fitting_vars = []
        
        self._equilibria, self._constants, self._species, self._micro_species, self._macro_species = self._parse_model_spec()

        self._c_species_name, self._ct_macrospecies_name = self._detect_polynomial_species()
        
        self._setup_symbolic_model()
    
    def _log(self, message):
        if self._debug:
            print(message)

    def _detect_polynomial_species(self):
        """
        Determines the free species and total species for the binding polynomial
        based on the last defined mass balance equation.
        """
        if not self._macro_species:
            # Fallback for specs without species block (legacy/testing only?)
            # Or assume 'C' if present in microspecies
            if "C" in self._micro_species:
                print("Warning: No species block found. Defaulting to 'C' and 'CT' (inferred).")
                return "C", "CT"
            raise ValueError("No species/mass-balance block found. Cannot determine polynomial variable.")

        # 1. Get the LAST defined macro species (User Rule)
        # self._macro_species is sorted alphabetically in parser!
        # Parse returns 'species' dict. Dict iteration order is insertion order in Python 3.7+
        # But _parse_model_spec receives 'species' dict from _parse_species_section().
        # _parse_species_section iterates lines. So 'species' dict IS ordered by definition in file.
        
        # However, _parse_model_spec returns:
        # micro_species, macro_species = self._validate_and_extract_species(...)
        # And _validate_and_extract_species SORTS macro_species!
        # self._macro_species is sorted! We lost the file order!
        
        # I must recover the order or access the 'species' dict directly from self._species (which is conserved?)
        # self._species was assigned in __init__ from _parse_model_spec return values?
        # Yes: self._species is the dict.
        
        last_macro = list(self._species.keys())[-1]
        
        # 2. Identify the free species within this Total
        # Candidates: microspecies in the mass balance equation
        micros_in_total, _ = self._species[last_macro]
        
        # 3. Filter out species that are Products of equilibria
        # (A species is a "Base/Free" species if it is not defined as a product in the reaction list)
        # Note: Some definitions might be reversible? "A + B <-> C" -> C is product.
        all_products = set()
        for _, products in self._equilibria.values():
            all_products.update(products)
            
        candidates = [m for m in micros_in_total if m not in all_products]
        
        if len(candidates) == 0:
            # Maybe it's a cyclic system or strange definition?
            # Or simple "P -> P*" and both P and P* are technically products of something else?
            # Fallback: Try to match Name (CT -> C)
            for m in micros_in_total:
                if m.upper() + "T" == last_macro.upper():
                    return m, last_macro
            raise ValueError(f"Could not identify a free species in the final mass balance: {last_macro} = ... All components act as products elsewhere.")
        
        if len(candidates) == 1:
            return candidates[0], last_macro
        
        # Multiple candidates? (e.g. C + E -> EC, and CT = C + EC + E/1000??)
        # Pick one that matches name convention
        for m in candidates:
            if m.upper() + "T" == last_macro.upper():
                return m, last_macro
        
        # If ambiguous, pick the first one? Or "C" if present?
        if "C" in candidates:
            return "C", last_macro
            
        # Last resort: just pick the first detected free species
        return candidates[0], last_macro

    def _parse_model_spec(self):
        self._log("\nParsing model specification...")
        equilibria, constants = self._parse_equilibria_section()
        species = self._parse_species_section()

        # Parse the reparameterize section to find all rules and new variables.
        self.reparam_rules, self.new_fitting_vars = self._parse_reparameterize_section(constants)

        # Update the list of constants to reflect the reparameterization.
        if self.reparam_rules:
            self._log(f"Reparameterization rules found. Updating fittable constants.")
            dependent_vars = [s.name for s in self.reparam_rules.keys()]
            
            # Distinguish new variables for the polynomial from other types (like dH).
            # This ensures only relevant parameters are included in self._constants.
            new_poly_vars = [v for v in self.new_fitting_vars if not v.startswith("dH_")]
            
            # The new constants for the polynomial are the old ones, minus the dependent ones,
            # plus any new non-enthalpy fitting variables.
            constants = sorted([c for c in constants if c not in dependent_vars] + new_poly_vars)
            self._log(f"New polynomial constants: {constants}")

        micro_species, macro_species = self._validate_and_extract_species(equilibria, species)
        self._log("Finished parsing model specification.")
        return equilibria, constants, species, micro_species, macro_species

    def _parse_reparameterize_section(self, existing_constants):
        """
        Parses the 'reparameterize:' section of the model spec. This section
        defines algebraic relationships between parameters.
        """
        reparam_rules = {}
        new_fitting_vars = set()
        in_reparam = False

        # Create a dictionary of symbols for all known constants to parse expressions
        known_symbols = {c: symbols(c) for c in existing_constants}

        for line in self._model_spec.split('\n'):
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            
            # Use section headers to switch the parser's context
            if 'reparameterize:' in line: in_reparam = True; continue
            if 'equilibria:' in line or 'species:' in line: in_reparam = False
            
            if in_reparam and '=' in line:
                dependent_str, expr_str = line.split('=', 1)
                dependent_sym = symbols(dependent_str.strip())

                # Safely parse the string into a SymPy expression
                try:
                    # sympify can convert a string to a symbolic expression, using
                    # a dictionary of known local variables.
                    expr = sympify(expr_str.strip(), locals=known_symbols)
                except Exception as e:
                    raise ValueError(f"Could not parse reparameterization expression: '{line}'. Error: {e}")

                reparam_rules[dependent_sym] = expr
                
                # Identify any new symbols in the expression that are not
                # existing constants. These are the new independent variables.
                for sym in expr.free_symbols:
                    if sym.name not in existing_constants:
                        new_fitting_vars.add(sym.name)
                        # Add the new symbol to our dict for parsing subsequent lines
                        known_symbols[sym.name] = sym
                        
        return reparam_rules, sorted(list(new_fitting_vars))

    def _parse_equilibria_section(self):
        equilibria, constants = {}, []
        in_equilibria = False
        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'equilibria:' in line: in_equilibria = True; continue
            if 'species:' in line or 'reparameterize:' in line: in_equilibria = False
            if in_equilibria and '->' in line and ';' in line:
                reaction, K = line.split(';'); K = K.strip()
                reactants_str, products_str = reaction.split('->')
                reactants = [r.strip() for r in reactants_str.split('+') if r.strip()]
                products = [p.strip() for p in products_str.split('+') if p.strip()]
                equilibria[K] = [reactants, products]
                if K not in constants: constants.append(K)
        return equilibria, sorted(constants)

    def _parse_species_section(self):
        species = {}
        in_species = False
        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'species:' in line: in_species = True; continue
            if 'equilibria:' in line or 'reparameterize:' in line: in_species = False
            if in_species and '=' in line:
                macro, micro_terms = line.split('=', 1)
                macro = macro.strip()
                
                micro_species_list = []
                stoichiometries = []
                for item in micro_terms.strip().split('+'):
                    item = item.strip()
                    if '*' in item:
                        coef_str, species_name = item.split('*', 1)
                        micro_species_list.append(species_name.strip())
                        stoichiometries.append(int(coef_str.strip()))
                    else:
                        micro_species_list.append(item)
                        stoichiometries.append(1)
                species[macro] = (micro_species_list, stoichiometries)
        return species

    def _validate_and_extract_species(self, equilibria, species):
        micro_in_equilibria = set(r for _, (reactants, products) in equilibria.items() for r in reactants + products)
        micro_in_species = set(m for micros, _ in species.values() for m in micros)
        all_micro_species = sorted(list(micro_in_equilibria.union(micro_in_species)))
        macro_species = sorted(list(species.keys()))
        return all_micro_species, macro_species


    def _setup_symbolic_model(self):
        self._log("\nSetting up symbolic model...")
        all_symbol_names = self._micro_species + self._macro_species + self._constants
        self.symbols = {name: symbols(name) for name in all_symbol_names}
        
        self._c_symbol = self.symbols[self._c_species_name]
        self.equilibrium_eqs = self._create_equilibrium_equations()
        self.simplified_eqs = self._simplify_equilibria(self.equilibrium_eqs)
        
        # If reparameterization rules exist, apply them now.
        if self.reparam_rules:
            self._log("Applying reparameterization substitutions to symbolic model.")
            # Filter for rules relevant to the polynomial (i.e., not dH rules)
            poly_rules = {s: e for s, e in self.reparam_rules.items() if not s.name.startswith("dH_")}
            
            substituted_eqs = {}
            for product_sym, rhs_expr in self.simplified_eqs.items():
                substituted_eqs[product_sym] = rhs_expr.subs(poly_rules)
            self.simplified_eqs = substituted_eqs

        self.solved_vars, self.final_rational_eq = self._solve_conservation_equations(self.simplified_eqs)
        self.binding_polynomial = self._create_polynomial_from_rational(self.final_rational_eq)
        self._log("Symbolic model setup complete.")

    def _create_equilibrium_equations(self):
        eqs = {}
        # Get all original constants from the equilibria block to create symbols
        all_constants_from_eq = list(self._equilibria.keys())
        all_symbols_to_create = self._micro_species + all_constants_from_eq
        temp_symbols = {name: symbols(name) for name in all_symbols_to_create}
        
        for K, (reactants, products) in self._equilibria.items():
            for product in products:
                reactant_syms = [temp_symbols[r] for r in reactants]
                eqs[temp_symbols[product]] = temp_symbols[K] * prod(reactant_syms)
        return eqs

    def _simplify_equilibria(self, equilibrium_eqs):
        simplified = {}
        for product_sym, rhs_expr in equilibrium_eqs.items():
            expr = rhs_expr
            # Iteratively substitute until no more changes can be made
            for _ in range(len(equilibrium_eqs) + 1):
                made_change = False
                # Use a temporary copy of the expression to check for symbols
                temp_expr = expr
                for sub_sym, sub_expr in equilibrium_eqs.items():
                    if sub_sym in temp_expr.free_symbols:
                        expr = expr.subs(sub_sym, sub_expr)
                        made_change = True
                if not made_change:
                    break
            simplified[product_sym] = expand(expr)
        return simplified

    def _solve_conservation_equations(self, simplified_eqs):
        solved_vars = {}
        base_vars_to_solve = {self.symbols.get(macro[:-1]) for macro in self._macro_species if macro.endswith('T') and macro != self._ct_macrospecies_name and macro[:-1] in self.symbols}
        
        for total_macro_name, (micro_list, stoich_list) in self._species.items():
            base_var_sym = self.symbols.get(total_macro_name[:-1])
            if base_var_sym in base_vars_to_solve:
                rhs_sum = 0
                for micro_str, stoich_val in zip(micro_list, stoich_list):
                    micro_sym = self.symbols[micro_str]
                    expr_for_micro = simplified_eqs.get(micro_sym, micro_sym)
                    rhs_sum += stoich_val * expr_for_micro
                
                for solved_sym, solved_expr in solved_vars.items():
                    if solved_sym in rhs_sum.free_symbols:
                        rhs_sum = rhs_sum.subs(solved_sym, solved_expr)

                rhs_sum = expand(rhs_sum)
                # Solve for base_var_sym: Total = base*coeff + terms_without
                collected = collect(rhs_sum, base_var_sym)
                coeff = collected.coeff(base_var_sym, 1)
                terms_without = collected.coeff(base_var_sym, 0)

                if coeff != 0:
                    total_sym = self.symbols[total_macro_name]
                    solved_vars[base_var_sym] = simplify((total_sym - terms_without) / coeff)
        
        ct_micros, ct_stoichs = self._species[self._ct_macrospecies_name]
        final_ct_rhs = 0
        for micro, stoich in zip(ct_micros, ct_stoichs):
            micro_sym = self.symbols[micro]
            expr_for_micro = simplified_eqs.get(micro_sym, micro_sym)
            final_ct_rhs += stoich * expr_for_micro
        
        final_ct_rhs = expand(final_ct_rhs)
        
        for base_var, expr in solved_vars.items():
            if base_var in final_ct_rhs.free_symbols:
                final_ct_rhs = final_ct_rhs.subs(base_var, expr)
        
        final_rational = simplify(final_ct_rhs - self.symbols[self._ct_macrospecies_name])
        return solved_vars, final_rational

    def _create_polynomial_from_rational(self, rational_eq):
        numer, denom = fraction(rational_eq, self._c_symbol)
        polynomial_eq = expand(numer)
        return polynomial_eq

    def get_polynomial_coefficients(self):
        if not hasattr(self, 'binding_polynomial'):
            raise RuntimeError("Binding polynomial has not been generated.")
        return Poly(self.binding_polynomial, self._c_symbol).all_coeffs()

    def print_summary(self):
        print("\n--- Binding Polynomial Summary ---")
        print(f"Free species: {self._c_species_name}, Total species: {self._ct_macrospecies_name}")
        print("\nConstants (Independent Fittable Parameters):", ", ".join(self._constants))
        print("Macrospecies:", ", ".join(self._macro_species))
        print("Microspecies:", ", ".join(self._micro_species))
        print("\n--- Derived Equations ---")

        if self.reparam_rules:
            print("\nReparameterization Rules:"); [print(f"  {s} = {e}") for s, e in self.reparam_rules.items()]
        
        print("\nSimplified Species Expressions (after reparameterization):"); [print(f"  {s} = {e}") for s, e in self.simplified_eqs.items()]
        print("\nSolved Base Variables:"); [print(f"  {s} = {e}") for s, e in self.solved_vars.items()]
        print(f"\nFinal Rational Equation for {self._c_species_name} (set to 0):\n  {self.final_rational_eq}")
        print(f"\nFinal Binding Polynomial for {self._c_species_name} (set to 0):\n  {self.binding_polynomial}")
        print("\nPolynomial Coefficients (descending power of C):")
        coeffs = self.get_polynomial_coefficients()
        for i, coeff in enumerate(coeffs): print(f"  C^{len(coeffs)-1-i}: {coeff}")
        print("\n--- End of Summary ---\n")