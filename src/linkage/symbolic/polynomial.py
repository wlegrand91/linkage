import warnings

import numpy as np
from sympy import symbols, expand, simplify, collect, prod, Poly, fraction, sympify


class BindingPolynomial:
    """
    Derives a binding polynomial from a chemical model specification.

    Parses a text specification containing an ``equilibria:`` block and a
    ``species:`` (mass-conservation) block, then uses SymPy to derive a
    polynomial equation ``P(c) = 0`` where ``c`` is the free-ligand
    concentration.  The polynomial is used by :class:`SymbolicBindingModel`
    to solve for equilibrium concentrations numerically.

    The derivation proceeds in four stages:

    1. **Parse** — extract equilibrium reactions, equilibrium constant
       names, and mass-balance equations from the spec string.
    2. **Symbolic setup** — create SymPy symbols for every species and
       constant; build equilibrium expressions (``product = K * reactants``).
    3. **Simplify** — iteratively substitute equilibrium expressions so
       that every product species is expressed solely in terms of the free
       monomer ``c`` and the equilibrium constants.
    4. **Polynomial extraction** — substitute simplified expressions into
       the final mass-balance equation, clear denominators via
       ``sympy.cancel``, and expand the numerator to get ``P(c)``.

    Parameters
    ----------
    model_spec : str
        Model specification string (or path to a ``.txt`` file containing
        one).  Must include ``equilibria:`` and ``species:`` sections.
    debug : bool, default False
        If ``True``, print diagnostic messages during derivation.

    Attributes
    ----------
    binding_polynomial : sympy.Expr
        The derived polynomial ``P(c)`` (set equal to zero to solve for
        the free-ligand concentration).
    symbols : dict of str → sympy.Symbol
        SymPy Symbol object for every species and constant in the model.
    reparam_rules : dict
        Reparameterization rules parsed from the optional
        ``reparameterize:`` section (symbol → expression).
    """

    def __init__(self, model_spec: str, debug: bool = False):
        """
        Parse the model spec and derive the binding polynomial.

        Parameters
        ----------
        model_spec : str
            Chemical model specification text.
        debug : bool, default False
            Enable verbose diagnostic output.

        Raises
        ------
        ValueError
            If the model specification is empty, or if the free ligand
            species cannot be identified from the spec.
        """
        if not model_spec:
            raise ValueError("The model specification is empty.")

        self._model_spec = model_spec
        self._debug = debug

        self.reparam_rules: dict = {}
        self.new_fitting_vars: list = []

        (self._equilibria,
         self._constants,
         self._species,
         self._micro_species,
         self._macro_species) = self._parse_model_spec()

        self._c_species_name, self._ct_macrospecies_name = (
            self._detect_polynomial_species()
        )

        self._setup_symbolic_model()

    # ------------------------------------------------------------------
    # Private helpers — parsing
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Print *message* if debug mode is enabled."""
        if self._debug:
            print(message)

    def _parse_model_spec(self) -> tuple:
        """
        Parse all sections of the model specification.

        Returns
        -------
        tuple
            ``(equilibria, constants, species, micro_species, macro_species)``
            where:

            * ``equilibria`` — dict of K-name → ``[reactants, products]``
            * ``constants`` — sorted list of equilibrium constant names
            * ``species`` — dict of macro-name → ``(micro_list, stoich_list)``
            * ``micro_species`` — sorted list of all micro-species names
            * ``macro_species`` — sorted list of all macro-species names
        """
        self._log("\nParsing model specification...")
        equilibria, constants = self._parse_equilibria_section()
        species = self._parse_species_section()

        self.reparam_rules, self.new_fitting_vars = (
            self._parse_reparameterize_section(constants)
        )

        if self.reparam_rules:
            self._log("Reparameterization rules found. Updating fittable constants.")
            dependent_vars = [s.name for s in self.reparam_rules.keys()]
            new_poly_vars = [
                v for v in self.new_fitting_vars if not v.startswith("dH_")
            ]
            constants = sorted(
                [c for c in constants if c not in dependent_vars] + new_poly_vars
            )
            self._log(f"New polynomial constants: {constants}")

        micro_species, macro_species = self._validate_and_extract_species(
            equilibria, species
        )
        self._log("Finished parsing model specification.")
        return equilibria, constants, species, micro_species, macro_species

    def _parse_reparameterize_section(self,
                                      existing_constants: list) -> tuple:
        """
        Parse the optional ``reparameterize:`` section.

        Rules of the form ``K2 = K1 * alpha`` make ``K2`` a dependent
        variable and introduce ``alpha`` as a new fitting variable.

        Parameters
        ----------
        existing_constants : list of str
            Equilibrium constant names already parsed from the
            ``equilibria:`` section.

        Returns
        -------
        tuple of (dict, list)
            * ``reparam_rules`` — SymPy symbol → expression for each
              dependent variable.
            * ``new_fitting_vars`` — sorted list of newly introduced
              independent parameter names.

        Raises
        ------
        ValueError
            If a reparameterization expression cannot be parsed.
        """
        reparam_rules: dict = {}
        new_fitting_vars: set = set()
        in_reparam = False

        known_symbols = {c: symbols(c) for c in existing_constants}

        for line in self._model_spec.split('\n'):
            line = line.strip()
            if line.startswith("#") or not line:
                continue

            if 'reparameterize:' in line:
                in_reparam = True
                continue
            if 'equilibria:' in line or 'species:' in line:
                in_reparam = False

            if in_reparam and '=' in line:
                dependent_str, expr_str = line.split('=', 1)
                dependent_sym = symbols(dependent_str.strip())

                try:
                    expr = sympify(expr_str.strip(), locals=known_symbols)
                except Exception as e:
                    raise ValueError(
                        f"Could not parse reparameterization expression: "
                        f"'{line}'. Error: {e}"
                    )

                reparam_rules[dependent_sym] = expr

                for sym in expr.free_symbols:
                    if sym.name not in existing_constants:
                        new_fitting_vars.add(sym.name)
                        known_symbols[sym.name] = sym

        return reparam_rules, sorted(list(new_fitting_vars))

    def _parse_equilibria_section(self) -> tuple:
        """
        Parse the ``equilibria:`` section.

        Each line is expected in the form::

            A + B -> AB ; K1

        Returns
        -------
        tuple of (dict, list)
            * ``equilibria`` — dict of K-name → ``[reactants, products]``
            * ``constants`` — ordered list of equilibrium constant names
              (sorted for reproducibility).
        """
        equilibria: dict = {}
        constants: list = []
        in_equilibria = False

        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'equilibria:' in line:
                in_equilibria = True
                continue
            if 'species:' in line or 'reparameterize:' in line:
                in_equilibria = False
            if in_equilibria and '->' in line and ';' in line:
                reaction, K = line.split(';')
                K = K.strip()
                reactants_str, products_str = reaction.split('->')
                reactants = [r.strip() for r in reactants_str.split('+') if r.strip()]
                products = [p.strip() for p in products_str.split('+') if p.strip()]
                equilibria[K] = [reactants, products]
                if K not in constants:
                    constants.append(K)

        return equilibria, sorted(constants)

    def _parse_species_section(self) -> dict:
        """
        Parse the ``species:`` section.

        Each line is expected in the form::

            AT = A + AB
            AT = A + 2*AB2      # stoichiometric coefficient example

        Returns
        -------
        dict
            Macro-species name → ``(micro_species_list, stoich_list)``
            preserving the order in which conservation equations appear.
        """
        species: dict = {}
        in_species = False

        for line in self._model_spec.split('\n'):
            line = line.strip()
            if 'species:' in line:
                in_species = True
                continue
            if 'equilibria:' in line or 'reparameterize:' in line:
                in_species = False
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

    def _validate_and_extract_species(self,
                                      equilibria: dict,
                                      species: dict) -> tuple:
        """
        Collect and deduplicate micro- and macro-species from parsed sections.

        Micro-species are gathered from both the equilibria reactions and
        the mass-balance equations (to handle any species that appears only
        in one section).  Macro-species are the keys of the ``species``
        dict, i.e. the total-concentration variables.

        Parameters
        ----------
        equilibria : dict
            Parsed equilibrium reactions (K-name → ``[reactants, products]``).
        species : dict
            Parsed mass-balance equations (macro → ``(micros, stoichs)``).

        Returns
        -------
        tuple of (list, list)
            ``(micro_species, macro_species)`` — sorted lists of unique names.
        """
        micro_in_equilibria = set(
            r
            for _, (reactants, products) in equilibria.items()
            for r in reactants + products
        )
        micro_in_species = set(
            m for micros, _ in species.values() for m in micros
        )
        all_micro = sorted(list(micro_in_equilibria.union(micro_in_species)))
        macro = sorted(list(species.keys()))
        return all_micro, macro

    def _detect_polynomial_species(self) -> tuple:
        """
        Identify the free-monomer species and its total-concentration
        counterpart that define the polynomial variable.

        By convention the polynomial is written in terms of the free-ligand
        concentration from the *last* defined mass-balance equation.  The
        free species is identified as the micro-species in that equation
        that is not a *product* of any equilibrium reaction (i.e. it is
        not synthesised — it only appears as a reactant).

        Returns
        -------
        tuple of (str, str)
            ``(free_species_name, total_species_name)``

        Raises
        ------
        ValueError
            If no mass-balance block is found, or if the free species
            cannot be unambiguously identified.
        """
        if not self._macro_species:
            if "C" in self._micro_species:
                warnings.warn(
                    "No species block found. Defaulting polynomial variable "
                    "to 'C' with total concentration 'CT' (inferred).",
                    UserWarning,
                    stacklevel=2,
                )
                return "C", "CT"
            raise ValueError(
                "No species/mass-balance block found. "
                "Cannot determine the polynomial variable."
            )

        # The last defined total concentration is taken as the main binding calc.
        last_macro = list(self._species.keys())[-1]
        micros_in_total, _ = self._species[last_macro]

        # Free species = micro-species that are never the *product* of a reaction.
        all_products: set = set()
        for _, products in self._equilibria.values():
            all_products.update(products)

        candidates = [m for m in micros_in_total if m not in all_products]

        if len(candidates) == 0:
            # Fallback: name-convention match (e.g. 'CT' → 'C').
            for m in micros_in_total:
                if m.upper() + "T" == last_macro.upper():
                    return m, last_macro
            raise ValueError(
                f"Could not identify a free species in the final mass balance "
                f"'{last_macro}'. All components are products of equilibria."
            )

        if len(candidates) == 1:
            return candidates[0], last_macro

        # Multiple candidates: prefer name-convention match, then 'C'.
        for m in candidates:
            if m.upper() + "T" == last_macro.upper():
                return m, last_macro
        if "C" in candidates:
            return "C", last_macro

        return candidates[0], last_macro

    # ------------------------------------------------------------------
    # Private helpers — symbolic derivation
    # ------------------------------------------------------------------

    def _setup_symbolic_model(self) -> None:
        """
        Orchestrate the symbolic derivation of the binding polynomial.

        Steps:

        1. Create SymPy Symbol objects for every species and constant.
        2. Build equilibrium expressions (``product = K * reactants``).
        3. Iteratively simplify so each product is expressed in terms of
           the free monomer and the constants.
        4. Optionally apply reparameterization substitutions.
        5. Solve the non-CT conservation equations to express intermediate
           free species in terms of the main free monomer.
        6. Substitute into the CT equation and extract the polynomial
           numerator via :meth:`_create_polynomial_from_rational`.
        """
        self._log("\nSetting up symbolic model...")
        all_symbol_names = (
            self._micro_species + self._macro_species + self._constants
        )
        self.symbols = {name: symbols(name) for name in all_symbol_names}

        self._c_symbol = self.symbols[self._c_species_name]
        self.equilibrium_eqs = self._create_equilibrium_equations()
        self.simplified_eqs = self._simplify_equilibria(self.equilibrium_eqs)

        if self.reparam_rules:
            self._log("Applying reparameterization substitutions to symbolic model.")
            poly_rules = {
                s: e for s, e in self.reparam_rules.items()
                if not s.name.startswith("dH_")
            }
            self.simplified_eqs = {
                product: rhs.subs(poly_rules)
                for product, rhs in self.simplified_eqs.items()
            }

        self.solved_vars, self.final_rational_eq = (
            self._solve_conservation_equations(self.simplified_eqs)
        )
        self.binding_polynomial = self._create_polynomial_from_rational(
            self.final_rational_eq
        )
        self._log("Symbolic model setup complete.")

    def _create_equilibrium_equations(self) -> dict:
        """
        Build symbolic mass-action expressions for each equilibrium reaction.

        For a reaction ``A + B → AB ; K1`` the expression is
        ``AB = K1 * A * B``.

        Returns
        -------
        dict of sympy.Symbol → sympy.Expr
            Product symbol mapped to its mass-action expression.
        """
        eqs: dict = {}
        all_constants_from_eq = list(self._equilibria.keys())
        all_symbols_to_create = self._micro_species + all_constants_from_eq
        temp_symbols = {name: symbols(name) for name in all_symbols_to_create}

        for K, (reactants, products) in self._equilibria.items():
            for product in products:
                reactant_syms = [temp_symbols[r] for r in reactants]
                eqs[temp_symbols[product]] = temp_symbols[K] * prod(reactant_syms)

        return eqs

    def _simplify_equilibria(self, equilibrium_eqs: dict) -> dict:
        """
        Express every product species purely in terms of the free monomer
        and the equilibrium constants.

        Iteratively substitutes ``product → K * reactants`` for all
        product symbols until no further substitutions can be made.  This
        handles cascaded reactions where the product of one equilibrium is
        a reactant in the next.

        Parameters
        ----------
        equilibrium_eqs : dict of sympy.Symbol → sympy.Expr
            Initial mass-action expressions from
            :meth:`_create_equilibrium_equations`.

        Returns
        -------
        dict of sympy.Symbol → sympy.Expr
            Fully simplified expressions, each containing only the free
            monomer and equilibrium constant symbols.
        """
        simplified: dict = {}
        for product_sym, rhs_expr in equilibrium_eqs.items():
            expr = rhs_expr
            for _ in range(len(equilibrium_eqs) + 1):
                made_change = False
                temp_expr = expr
                for sub_sym, sub_expr in equilibrium_eqs.items():
                    if sub_sym in temp_expr.free_symbols:
                        expr = expr.subs(sub_sym, sub_expr)
                        made_change = True
                if not made_change:
                    break
            simplified[product_sym] = expand(expr)
        return simplified

    def _solve_conservation_equations(self, simplified_eqs: dict) -> tuple:
        """
        Solve the non-CT mass-balance equations to express intermediate
        free species in terms of the main free monomer ``c``.

        For a two-component model (``E + C ⇌ EC``) the ET equation gives
        ``E`` as a function of ``c``.  That expression is then substituted
        into the CT equation to yield a rational expression in ``c`` alone.

        The free "base variable" for each non-CT balance is identified as
        the micro-species that is *not* a product of any equilibrium —
        i.e. the species that can exist in free form.  This approach is
        more robust than the old name-convention (``AT`` → ``A``) and
        handles models where the free form has an unrelated name (e.g.
        ``I`` for an inactive state in a conformational-change model).

        Parameters
        ----------
        simplified_eqs : dict of sympy.Symbol → sympy.Expr
            Simplified equilibrium expressions from
            :meth:`_simplify_equilibria`.

        Returns
        -------
        tuple of (dict, sympy.Expr)
            * ``solved_vars`` — base-variable symbol → expression in ``c``.
            * ``final_rational`` — rational expression that equals zero at
              the physical solution (to be converted to a polynomial).
        """
        solved_vars: dict = {}

        all_eq_products: set = set()
        for _, (_, products) in self._equilibria.items():
            all_eq_products.update(products)

        # Map each non-CT macro species to its free-form micro-species symbol.
        base_var_for_macro: dict = {}
        for macro_name, (micro_list, _) in self._species.items():
            if macro_name == self._ct_macrospecies_name:
                continue
            non_product_micros = [
                m for m in micro_list
                if m not in all_eq_products and m in self.symbols
            ]
            if len(non_product_micros) == 1:
                base_var_for_macro[macro_name] = self.symbols[non_product_micros[0]]
            elif len(non_product_micros) == 0:
                # Fallback: old naming convention (e.g. 'AT' → 'A').
                fallback = self.symbols.get(macro_name[:-1])
                if fallback is not None:
                    base_var_for_macro[macro_name] = fallback

        for total_macro_name, (micro_list, stoich_list) in self._species.items():
            base_var_sym = base_var_for_macro.get(total_macro_name)
            if base_var_sym is None:
                continue

            rhs_sum = 0
            for micro_str, stoich_val in zip(micro_list, stoich_list):
                micro_sym = self.symbols[micro_str]
                expr_for_micro = simplified_eqs.get(micro_sym, micro_sym)
                rhs_sum += stoich_val * expr_for_micro

            for solved_sym, solved_expr in solved_vars.items():
                if solved_sym in rhs_sum.free_symbols:
                    rhs_sum = rhs_sum.subs(solved_sym, solved_expr)

            rhs_sum = expand(rhs_sum)
            collected = collect(rhs_sum, base_var_sym)
            coeff = collected.coeff(base_var_sym, 1)
            terms_without = collected.coeff(base_var_sym, 0)

            if coeff != 0:
                total_sym = self.symbols[total_macro_name]
                solved_vars[base_var_sym] = simplify(
                    (total_sym - terms_without) / coeff
                )

        # Build the final CT mass-balance expression in c.
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

        final_rational = simplify(
            final_ct_rhs - self.symbols[self._ct_macrospecies_name]
        )
        return solved_vars, final_rational

    def _create_polynomial_from_rational(self, rational_eq) -> object:
        """
        Convert the final rational equation into a polynomial expression.

        Calls ``sympy.cancel`` to combine all additive rational sub-terms
        over a common denominator, then extracts the numerator.  Without
        ``cancel``, expressions such as
        ``a/(1+K1*C) + b/(1+KE*C) + C − CT`` are not properly merged and
        ``fraction()`` returns a still-rational numerator.

        Parameters
        ----------
        rational_eq : sympy.Expr
            The rational equation ``CT_expr(c) − CT = 0`` produced by
            :meth:`_solve_conservation_equations`.

        Returns
        -------
        sympy.Expr
            Expanded polynomial numerator ``P(c)``.
        """
        from sympy import cancel
        combined = cancel(rational_eq)
        numer, _ = fraction(combined)
        return expand(numer)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_polynomial_coefficients(self) -> list:
        """
        Return the coefficients of the binding polynomial ``P(c)``.

        Coefficients are listed in descending power order (highest degree
        first) as required by ``numpy.roots``.

        Returns
        -------
        list of sympy.Expr
            Polynomial coefficients, each a SymPy expression in the
            equilibrium constants and total concentrations.

        Raises
        ------
        RuntimeError
            If the binding polynomial has not yet been generated.
        """
        if not hasattr(self, 'binding_polynomial'):
            raise RuntimeError("Binding polynomial has not been generated.")
        return Poly(self.binding_polynomial, self._c_symbol).all_coeffs()

    def print_summary(self) -> None:
        """
        Print a human-readable summary of the derived symbolic model.

        Displays equilibrium constants, species lists, simplified
        equilibrium expressions, solved base-variable expressions, the
        final rational equation, the binding polynomial, and the
        polynomial coefficients.
        """
        print("\n--- Binding Polynomial Summary ---")
        print(
            f"Free species: {self._c_species_name}, "
            f"Total species: {self._ct_macrospecies_name}"
        )
        print("\nConstants (Independent Fittable Parameters):",
              ", ".join(self._constants))
        print("Macrospecies:", ", ".join(self._macro_species))
        print("Microspecies:", ", ".join(self._micro_species))
        print("\n--- Derived Equations ---")

        if self.reparam_rules:
            print("\nReparameterization Rules:")
            for s, e in self.reparam_rules.items():
                print(f"  {s} = {e}")

        print("\nSimplified Species Expressions (after reparameterization):")
        for s, e in self.simplified_eqs.items():
            print(f"  {s} = {e}")

        print("\nSolved Base Variables:")
        for s, e in self.solved_vars.items():
            print(f"  {s} = {e}")

        print(
            f"\nFinal Rational Equation for {self._c_species_name} "
            f"(set to 0):\n  {self.final_rational_eq}"
        )
        print(
            f"\nFinal Binding Polynomial for {self._c_species_name} "
            f"(set to 0):\n  {self.binding_polynomial}"
        )
        print("\nPolynomial Coefficients (descending power of C):")
        coeffs = self.get_polynomial_coefficients()
        for i, coeff in enumerate(coeffs):
            print(f"  C^{len(coeffs) - 1 - i}: {coeff}")

        print("\n--- End of Summary ---\n")
