import warnings

import numpy as np
from sympy import symbols, sympify, Matrix, diff, lambdify


class ParameterMapper:
    """
    Translates between regression parameters (what the optimizer sees) and
    physical parameters (what the chemical model uses).

    Physical parameters are the quantities that appear directly in the
    binding polynomial — equilibrium constants ``K1``, ``K2``, … and
    enthalpies ``dH_1``, ``dH_2``, ….  Regression parameters are the
    free variables actually optimised; they equal the physical parameters
    unless a reparameterization rule is provided.

    Reparameterization rules allow constraints such as ``K2 = K1 * alpha``
    (symmetric two-site model) or ``dH_2 = dH_1`` (enthalpy linkage).
    The mapper parses these rules symbolically, identifies which parameters
    become dependent (and therefore are removed from the regression set),
    and introduces any new independent parameters (``alpha`` in the example
    above).

    All mappings are compiled into lambdified NumPy functions at
    construction time for fast numerical evaluation.

    Parameters
    ----------
    physical_params : list of str
        Full list of physical parameter names required by the model,
        e.g. ``['K1', 'K2', 'dH_1', 'dH_2']``.
    reparam_rules_dict : dict of str → str
        Rules mapping dependent parameter names to their string expressions.
        Example: ``{'K2': 'K1 * alpha', 'dH_2': 'dH_1'}``.
        Pass an empty dict when there are no reparameterisation rules.
    """

    def __init__(self,
                 physical_params: list,
                 reparam_rules_dict: dict):
        """
        Build the parameter mapper.

        Parses ``reparam_rules_dict``, determines the regression parameter
        set, builds the full symbolic mapping from regression → physical,
        and compiles forward-map and Jacobian functions.

        Parameters
        ----------
        physical_params : list of str
            Full list of physical parameter names required by the model.
        reparam_rules_dict : dict of str → str
            Reparameterization rules; may be empty.
        """
        self.physical_params = sorted(physical_params)
        self.reparam_rules_str = reparam_rules_dict

        self.regression_params: list = []
        self.rules_sympy: dict = {}
        self.mapping_funcs: dict = {}
        self.jacobian_func = None

        self._parse_rules()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_rules(self) -> None:
        """
        Parse the string reparameterization rules into SymPy expressions
        and compile numerical forward-map and Jacobian functions.

        Steps:

        1. Build a consistent set of SymPy Symbol objects for all
           physical parameters.
        2. Parse each rule string; identify new independent symbols
           introduced on the RHS (e.g. ``alpha`` in ``K2 = K1 * alpha``).
        3. Determine the regression-parameter set:
           ``(physical_params − dependent_params) ∪ new_params``.
        4. Iteratively substitute dependent-variable expressions until
           every physical parameter is expressed purely in terms of
           regression parameters.
        5. Lambdify the forward map and Jacobian for fast evaluation.
        """
        # 1. Consistent SymPy symbols for all physical parameters.
        self.symbols_map = {p: symbols(p) for p in self.physical_params}

        def get_sym(name: str):
            if name not in self.symbols_map:
                self.symbols_map[name] = symbols(name)
            return self.symbols_map[name]

        # 2. Parse rules → SymPy; collect new independent symbols.
        dependent_vars = set(self.reparam_rules_str.keys())
        potential_independent = set(self.physical_params) - dependent_vars

        new_symbols_found: set = set()
        self.rules_sympy = {}

        for dep_name, expr_str in self.reparam_rules_str.items():
            try:
                temp_expr = sympify(expr_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse rule '{dep_name} = {expr_str}': {e}"
                )

            final_expr = temp_expr
            for sym in temp_expr.free_symbols:
                s_name = str(sym)
                canonical_sym = get_sym(s_name)
                final_expr = final_expr.subs(sym, canonical_sym)

                if (s_name not in self.physical_params
                        and s_name not in dependent_vars):
                    new_symbols_found.add(s_name)

            self.rules_sympy[get_sym(dep_name)] = final_expr

        # 3. Regression parameters = independent physical + newly introduced.
        self.regression_params = sorted(
            list(potential_independent) + list(new_symbols_found)
        )

        # 4. Build the full symbolic mapping: physical_name → expr(regression).
        self.full_mapping_sympy: dict = {}
        for p_name in self.physical_params:
            p_sym = get_sym(p_name)
            if p_name in self.reparam_rules_str:
                self.full_mapping_sympy[p_name] = self.rules_sympy[p_sym]
            else:
                self.full_mapping_sympy[p_name] = p_sym

        # Iterative substitution until all expressions are in terms of
        # regression parameters only (handles chained rules).
        reg_param_names = set(self.regression_params)
        for _ in range(len(self.physical_params) + 5):
            dirty = False
            for p_name in self.physical_params:
                expr = self.full_mapping_sympy[p_name]
                current_syms = expr.free_symbols
                if not any(str(s) not in reg_param_names for s in current_syms):
                    continue
                subs_dict = {
                    get_sym(dep): self.full_mapping_sympy[dep]
                    for dep in dependent_vars
                    if get_sym(dep) in current_syms
                }
                if subs_dict:
                    new_expr = expr.subs(subs_dict)
                    if new_expr != expr:
                        self.full_mapping_sympy[p_name] = new_expr
                        dirty = True
            if not dirty:
                break

        # 5. Lambdify forward map and Jacobian.
        self.reg_syms = [get_sym(p) for p in self.regression_params]
        self._setup_numerical_functions()

    def _setup_numerical_functions(self) -> None:
        """
        Compile symbolic expressions into fast NumPy functions via
        ``sympy.lambdify``.

        Produces:

        * ``forward_funcs`` — dict mapping each physical parameter name to
          a callable that computes its value from the regression parameters.
        * ``jacobian_lam`` — callable that returns the ``(n_physical ×
          n_regression)`` Jacobian matrix ``d[physical] / d[regression]``.
        """
        # Forward map: one lambdified function per physical parameter.
        self.forward_funcs: dict = {}
        for p_name, expr in self.full_mapping_sympy.items():
            self.forward_funcs[p_name] = lambdify(
                self.reg_syms, expr, modules="numpy"
            )

        # Jacobian: d(physical_i) / d(regression_j).
        rows = []
        for p_name in self.physical_params:
            expr = self.full_mapping_sympy[p_name]
            rows.append([diff(expr, r_sym) for r_sym in self.reg_syms])

        self.jacobian_matrix = Matrix(rows)
        self.jacobian_lam = lambdify(
            self.reg_syms, self.jacobian_matrix, modules="numpy"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_physical_params(self, regression_params_dict: dict) -> dict:
        """
        Convert a regression-parameter value dict to a physical-parameter dict.

        Parameters
        ----------
        regression_params_dict : dict
            Mapping of regression parameter name → current numerical value.
            May contain extra keys (e.g. species concentrations); only the
            regression parameters are extracted.

        Returns
        -------
        dict
            Physical parameter names mapped to their computed values.

        Raises
        ------
        KeyError
            If a required regression parameter is absent from
            ``regression_params_dict``.
        ValueError
            If the result for any physical parameter is still symbolic
            (indicates an incomplete mapping — likely a bug in rule parsing).
        RuntimeError
            If numerical evaluation fails for any other reason.
        """
        try:
            args = [regression_params_dict[p] for p in self.regression_params]
        except KeyError as e:
            raise KeyError(
                f"Missing regression parameter: {e}. "
                f"Available: {list(regression_params_dict.keys())}"
            )

        phys_vals = {}
        for p_name in self.physical_params:
            try:
                func = self.forward_funcs[p_name]
                val = func(*args)

                if hasattr(val, 'free_symbols') and val.free_symbols:
                    raise ValueError(
                        f"Result for '{p_name}' is still symbolic: {val}. "
                        "Check that all reparameterization rules are fully resolved."
                    )

                phys_vals[p_name] = float(val)

            except (ValueError, KeyError):
                raise
            except Exception as e:
                warnings.warn(
                    f"Unexpected error evaluating physical parameter '{p_name}': {e}. "
                    f"Expression: {self.full_mapping_sympy[p_name]}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                raise RuntimeError(
                    f"Failed to evaluate '{p_name}': {e}"
                ) from e

        return phys_vals

    def get_jacobian(self, regression_params_dict: dict) -> np.ndarray:
        """
        Compute ``d[physical] / d[regression]``.

        Parameters
        ----------
        regression_params_dict : dict
            Mapping of regression parameter name → current numerical value.
            May contain extra keys; only the regression parameters are used.

        Returns
        -------
        numpy.ndarray, shape (n_physical, n_regression)
            Jacobian matrix where row ``i`` corresponds to
            ``physical_params[i]`` and column ``j`` to
            ``regression_params[j]``.
        """
        args = [regression_params_dict[p] for p in self.regression_params]
        return np.array(self.jacobian_lam(*args))
