import numpy as np
import pandas as pd
from sympy import symbols, diff, Matrix, lambdify, Poly, sympify
import warnings

from .polynomial import BindingPolynomial
from .parameter_map import ParameterMapper


class SymbolicBindingModel:
    """
    Symbolic thermodynamic binding model.

    Wraps :class:`BindingPolynomial` to provide the full pipeline needed
    for fitting: reparameterization, concentration solving, and analytical
    Jacobian (and optionally Hessian) computation.

    **Construction sequence**

    1. Split the ``reparameterize:`` block from the rest of the spec.
    2. Build a :class:`BindingPolynomial` from the stripped spec.
    3. Identify all physical parameters (equilibrium constants, enthalpies,
       and any extra symbols introduced by reparameterization rules).
    4. Build a :class:`ParameterMapper` from the reparameterization rules
       to handle regression ↔ physical parameter conversions.
    5. Derive species expressions in terms of the free-monomer symbol.
    6. Compile the symbolic Jacobian ``d[species] / d[physical Ks]`` using
       the implicit function theorem (always performed).
    7. Optionally compile the symbolic Hessian (expensive; off by default).

    Parameters
    ----------
    model_spec : str
        Model specification text.  Must contain ``equilibria:`` and
        ``species:`` sections; may contain an optional
        ``reparameterize:`` section.
    debug : bool, default False
        If ``True``, print diagnostic messages during construction and
        concentration solving.
    use_symbolic_hessian : bool, default False
        If ``True``, derive and compile the second-derivative (Hessian)
        ``d²[species] / d[Ks]²`` in addition to the Jacobian.  Useful for
        Newton-type optimisers and HMC samplers that need exact curvature
        information.  Can be slow for models with many equilibria.

    Attributes
    ----------
    physical_poly : BindingPolynomial
        The polynomial derived from the physical (non-reparameterized) model.
    mapper : ParameterMapper
        Handles regression ↔ physical parameter mapping.
    regression_params : list of str
        Names of the free parameters that the optimiser sees.
    equilibrium_constants : list of str
        Names of all equilibrium constants in the physical model.
    c_symbol : sympy.Symbol
        The free-monomer SymPy symbol (polynomial variable).
    J_phys_func : callable or None
        Compiled lambdified Jacobian function; ``None`` until
        :meth:`_setup_symbolic_jacobian_components` completes.
    H_phys_func : callable or None
        Compiled lambdified Hessian function; ``None`` unless
        ``use_symbolic_hessian=True``.
    """

    def __init__(self,
                 model_spec: str,
                 debug: bool = False,
                 use_symbolic_hessian: bool = False):
        """
        Build the symbolic binding model.

        Parameters
        ----------
        model_spec : str
            Chemical model specification text.
        debug : bool, default False
            Enable verbose diagnostic output.
        use_symbolic_hessian : bool, default False
            Derive and compile the symbolic Hessian in addition to the
            Jacobian.  Increases construction time significantly for
            complex models.
        """
        self._model_spec = model_spec
        self._debug = debug
        self._use_symbolic_hessian = use_symbolic_hessian
        self.J_phys_func = None
        self.H_phys_func = None

        # 1. Separate the reparameterize block from the rest.
        self._clean_spec, self._reparam_block = self._split_reparam_section(
            model_spec
        )

        # 2. Physical binding polynomial (no reparameterization yet).
        self.physical_poly = BindingPolynomial(self._clean_spec, debug=debug)

        # 3. Collect all physical parameters.
        self.equilibrium_constants = self.physical_poly._constants
        self.enthalpy_params = [
            self._derive_dH_name(k) for k in self.equilibrium_constants
        ]
        self.existing_physical = self.equilibrium_constants + self.enthalpy_params
        self.other_physical_params = self._identify_all_reparam_symbols(
            self._reparam_block, self.existing_physical
        )
        self.all_physical_params = sorted(
            list(set(self.existing_physical + self.other_physical_params))
        )

        # 4. Parse reparameterization rules and build the mapper.
        self.reparam_rules_dict = self._parse_reparam_rules(self._reparam_block)
        self.mapper = ParameterMapper(
            self.all_physical_params, self.reparam_rules_dict
        )
        self.regression_params = self.mapper.regression_params

        # 5. Free-monomer symbol and species expressions.
        self.c_symbol = self.physical_poly._c_symbol

        solved_vars = self.physical_poly.solved_vars
        simplified_eqs = self.physical_poly.simplified_eqs
        base_subs = dict(solved_vars)

        self.species_exprs: dict = {}
        for s_name in self.physical_poly._micro_species:
            s_sym = self.physical_poly.symbols[s_name]
            if s_sym == self.c_symbol:
                self.species_exprs[s_name] = self.c_symbol
            elif s_sym in solved_vars:
                self.species_exprs[s_name] = solved_vars[s_sym]
            elif s_sym in simplified_eqs:
                self.species_exprs[s_name] = (
                    simplified_eqs[s_sym].subs(base_subs)
                )
            else:
                self.species_exprs[s_name] = s_sym

        # 6. Symbolic Jacobian (always compiled).
        self._setup_symbolic_jacobian_components()

        # 7. Symbolic Hessian (optional, expensive).
        if self._use_symbolic_hessian:
            self._setup_symbolic_hessian_components()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reparam_rules(self) -> dict:
        """
        Symbolic reparameterization rules (symbol → expression).

        Returns
        -------
        dict
            As stored by :class:`ParameterMapper`.
        """
        return self.mapper.rules_sympy

    # ------------------------------------------------------------------
    # Private helpers — spec parsing
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        """Print *msg* prefixed with the class name if debug mode is on."""
        if self._debug:
            print(f"[SymbolicBindingModel] {msg}")

    @staticmethod
    def _derive_dH_name(k_name: str) -> str:
        """
        Map an equilibrium-constant name to its enthalpy-parameter name.

        ``K1`` → ``dH_1``, ``KE`` → ``dH_E``, etc.

        Parameters
        ----------
        k_name : str
            Equilibrium constant name (expected to start with ``'K'``).

        Returns
        -------
        str
            Corresponding enthalpy parameter name.
        """
        suffix = k_name[1:] if k_name.startswith('K') else k_name
        return f"dH_{suffix}"

    def _split_reparam_section(self, spec: str) -> tuple:
        """
        Separate the ``reparameterize:`` block from the rest of the spec.

        The reparameterize block is handled by :class:`ParameterMapper`
        rather than by :class:`BindingPolynomial`, so it must be stripped
        before passing the spec to the polynomial parser.

        Parameters
        ----------
        spec : str
            Full model specification text.

        Returns
        -------
        tuple of (str, str)
            ``(clean_spec, reparam_block)`` where ``clean_spec`` has the
            reparameterize section removed and ``reparam_block`` contains
            only the reparameterization lines.
        """
        clean_lines = []
        reparam_lines = []
        in_reparam = False

        for line in spec.split('\n'):
            if 'reparameterize:' in line:
                in_reparam = True
                continue
            if in_reparam:
                # A new section header ends the reparam block.
                if (line.strip().endswith(':')
                        and '=' not in line
                        and line.strip() != "reparameterize:"):
                    in_reparam = False
                    clean_lines.append(line)
                else:
                    reparam_lines.append(line)
            else:
                clean_lines.append(line)

        return "\n".join(clean_lines), "\n".join(reparam_lines)

    def _parse_reparam_rules(self, reparam_str: str) -> dict:
        """
        Parse the reparameterize block into a string-keyed rules dict.

        Parameters
        ----------
        reparam_str : str
            The reparameterization block (lines of the form
            ``K2 = K1 * alpha``).

        Returns
        -------
        dict of str → str
            Dependent variable name → expression string.
        """
        rules: dict = {}
        for line in reparam_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                rules[lhs.strip()] = rhs.strip()
        return rules

    def _identify_all_reparam_symbols(self,
                                      reparam_str: str,
                                      existing_ks: list) -> list:
        """
        Find all new independent symbols introduced in the reparam block.

        Any symbol that appears in a reparameterization expression and is
        not already in ``existing_ks`` (equilibrium constants and enthalpies)
        is a new free parameter, e.g. ``alpha`` in ``K2 = K1 * alpha``.

        Parameters
        ----------
        reparam_str : str
            The reparameterization block text.
        existing_ks : list of str
            Already-known physical parameter names.

        Returns
        -------
        list of str
            Sorted list of new symbol names found on the RHS of rules.
        """
        found_symbols: set = set()

        for line in reparam_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                lhs, rhs = line.split('=', 1)
                found_symbols.add(lhs.strip())
                try:
                    expr = sympify(rhs.strip())
                    for sym in expr.free_symbols:
                        found_symbols.add(str(sym))
                except Exception as e:
                    if self._debug:
                        warnings.warn(
                            f"Could not parse RHS of '{line}': {e}",
                            UserWarning,
                            stacklevel=2,
                        )

        return sorted(list(found_symbols - set(existing_ks)))

    # ------------------------------------------------------------------
    # Private helpers — symbolic Jacobian / Hessian setup
    # ------------------------------------------------------------------

    def _setup_symbolic_jacobian_components(self) -> None:
        """
        Derive and compile the analytical Jacobian
        ``d[species] / d[physical Ks]``.

        Uses the implicit function theorem applied to the binding
        polynomial ``P(c, K) = 0``::

            dc/dK = −(∂P/∂K) / (∂P/∂c)

        Then applies the chain rule for each species expression
        ``S(c, K)``::

            dS/dK = (∂S/∂c) · (dc/dK) + (∂S/∂K)

        The result is stored as ``self.J_phys_symbolic`` (SymPy Matrix of
        shape ``(n_species, n_Ks)``) and compiled into
        ``self.J_phys_func`` via :func:`sympy.lambdify`.
        """
        self._log("Deriving Jacobians...")

        self.c_symbol = self.physical_poly._c_symbol
        self.species_symbols = [
            self.physical_poly.symbols[s]
            for s in self.physical_poly._micro_species
        ]
        self.k_symbols = [
            self.physical_poly.symbols[k]
            for k in self.equilibrium_constants
        ]

        P = self.physical_poly.binding_polynomial
        dP_dc = diff(P, self.c_symbol)
        dP_dKs = [diff(P, k) for k in self.k_symbols]
        dc_dKs = [-dpk / dP_dc for dpk in dP_dKs]

        # Recompute species expressions (may differ from __init__ if
        # _setup_symbolic_jacobian_components is called after init).
        solved_vars = self.physical_poly.solved_vars
        simplified_eqs = self.physical_poly.simplified_eqs
        base_subs = dict(solved_vars)

        self.species_exprs = {}
        for s_name in self.physical_poly._micro_species:
            s_sym = self.physical_poly.symbols[s_name]
            if s_sym == self.c_symbol:
                self.species_exprs[s_name] = self.c_symbol
            elif s_sym in solved_vars:
                self.species_exprs[s_name] = solved_vars[s_sym]
            elif s_sym in simplified_eqs:
                self.species_exprs[s_name] = (
                    simplified_eqs[s_sym].subs(base_subs)
                )
            else:
                self.species_exprs[s_name] = s_sym

        rows = []
        for s_name in self.physical_poly._micro_species:
            expr = self.species_exprs[s_name]
            d_expr_dc = diff(expr, self.c_symbol)
            row = [
                d_expr_dc * dc_dKs[i] + diff(expr, k_sym)
                for i, k_sym in enumerate(self.k_symbols)
            ]
            rows.append(row)

        # Shape: (n_species, n_Ks)
        self.J_phys_symbolic = Matrix(rows)

        self.input_syms_J_phys = sorted(
            list(self.J_phys_symbolic.free_symbols), key=lambda s: s.name
        )
        self.J_phys_func = lambdify(
            self.input_syms_J_phys, self.J_phys_symbolic, modules="numpy"
        )

    def _setup_symbolic_hessian_components(self) -> None:
        """
        Derive and compile the analytical Hessian
        ``d²[species] / d[Ki] d[Kj]``.

        Differentiates each entry of ``J_phys_symbolic`` with respect to
        every physical-K symbol to produce a list of
        ``(n_Ks × n_Ks)`` SymPy matrices, one per species.  The result is
        compiled into ``self.H_phys_func`` which returns a NumPy array of
        shape ``(n_species, n_Ks, n_Ks)``.

        .. note::
            This can be slow for models with more than two or three
            equilibria.  Only enable when second-order curvature information
            is genuinely needed (e.g. Newton solvers or HMC samplers).
        """
        self._log("Deriving symbolic Hessian (second diff pass)...")

        n_species = len(self.physical_poly._micro_species)
        n_ks = len(self.k_symbols)

        hess_matrices = []
        for s_idx in range(n_species):
            hess_rows = []
            for i, k_i in enumerate(self.k_symbols):
                hess_row = [
                    diff(self.J_phys_symbolic[s_idx, i], k_j)
                    for k_j in self.k_symbols
                ]
                hess_rows.append(hess_row)
            hess_matrices.append(Matrix(hess_rows))

        # list of (n_Ks × n_Ks) SymPy matrices, one per species
        self.H_phys_symbolic = hess_matrices

        all_free: set = set()
        for mat in self.H_phys_symbolic:
            all_free |= mat.free_symbols
        all_free |= self.J_phys_symbolic.free_symbols
        self.input_syms_H_phys = sorted(list(all_free), key=lambda s: s.name)

        self.H_phys_func = lambdify(
            self.input_syms_H_phys,
            self.H_phys_symbolic,
            modules="numpy"
        )

    # ------------------------------------------------------------------
    # Public interface — parameter conversion
    # ------------------------------------------------------------------

    def get_physical_params(self, reg_params_dict: dict) -> dict:
        """
        Convert regression parameters to physical parameters.

        Delegates to :class:`ParameterMapper`.

        Parameters
        ----------
        reg_params_dict : dict
            Regression parameter name → value.

        Returns
        -------
        dict
            Physical parameter name → value.
        """
        return self.mapper.get_physical_params(reg_params_dict)

    def get_physical_jacobian(self, reg_params_dict: dict) -> np.ndarray:
        """
        Compute ``d[physical] / d[regression]``.

        Delegates to :class:`ParameterMapper`.

        Parameters
        ----------
        reg_params_dict : dict
            Regression parameter name → value.

        Returns
        -------
        numpy.ndarray, shape (n_physical, n_regression)
            Jacobian matrix.
        """
        return self.mapper.get_jacobian(reg_params_dict)

    # ------------------------------------------------------------------
    # Public interface — concentration solving
    # ------------------------------------------------------------------

    def solve_concentrations(self,
                             regression_params_dict: dict,
                             macro_concentrations_dict: dict,
                             prev_c: float | None = None) -> dict:
        """
        Solve for equilibrium micro-species concentrations.

        Steps:

        1. Convert regression parameters to physical parameters via the
           mapper.
        2. Evaluate the polynomial coefficients numerically.
        3. Find polynomial roots with ``numpy.roots``; filter for real,
           non-negative roots ``≤ CT``.
        4. Select the root using continuity tracking (pick the root
           closest to ``prev_c`` when multiple valid roots exist) or fall
           back to the smallest valid root.
        5. Back-calculate all micro-species concentrations from the
           selected free-monomer value.

        Parameters
        ----------
        regression_params_dict : dict
            Optimisation parameter values (regression space).
        macro_concentrations_dict : dict
            Total concentrations of all macro-species.
        prev_c : float or None, default None
            Free-monomer concentration from the previous titration point,
            used for continuity-based root selection.  Pass ``None`` for
            the first injection.

        Returns
        -------
        dict
            Complete state dictionary including regression parameters,
            physical parameters, total concentrations, the free-monomer
            value keyed by the free-monomer symbol name (e.g. ``'C'``),
            and all micro-species concentrations.

        Raises
        ------
        KeyError
            If any symbol required by the polynomial or species expressions
            is missing from the combined parameter + concentration dict.
        """
        # 1. Map regression → physical.
        phys_vals = self.mapper.get_physical_params(regression_params_dict)
        solver_input = {**phys_vals, **macro_concentrations_dict}

        # 2. Evaluate polynomial coefficients (cache lambdified functions).
        coeffs_sym = self.physical_poly.get_polynomial_coefficients()

        if not hasattr(self, '_coeff_funcs'):
            self._coeff_funcs = []
            self._coeff_args_maps = []
            for c_expr in coeffs_sym:
                free_syms = sorted(
                    list(c_expr.free_symbols), key=lambda s: s.name
                )
                self._coeff_args_maps.append(free_syms)
                self._coeff_funcs.append(
                    lambdify(free_syms, c_expr, modules='numpy')
                )

        coeffs_num = []
        for func, needed_syms in zip(self._coeff_funcs, self._coeff_args_maps):
            try:
                func_args = [solver_input[s.name] for s in needed_syms]
            except KeyError:
                missing = [
                    s.name for s in needed_syms if s.name not in solver_input
                ]
                raise KeyError(
                    f"Missing symbols {missing} required to evaluate "
                    "polynomial coefficients."
                )
            coeffs_num.append(func(*func_args))

        # 3. Find and filter roots.
        roots = np.roots(coeffs_num)

        ct_val = solver_input[self.physical_poly._ct_macrospecies_name]
        real_roots = roots[np.isreal(roots)].real
        valid_roots = real_roots[(real_roots >= 0) & (real_roots <= ct_val)]

        if len(valid_roots) == 0:
            # Relax bounds slightly to absorb floating-point noise.
            valid_roots = real_roots[
                (real_roots >= -1e-12) & (real_roots <= ct_val * 1.00001)
            ]

        # 4. Select root.
        if len(valid_roots) > 0:
            if prev_c is not None and len(valid_roots) > 1:
                c_sol = max(
                    0.0, valid_roots[np.argmin(np.abs(valid_roots - prev_c))]
                )
            else:
                c_sol = max(0.0, np.min(valid_roots))
        else:
            c_sol = 0.0

        # 5. Back-calculate all species concentrations.
        result = {
            **regression_params_dict,
            **phys_vals,
            **macro_concentrations_dict,
        }
        result[str(self.c_symbol)] = c_sol

        if not hasattr(self, '_species_funcs'):
            self._species_funcs = {}
            self._species_args_maps = {}
            for s_name, expr in self.species_exprs.items():
                free_syms = sorted(
                    list(expr.free_symbols), key=lambda s: s.name
                )
                self._species_args_maps[s_name] = free_syms
                self._species_funcs[s_name] = lambdify(
                    free_syms, expr, modules='numpy'
                )

        full_input = {**solver_input, str(self.c_symbol): c_sol}
        for s_name, func in self._species_funcs.items():
            needed_syms = self._species_args_maps[s_name]
            try:
                func_args = [full_input[s.name] for s in needed_syms]
            except KeyError:
                missing = [
                    s.name for s in needed_syms if s.name not in full_input
                ]
                raise KeyError(
                    f"Missing symbols {missing} required to evaluate "
                    f"species '{s_name}'."
                )
            result[s_name] = func(*func_args)

        return result

    # ------------------------------------------------------------------
    # Public interface — Jacobian evaluation
    # ------------------------------------------------------------------

    def get_conc_jacobian_vs_regression(self,
                                        concentrations_dict: dict,
                                        calibration_dict: dict | None = None
                                        ) -> np.ndarray:
        """
        Compute ``d[species] / d[regression_params]``.

        Applies the chain rule::

            J_reg = J_phys @ J_map

        where ``J_phys`` is ``d[species] / d[physical Ks]`` (evaluated
        from the compiled symbolic function) and ``J_map`` is
        ``d[physical Ks] / d[regression_params]`` (from
        :class:`ParameterMapper`).

        Parameters
        ----------
        concentrations_dict : dict
            Must contain numerical values for all free symbols in
            ``J_phys_symbolic`` (free-monomer ``c``, equilibrium constants,
            and total concentrations).
        calibration_dict : dict or None, optional
            Unused; retained for API compatibility with external callers.

        Returns
        -------
        numpy.ndarray, shape (n_species, n_regression)
            Jacobian of micro-species concentrations with respect to
            regression parameters.

        Raises
        ------
        ValueError
            If a required symbol is absent from ``concentrations_dict``.
        """
        # 1. Evaluate J_phys: shape (n_species, n_Ks)
        try:
            args = [concentrations_dict[sym.name]
                    for sym in self.input_syms_J_phys]
        except KeyError as e:
            raise ValueError(
                f"Missing value for symbol {e} in concentrations_dict. "
                f"Required keys: {[s.name for s in self.input_syms_J_phys]}"
            )

        J_phys_val = np.array(self.J_phys_func(*args))

        # 2. Evaluate J_map: shape (n_physical, n_regression)
        J_map_full = self.mapper.get_jacobian(concentrations_dict)

        k_indices = [
            self.all_physical_params.index(k)
            for k in self.equilibrium_constants
        ]
        J_map_Ks = J_map_full[k_indices, :]   # shape (n_Ks, n_regression)

        # 3. Chain rule: J_reg = J_phys @ J_map_Ks
        return J_phys_val @ J_map_Ks

    # ------------------------------------------------------------------
    # Public interface — Hessian evaluation
    # ------------------------------------------------------------------

    def get_conc_hessian_phys(self,
                               concentrations_dict: dict) -> np.ndarray:
        """
        Evaluate the symbolic Hessian of species concentrations with
        respect to the physical equilibrium constants.

        Parameters
        ----------
        concentrations_dict : dict
            Must contain numerical values for all free symbols in
            ``H_phys_symbolic``.

        Returns
        -------
        numpy.ndarray, shape (n_species, n_Ks, n_Ks)
            ``H[s, i, j] = d²[species_s] / d[K_i] d[K_j]``.

        Raises
        ------
        RuntimeError
            If the Hessian was not compiled (``use_symbolic_hessian=False``).
        """
        if self.H_phys_func is None:
            raise RuntimeError(
                "Symbolic Hessian not compiled. "
                "Initialise with use_symbolic_hessian=True."
            )
        args = [concentrations_dict[sym.name]
                for sym in self.input_syms_H_phys]
        raw = self.H_phys_func(*args)   # list of (n_Ks × n_Ks) arrays
        return np.array([np.array(m, dtype=float) for m in raw])

    def get_cost_gradient_and_hessian(self,
                                      concentrations_dict: dict,
                                      residuals: np.ndarray,
                                      obs_weights: np.ndarray,
                                      observation_jac: np.ndarray
                                      ) -> tuple:
        """
        Compute the gradient and approximate Hessian of the chi-squared
        cost function with respect to regression parameters.

        This method is intended for external use by second-order optimisers
        and HMC samplers.  It is **not** called anywhere in the standard
        ``GlobalModel`` fitting pipeline.

        The cost function is::

            C = 0.5 · sum_i( (r_i / σ_i)² )

        Gradient::

            g = −J_reg^T · r           (r already weighted by 1/σ)

        Hessian (Gauss-Newton approximation, always computed)::

            H ≈ J_reg^T · J_reg

        When ``use_symbolic_hessian=True`` the full second-order correction
        is added::

            H += sum_s( w_s · J_map_Ks^T · H_phys_s · J_map_Ks )

        where the per-species weight ``w_s`` is the mean absolute residual.
        This is a simplified approximation; a rigorous implementation would
        weight by the per-observation sensitivity to each species.

        Parameters
        ----------
        concentrations_dict : dict
            Full state dict (regression params, physical params, species
            concentrations).
        residuals : np.ndarray, shape (n_obs,)
            Weighted residuals ``(y_obs − y_pred) / σ``.
        obs_weights : np.ndarray, shape (n_obs,)
            ``1/σ`` values for each observation.
        observation_jac : np.ndarray, shape (n_obs, n_regression)
            Pre-computed weighted Jacobian ``d[y] / d[theta]``.

        Returns
        -------
        g : numpy.ndarray, shape (n_regression,)
            Gradient of the cost with respect to regression parameters.
        H : numpy.ndarray, shape (n_regression, n_regression)
            Approximate Hessian of the cost.
        """
        J_reg = observation_jac

        g = -J_reg.T @ residuals
        H = J_reg.T @ J_reg

        if self._use_symbolic_hessian and self.H_phys_func is not None:
            J_map_full = self.mapper.get_jacobian(concentrations_dict)
            k_indices = [
                self.all_physical_params.index(k)
                for k in self.equilibrium_constants
            ]
            J_map_Ks = J_map_full[k_indices, :]

            try:
                H_phys = self.get_conc_hessian_phys(concentrations_dict)
            except Exception:
                H_phys = None

            if H_phys is not None:
                r_weight = (
                    np.mean(np.abs(residuals)) if len(residuals) > 0 else 0.0
                )
                H_correction = np.zeros(
                    (J_map_Ks.shape[1], J_map_Ks.shape[1])
                )
                for s_idx in range(H_phys.shape[0]):
                    H_s_reg = J_map_Ks.T @ H_phys[s_idx] @ J_map_Ks
                    H_correction += r_weight * H_s_reg
                H = H + H_correction

        return g, H
