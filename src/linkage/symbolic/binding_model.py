import warnings

import numpy as np
import pandas as pd

from .model import SymbolicBindingModel


class BindingModel:
    """
    Adapter between :class:`SymbolicBindingModel` and :class:`GlobalModel`.

    :class:`SymbolicBindingModel` handles the symbolic math (polynomial
    derivation, Jacobian compilation, root-finding).  :class:`BindingModel`
    wraps it with the interface that :class:`GlobalModel` expects:

    * Flat parameter-name arrays (``param_names``, ``macro_species``,
      ``micro_species``).
    * A ``get_concs`` method that takes NumPy arrays rather than dicts and
      applies the log-K → K transform.
    * A ``get_numerical_jacobian`` method that applies the log-K chain rule
      so the returned matrix represents ``d[concs] / d[log_K]``.
    * Concentration history tracking via an internal DataFrame.

    Parameters
    ----------
    model_spec : str
        Model specification text (or path to a ``.txt`` file).  Must
        contain ``equilibria:`` and ``species:`` sections.
    debug : bool, default False
        If ``True``, print diagnostic messages during concentration
        solving and Jacobian calculation.
    """

    def __init__(self, model_spec: str, debug: bool = False):
        """
        Build the binding-model adapter.

        Parameters
        ----------
        model_spec : str
            Chemical model specification text.
        debug : bool, default False
            Enable verbose diagnostic output.

        Raises
        ------
        ValueError
            If *model_spec* is ``None``.
        """
        if model_spec is None:
            raise ValueError("No model specification provided.")

        self.model_spec = model_spec
        self._debug = debug

        # Build the full symbolic model.
        self._bm = SymbolicBindingModel(model_spec, debug=debug)

        self._equilibria = self._bm.physical_poly._equilibria
        self._constants = self._bm.equilibrium_constants
        self._micro_species = self._bm.physical_poly._micro_species
        self._macro_species = self._bm.physical_poly._macro_species
        self._fit_params = self._bm.regression_params

        self._concentrations_df = pd.DataFrame(
            columns=self._micro_species, dtype=float
        )
        self._last_result: dict | None = None
        # When True, solved concentrations are appended to _concentrations_df.
        self._track_concentrations: bool = True

        self.simplified_eqs = self._bm.physical_poly.simplified_eqs
        self.solved_vars = self._bm.physical_poly.solved_vars
        self.final_ct = self._bm.physical_poly.binding_polynomial

        # Flags checked by GlobalModel to select code paths.
        self.jacobian_function: bool | None = (
            True
            if (hasattr(self._bm, "get_conc_jacobian_vs_regression")
                and self._bm.J_phys_func is not None)
            else None
        )
        self.hessian_function: bool | None = (
            True
            if (hasattr(self._bm, "H_phys_func")
                and self._bm.H_phys_func is not None)
            else None
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def param_names(self) -> np.ndarray:
        """
        Regression parameter names in the order expected by
        :meth:`get_concs`.

        Returns
        -------
        numpy.ndarray of str
        """
        return np.array(self._fit_params)

    @property
    def macro_species(self) -> np.ndarray:
        """
        Total-concentration (macro-species) names.

        Returns
        -------
        numpy.ndarray of str
        """
        return np.array(self._macro_species)

    @property
    def micro_species(self) -> np.ndarray:
        """
        Micro-species names in the order returned by :meth:`get_concs`.

        Returns
        -------
        numpy.ndarray of str
        """
        return np.array(self._micro_species)

    @property
    def equilibria(self) -> dict:
        """
        Chemical equilibria parsed from the model spec.

        Returns
        -------
        dict
            K-name → ``[reactants, products]``.
        """
        return self._equilibria

    @property
    def concentrations_df(self) -> pd.DataFrame:
        """
        History of micro-species concentrations from every :meth:`get_concs`
        call (when :attr:`_track_concentrations` is ``True``).

        Returns
        -------
        pandas.DataFrame
        """
        return self._concentrations_df

    @property
    def reparam_rules(self) -> dict:
        """
        Symbolic reparameterization rules (symbol → expression).

        Returns
        -------
        dict
        """
        return self._bm.reparam_rules

    @property
    def physical_param_names(self) -> list:
        """
        Names of the physical parameters (pre-reparameterization).

        Returns
        -------
        list of str
        """
        return self._bm.mapper.physical_params

    # ------------------------------------------------------------------
    # Parameter conversion helpers
    # ------------------------------------------------------------------

    def get_physical_params(self, reg_params_dict: dict) -> dict:
        """
        Convert regression parameters to physical parameters.

        Parameters
        ----------
        reg_params_dict : dict
            Regression parameter name → value.

        Returns
        -------
        dict
            Physical parameter name → value.
        """
        return self._bm.get_physical_params(reg_params_dict)

    def get_physical_jacobian(self, reg_params_dict: dict) -> np.ndarray:
        """
        Compute ``d[physical] / d[regression]``.

        Parameters
        ----------
        reg_params_dict : dict
            Regression parameter name → value.

        Returns
        -------
        numpy.ndarray, shape (n_physical, n_regression)
        """
        return self._bm.get_physical_jacobian(reg_params_dict)

    # ------------------------------------------------------------------
    # Concentration solving
    # ------------------------------------------------------------------

    def get_concs(self,
                  param_array: np.ndarray,
                  macro_array: np.ndarray) -> np.ndarray:
        """
        Solve for equilibrium micro-species concentrations.

        The caller provides parameter values in **regression space** (i.e.
        ``log K`` for binding constants, raw ``dH`` values for enthalpies).
        :class:`SymbolicBindingModel` handles the regression → physical
        conversion internally.

        Continuity-based root selection is applied automatically by
        passing the free-monomer value from the previous call (stored in
        ``self._last_result``) as ``prev_c``.

        Parameters
        ----------
        param_array : numpy.ndarray
            Regression parameter values in the order of
            :attr:`param_names`.
        macro_array : numpy.ndarray
            Total macro-species concentrations in the order of
            :attr:`macro_species`.

        Returns
        -------
        numpy.ndarray, shape (n_micro,)
            Micro-species concentrations in the order of
            :attr:`micro_species`.  Returns an array of ``nan`` values if
            concentration solving fails.
        """
        reg_dict = dict(zip(self._fit_params, param_array))
        macro_dict = dict(zip(self._macro_species, macro_array))

        prev_c: float | None = None
        if self._last_result is not None:
            prev_c = self._last_result.get(str(self._bm.c_symbol))

        try:
            result = self._bm.solve_concentrations(
                reg_dict, macro_dict, prev_c=prev_c
            )
            self._last_result = result

            concs_dict = {s: result[s] for s in self._micro_species}

            if self._track_concentrations:
                df_row = pd.DataFrame(
                    [concs_dict], columns=self._micro_species
                )
                self._concentrations_df = pd.concat(
                    [self._concentrations_df, df_row], ignore_index=True
                )

            return np.array([concs_dict[s] for s in self._micro_species])

        except Exception as e:
            if self._debug:
                print(f"Solving failed: {e}")
            self._last_result = None
            return np.full(len(self._micro_species), np.nan)

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def get_numerical_jacobian(self,
                                concs_dict: dict) -> np.ndarray | None:
        """
        Compute ``d[micro_species] / d[fit_params]``.

        Uses the compiled symbolic Jacobian from
        :class:`SymbolicBindingModel`.  The chain rule for log-transformed
        binding-constant parameters is applied here so that the returned
        matrix represents ``d[concs] / d[log_K]`` for those columns::

            d[concs] / d[log_K] = d[concs] / d[K]  ·  K

        Parameters
        ----------
        concs_dict : dict
            Current regression-space parameter values and solved
            micro-species concentrations, keyed by name.  K values should
            be in **linear space** (i.e. the actual equilibrium constants,
            not their logarithms) because the log chain rule is applied
            inside this method.

        Returns
        -------
        numpy.ndarray, shape (n_micro, n_fit_params), or None
            Jacobian matrix, or ``None`` if the calculation fails.
        """
        if concs_dict is None:
            return None

        try:
            phys_params = self._bm.mapper.get_physical_params(concs_dict)
            solver_result = {**concs_dict, **phys_params}
            J_lin = self._bm.get_conc_jacobian_vs_regression(solver_result)

            # Chain rule: d[concs]/d[log_K] = d[concs]/d[K] · K
            J_fit = J_lin.copy()
            for i, p_name in enumerate(self._fit_params):
                if not (p_name.startswith("dH") or "nuisance" in p_name):
                    J_fit[:, i] *= concs_dict.get(p_name, 1.0)

            return J_fit

        except Exception as e:
            if self._debug:
                print(f"Jacobian calculation failed: {e}")
            return None
