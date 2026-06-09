from linkage.global_model.point.spec_point import SpecPoint
from linkage.global_model.point.itc_point import ITCPoint

import numpy as np
import pandas as pd
import copy
import warnings
import traceback
import os

from linkage.experiment import Experiment


class GlobalModel:

    def __init__(self,
                 expt_list: list,
                 model_spec: str,
                 verbose: bool = True):
        """
        Integrate a list of experiments with a symbolic thermodynamic model.

        Combines one or more ``Experiment`` objects with a ``BindingModel``
        to produce a callable model that predicts experimental observables
        (ITC heats and/or spectroscopic signals) from a shared parameter vector.

        The parameter vector assembled by this class contains, in order:

        1. **Binding constants** — one ``K`` per equilibrium, fit in log-space.
        2. **Reaction enthalpies** — one ``dH`` per independent equilibrium
           (dependent enthalpies are eliminated via reparameterisation rules
           defined in the model spec).
        3. **Heats of dilution** — one ``nuisance_dil_<species>`` per titrating
           species across all ITC experiments.
        4. **Concentration fudge factors** — one ``nuisance_expt_<i>_<s>_fudge``
           per experiment whose ``conc_to_float`` is set.

        Observable normalisation
        ------------------------
        Each observable type (e.g. ITC heat, a named spectroscopic channel) is
        normalised to ``(value - mean) / std`` across all experiments that share
        that observable, ensuring residuals from different observable types are
        on a comparable scale.

        Experiment weighting
        --------------------
        Each experiment is up-weighted inversely to its number of data points so
        that large experiments do not dominate the fit over small ones.

        Parameters
        ----------
        expt_list : list of Experiment
            Experiments with observables already registered via
            ``define_itc_observable`` or ``define_spectroscopic_observable``.
            Pass an empty list to use the model in simulation-only mode.
        model_spec : str
            Model specification for ``BindingModel``.  Can be either the
            spec text directly or a path to a ``.txt`` file containing it.
        verbose : bool, default True
            If ``True``, print status messages during model initialisation.
        """

        self._expt_list = copy.deepcopy(expt_list) if expt_list is not None else []
        self._verbose = verbose

        # Handle model_spec — accept either raw spec text or a file path
        self._model_spec_location = None
        self._model_spec = model_spec

        if isinstance(model_spec, str):
            if len(model_spec) < 1024 and os.path.isfile(model_spec):
                self._model_spec_location = os.path.abspath(model_spec)
                try:
                    with open(self._model_spec_location, 'r') as f:
                        self._model_spec = f.read()
                except Exception as e:
                    if self._verbose:
                        print(f"Warning: could not read model spec file "
                              f"'{self._model_spec_location}': {e}")
                    self._model_spec = model_spec

        # Scaling factor: enthalpy parameters are fit in kcal/mol but the
        # physics uses cal/mol internally.
        self._dh_scale = 1000.0

        self._load_model()

        self._get_expt_std_scalar()
        self._get_expt_normalization()
        self._load_observables()
        self._get_enthalpy_param()
        self._get_expt_fudge()
        self._build_point_map()

        # Cache the parameter vector last used by model() so that
        # jacobian_normalized() can skip redundant re-computation.
        self._model_state_params = None

    # ---------------------------------------------------------------------- #
    # Private initialisation helpers                                           #
    # ---------------------------------------------------------------------- #

    def _load_model(self) -> None:
        """
        Instantiate ``BindingModel`` from the provided model spec.

        Passes ``model_spec`` directly to ``BindingModel``, which
        parses the equilibria and species definitions, builds the binding
        polynomial, and compiles the symbolic Jacobian.
        """

        from linkage.symbolic import BindingModel
        self._bm = BindingModel(model_spec=self._model_spec)

        self._parameter_names: list[str] = []
        self._parameter_guesses: dict[str, float] = {}
        for p in self._bm.param_names:
            self._parameter_names.append(p)
            self._parameter_guesses[p] = 0.0

        self._bm_param_start_idx = 0
        self._bm_param_end_idx = len(self._parameter_names) - 1

    def _get_expt_std_scalar(self) -> None:
        """
        Compute per-experiment standard-deviation scaling factors.

        Scales each experiment's uncertainties so that experiments with many
        data points do not dominate over those with few.  The scalar for
        experiment ``i`` is::

            theta_i = n_i / sum(n)
            scalar_i = 1 - theta_i + max(theta)

        where ``n_i`` is the total number of non-ignored observation–point
        pairs for that experiment.  The scalar is 1 for the smallest experiment
        and increases for larger ones.
        """

        points_per_expt = []
        for expt in self._expt_list:
            num_obs = len(expt.observables)
            num_not_ignore = np.sum(np.logical_not(expt.expt_data["ignore_point"]))
            points_per_expt.append(num_obs * num_not_ignore)

        points_per_expt = np.array(points_per_expt)
        if np.sum(points_per_expt) > 0:
            theta = points_per_expt / np.sum(points_per_expt)
            self._expt_std_scalar = 1 - theta + np.max(theta)
        else:
            self._expt_std_scalar = np.ones(len(points_per_expt))

    def _get_expt_normalization(self) -> None:
        """
        Compute per-observable-type normalisation parameters.

        For each unique observable name seen across all experiments, collects
        all non-ignored values and computes their mean and standard deviation.
        These are later used to normalise residuals via::

            y_normalised = (y - mean) / std

        If all values for an observable are identical (std ≈ 0), std is set to
        1 to avoid division by zero.
        """

        obs_values_seen: dict[str, list] = {}
        for expt in self._expt_list:
            for obs in expt.observables:
                keep = np.logical_not(expt.expt_data["ignore_point"])
                obs_values = list(expt.expt_data.loc[keep, obs])
                if obs not in obs_values_seen:
                    obs_values_seen[obs] = []
                obs_values_seen[obs].extend(obs_values)

        self._normalization_params: dict[str, list] = {}
        for obs, raw_values in obs_values_seen.items():
            values = np.array(raw_values)
            values = values[np.logical_not(np.isnan(values))]
            if len(values) == 0:
                mean_value, std_value = 0.0, 1.0
            else:
                mean_value = float(np.mean(values))
                std_value = float(np.std(values))
                if np.isclose(std_value, 0):
                    std_value = 1.0
            self._normalization_params[obs] = [mean_value, std_value]

    def _load_observables(self) -> None:
        """
        Flatten all experimental observations into aligned numpy arrays.

        Iterates over every non-ignored (experiment, observable, point) triple
        and populates six parallel arrays:

        * ``_y_obs`` / ``_y_std`` — raw observed values and uncertainties.
        * ``_y_obs_normalized`` / ``_y_std_normalized`` — normalised values.
        * ``_y_norm_mean`` / ``_y_norm_std`` — normalisation offsets used to
          reverse the transformation.
        * ``_y_std_scalar`` — per-experiment size-balancing scalars.
        """

        self._y_obs = []
        self._y_std = []
        self._y_obs_normalized = []
        self._y_std_normalized = []
        self._y_norm_mean = []
        self._y_norm_std = []
        self._y_std_scalar = []

        for expt_counter, expt in enumerate(self._expt_list):

            # Ensure every macrospecies the model knows about has a column
            not_in_expt = set(self._bm.macro_species) - set(expt.expt_concs.columns)
            for missing in not_in_expt:
                expt.add_expt_conc_column(new_column=missing)

            for obs in expt.observables:
                for point_idx in range(len(expt.expt_data)):

                    obs_info = expt.observables[obs]
                    expt_data = expt.expt_data.loc[expt.expt_data.index[point_idx], :]

                    if expt_data["ignore_point"]:
                        continue

                    self._y_obs.append(expt_data[obs])
                    self._y_std.append(expt_data[obs_info["std_column"]])

                    obs_mean = self._normalization_params[obs][0]
                    obs_std = self._normalization_params[obs][1]
                    y_std_scalar = self._expt_std_scalar[expt_counter]

                    self._y_norm_mean.append(obs_mean)
                    self._y_norm_std.append(obs_std)
                    self._y_std_scalar.append(y_std_scalar)

                    self._y_obs_normalized.append((self._y_obs[-1] - obs_mean) / obs_std)
                    self._y_std_normalized.append(self._y_std[-1] / obs_std * y_std_scalar)

        self._y_obs = np.array(self._y_obs)
        self._y_std = np.array(self._y_std)
        self._y_norm_mean = np.array(self._y_norm_mean)
        self._y_norm_std = np.array(self._y_norm_std)
        self._y_std_scalar = np.array(self._y_std_scalar)
        self._y_obs_normalized = np.array(self._y_obs_normalized)
        self._y_std_normalized = np.array(self._y_std_normalized)

    def _get_enthalpy_param(self) -> None:
        """
        Register enthalpy and heat-of-dilution parameters.

        Reaction enthalpies (``dH_*``) are always added — one per independent
        equilibrium after any reparameterisation rules defined in the model spec
        are applied.  Heats of dilution (``nuisance_dil_<species>``) are only
        added when at least one ITC experiment is present.

        Also pre-computes:

        * ``_dh_sign`` — sign convention (+1 forward, -1 reverse) per reaction.
        * ``_dh_product_mask`` — boolean mask selecting product micro-species
          per reaction, used when computing concentration changes.
        * ``_dh_stoich_weight`` — stoichiometric weight per reaction derived
          from the analyte mass balance (accounts for statistical factors in
          multi-site models).
        * ``_extent_matrix`` — pseudo-inverse of the stoichiometric matrix N,
          used to recover true reaction extents from observed concentration
          changes in cascaded reactions.
        """

        has_itc = any(
            obs_info["type"] == "itc"
            for expt in self._expt_list
            for obs_info in expt.observables.values()
        )

        self._dh_param_start_idx = len(self._parameter_names)

        self._dh_sign = []
        self._dh_product_mask = []
        self._dh_name_map = {}

        # Collect any reparameterisation rules for dH parameters from the model
        dh_reparam_rules = {}
        if hasattr(self._bm, "reparam_rules"):
            dh_reparam_rules = {
                s.name: getattr(e, "name", str(e))
                for s, e in self._bm.reparam_rules.items()
                if s.name.startswith("dH_")
            }

        original_dh_names = [f"dH_{k[1:]}" for k in self._bm.equilibria]
        dependent_dh_names = set(dh_reparam_rules.keys())
        potential_dh_params = sorted(
            name for name in original_dh_names if name not in dependent_dh_names
        )

        for name in potential_dh_params:
            if name not in self._parameter_names:
                self._parameter_names.append(name)
                self._parameter_guesses[name] = 0.0

        for k in self._bm.equilibria:
            dh_name = f"dH_{k[1:]}"
            self._dh_name_map[k] = dh_reparam_rules.get(dh_name, dh_name)

        # Build stoichiometric weight lookup from the first (analyte) mass balance
        first_macro = list(self._bm._bm.physical_poly._species.keys())[0]
        mb_micros, mb_stoichs = self._bm._bm.physical_poly._species[first_macro]
        stoich_lookup = dict(zip(mb_micros, mb_stoichs))

        self._dh_stoich_weight = []
        for k in self._bm.equilibria:
            reactants, products = self._bm.equilibria[k]
            if len(products) <= len(reactants):
                self._dh_sign.append(1.0)
                key_species = products[:]
            else:
                self._dh_sign.append(-1.0)
                key_species = reactants[:]
            self._dh_product_mask.append(np.isin(self._bm.micro_species, key_species))
            weight = float(np.mean([stoich_lookup.get(s, 1) for s in key_species]))
            self._dh_stoich_weight.append(weight)

        # Build stoichiometric extent-recovery matrix M = pinv(N).
        # N[i, j] captures how the product species of reaction i appear as
        # reactants of reaction j.  Solving N @ xi = delta_c gives the true
        # reaction extents from the observed concentration changes, correctly
        # disentangling cascaded reactions.
        n_rxn = len(self._dh_product_mask)
        N = np.zeros((n_rxn, n_rxn), dtype=float)
        micro_species_arr = np.array(self._bm.micro_species)
        equil_items = list(self._bm.equilibria.items())
        for i in range(n_rxn):
            N[i, i] = 1.0
            product_species_i = set(micro_species_arr[self._dh_product_mask[i]])
            for j, (_, (reactants, _products)) in enumerate(equil_items):
                if j == i:
                    continue
                overlap = sum(1 for sp in reactants if sp in product_species_i)
                if overlap:
                    N[i, j] -= float(overlap)
        self._extent_matrix = np.linalg.pinv(N)

        # Heats of dilution — one per titrating species, only when ITC is present
        self._dh_dilution_idx_map: dict[str, int] = {}
        if has_itc:
            to_dilute = sorted(set(
                s
                for e in self._expt_list
                for o in e.observables
                if e.observables[o]["type"] == "itc"
                for s in e.titrating_macro_species
            ))
            for s in to_dilute:
                param_name = f"nuisance_dil_{s}"
                if param_name not in self._parameter_names:
                    self._parameter_names.append(param_name)
                    self._parameter_guesses[param_name] = 0.0
                self._dh_dilution_idx_map[s] = self._parameter_names.index(param_name)

        self._dh_param_end_idx = len(self._parameter_names) - 1

    def _get_expt_fudge(self) -> None:
        """
        Register per-experiment concentration fudge parameters.

        For each experiment whose ``conc_to_float`` attribute is set, adds a
        ``nuisance_expt_<i>_<species>_fudge`` multiplicative scaling parameter
        to the parameter vector.  The fudge factor allows the total concentration
        of that species to float during the fit, accommodating uncertainty in
        pipetted amounts.
        """

        self._fudge_list = []
        for i, expt in enumerate(self._expt_list):
            if expt.conc_to_float:
                param_name = f"nuisance_expt_{i}_{expt.conc_to_float}_fudge"
                if param_name not in self._parameter_names:
                    self._parameter_names.append(param_name)
                    self._parameter_guesses[param_name] = 1.0

                fudge_species_idx = np.where(
                    self._bm.macro_species == expt.conc_to_float)[0][0]
                self._fudge_list.append(
                    (fudge_species_idx, self._parameter_names.index(param_name)))
            else:
                self._fudge_list.append(None)

    def _add_point(self, point_idx: int, expt_idx: int, obs: str) -> None:
        """
        Construct and append a single observation point to ``_points``.

        Dispatches to ``ITCPoint`` or ``SpecPoint`` depending on the observable
        type.  Silently skips points flagged as ``ignore_point``.

        Parameters
        ----------
        point_idx : int
            Row index within the experiment's data array.
        expt_idx : int
            Index of the parent experiment in ``_expt_list``.
        obs : str
            Observable key (column name) within the experiment.

        Raises
        ------
        ValueError
            If the observable type is not ``'itc'`` or ``'spec'``.
        """

        expt = self._expt_list[expt_idx]
        obs_info = expt.observables[obs]
        data_idx = expt.expt_data.index[point_idx]
        if expt.expt_data.loc[data_idx, "ignore_point"]:
            return

        point_kwargs = {
            "idx": point_idx,
            "expt_idx": expt_idx,
            "obs_key": obs,
            "micro_array": self._micro_arrays[-1],
            "macro_array": self._macro_arrays[-1],
            "del_macro_array": self._del_macro_arrays[-1],
            "total_volume": float(expt.expt_concs.loc[data_idx, "volume"]),
            "injection_volume": float(expt.expt_data.loc[data_idx, "injection"]),
        }

        if obs_info["type"] == "spec":
            point_kwargs["obs_mask"] = np.isin(
                self._bm.micro_species, obs_info["microspecies"])
            point_kwargs["denom"] = np.where(
                self._bm.macro_species == obs_info["macrospecies"])[0][0]
            pt = SpecPoint(**point_kwargs)

        elif obs_info["type"] == "itc":
            local_dilution_idx = [
                self._dh_dilution_idx_map[s]
                for s in self._bm.macro_species
                if s in expt.titrating_macro_species and s in self._dh_dilution_idx_map
            ]
            point_kwargs.update({
                "dh_sign": self._dh_sign,
                "dh_product_mask": self._dh_product_mask,
                "dh_dilution_idx": local_dilution_idx,
                "titrating_species_mask": np.array(
                    [s in expt.titrating_macro_species for s in self._bm.macro_species]),
                "extent_matrix": self._extent_matrix,
                "stoich_weight": self._dh_stoich_weight,
            })
            pt = ITCPoint(**point_kwargs)

        else:
            raise ValueError(f"Observable type '{obs_info['type']}' not recognised.")

        self._points.append(pt)

    def _build_point_map(self) -> None:
        """
        Build per-experiment concentration arrays and the flat ``_points`` list.

        For each experiment, allocates:

        * ``_micro_arrays`` — (n_points × n_micro) array, filled by ``model()``.
        * ``_macro_arrays`` — (n_points × n_macro) array of total concentrations.
        * ``_ref_macro_arrays`` — unmodified copy used to reset after fudge factors.
        * ``_del_macro_arrays`` — (syringe - cell) concentration differences.
        * ``_expt_syringe_concs`` — syringe concentrations in macro-species order.

        Then iterates over all (experiment, observable, point) triples and
        calls ``_add_point`` to populate ``_points``.
        """

        self._ref_macro_arrays = []
        self._macro_arrays = []
        self._micro_arrays = []
        self._del_macro_arrays = []
        self._expt_syringe_concs = []
        self._points = []

        for i, expt in enumerate(self._expt_list):
            self._micro_arrays.append(
                np.full((len(expt.expt_data), len(self._bm.micro_species)), np.nan))

            macro_array = np.zeros((len(expt.expt_data), len(self._bm.macro_species)))
            for j, species in enumerate(self._bm.macro_species):
                macro_array[:, j] = expt.expt_concs[species].values
            self._ref_macro_arrays.append(macro_array)
            self._macro_arrays.append(macro_array.copy())

            syringe_concs = np.array(
                [expt.syringe_contents.get(s, 0.0) for s in self._bm.macro_species])
            self._expt_syringe_concs.append(syringe_concs)
            self._del_macro_arrays.append(syringe_concs - macro_array)

            for obs in expt.observables:
                for j in range(len(expt.expt_data)):
                    self._add_point(point_idx=j, expt_idx=i, obs=obs)

    # ---------------------------------------------------------------------- #
    # Public model interface                                                   #
    # ---------------------------------------------------------------------- #

    def model(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute model predictions for all observation points.

        Converts the regression parameter vector to physical quantities
        (binding constants in linear space, enthalpies in cal/mol), solves for
        micro-species concentrations at every injection point of every
        experiment, then evaluates each registered observation point.

        Binding-constant parameters are stored and passed to the model in
        log-space (i.e. ``K_linear = exp(param)``).  Enthalpy and nuisance
        parameters are stored in kcal/mol and rescaled to cal/mol internally.

        Parameters
        ----------
        parameters : array-like of float
            Full regression parameter vector in the order defined by
            ``parameter_names``.

        Returns
        -------
        numpy.ndarray
            Predicted values, one per non-ignored observation point, in the
            same order as ``y_obs``.
        """

        start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1

        parameters_internal = np.atleast_1d(np.array(parameters, dtype=float))

        if len(parameters_internal) < len(self._parameter_names):
            missing_count = len(self._parameter_names) - len(parameters_internal)
            parameters_internal = np.concatenate(
                [parameters_internal, np.zeros(missing_count)])

        # Scale enthalpy and dilution parameters from kcal/mol to cal/mol
        for i, name in enumerate(self._parameter_names):
            if name.startswith("dH") or "nuisance_dil" in name:
                parameters_internal[i] *= self._dh_scale

        # Build linearised parameter dict for the binding model
        bm_params_prepared = {}
        for i in range(start, end):
            p_name = self._parameter_names[i]
            val = parameters_internal[i]
            if p_name.startswith("dH") or "nuisance" in p_name:
                bm_params_prepared[p_name] = val
            else:
                bm_params_prepared[p_name] = np.exp(val)

        phys_params = self._bm.get_physical_params(bm_params_prepared)

        # Collect dH values in equilibrium order, respecting reparameterisation
        full_dh_array = np.zeros(len(self._bm.equilibria))
        for i, k_name in enumerate(self._bm.equilibria):
            dh_name = f"dH_{k_name[1:]}"
            val = phys_params.get(dh_name, None)
            if val is None:
                val = parameters_internal[self._parameter_names.index(dh_name)] \
                    if dh_name in self._parameter_names else 0.0
            full_dh_array[i] = val

        bm_param_array = np.array(
            [bm_params_prepared[self._parameter_names[i]] for i in range(start, end)])

        for i in range(len(self._macro_arrays)):
            self._macro_arrays[i] = self._ref_macro_arrays[i].copy()
            if self._fudge_list[i] is not None:
                fudge_species_idx, fudge_param_idx = self._fudge_list[i]
                self._macro_arrays[i][:, fudge_species_idx] *= parameters_internal[fudge_param_idx]

            self._del_macro_arrays[i] = (
                self._expt_syringe_concs[i] - self._macro_arrays[i])

            for j in range(len(self._macro_arrays[i])):
                self._micro_arrays[i][j, :] = self._bm.get_concs(
                    param_array=bm_param_array,
                    macro_array=self._macro_arrays[i][j, :])

        y_calc = np.full(len(self._points), np.nan)
        for i, pt in enumerate(self._points):
            if isinstance(pt, ITCPoint):
                y_calc[i] = pt.calc_value(parameters_internal, full_dh_array=full_dh_array)
            else:
                y_calc[i] = pt.calc_value(parameters_internal)

        self._model_state_params = np.array(parameters, dtype=float).copy()

        return y_calc

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """Alias for ``model(parameters)``."""
        return self.model(parameters)

    def model_normalized(self, parameters: np.ndarray) -> np.ndarray:
        """
        Return normalised model predictions.

        Applies the same ``(value - mean) / std`` normalisation used to
        construct ``y_obs_normalized``, so that the output can be directly
        compared to ``y_obs_normalized`` or passed to a normalised residual
        function.

        Parameters
        ----------
        parameters : array-like of float
            Full regression parameter vector.

        Returns
        -------
        numpy.ndarray
            Normalised predictions aligned with ``y_obs_normalized``.
        """

        y_calc = self.model(parameters)
        if np.all(np.isclose(self._y_norm_std, 0)):
            return y_calc - self._y_norm_mean
        return (y_calc - self._y_norm_mean) / self._y_norm_std

    def jacobian_normalized(self, parameters: np.ndarray) -> np.ndarray:
        """
        Compute the analytical Jacobian of the normalised model output.

        Returns ``d(y_calc_normalized) / d(parameters)`` using the symbolic
        Jacobian compiled by ``BindingModel``.  Skips re-running
        ``model()`` if the parameter vector is unchanged since the last call.

        The chain rule is applied to account for:

        * Log-space binding constants (``d/d(log K) = K * d/d(K)``).
        * Enthalpy scaling (kcal/mol ↔ cal/mol).
        * Reparameterisation of dependent enthalpies.
        * ITC-specific cross-shot concentration differences.

        Fudge-factor columns are evaluated by finite difference, as these
        parameters do not affect micro-species concentrations through the
        binding model Jacobian directly.

        Parameters
        ----------
        parameters : array-like of float
            Full regression parameter vector.

        Returns
        -------
        numpy.ndarray, shape (n_obs, n_params)
            Jacobian matrix.  Returns an all-NaN matrix on failure, with a
            warning indicating the cause.
        """

        try:
            params_arr = np.atleast_1d(np.array(parameters, dtype=float))
            if (self._model_state_params is None
                    or len(self._model_state_params) != len(params_arr)
                    or not np.array_equal(self._model_state_params, params_arr)):
                self.model(parameters)

            num_obs = len(self._points)
            num_params = len(self.parameter_names)
            J = np.zeros((num_obs, num_params))
            start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1

            # Linearised parameter dict
            bm_params_prepared = {}
            for i in range(start, end):
                p_name = self._parameter_names[i]
                val = parameters[i]
                if p_name.startswith("dH") or "nuisance_dil" in p_name:
                    bm_params_prepared[p_name] = val * self._dh_scale
                elif "nuisance" in p_name:
                    bm_params_prepared[p_name] = val
                else:
                    bm_params_prepared[p_name] = np.exp(val)

            phys_params = self._bm.get_physical_params(bm_params_prepared)
            d_phys_d_reg = self._bm.get_physical_jacobian(bm_params_prepared).astype(float)
            phys_param_names = self._bm.physical_param_names

            # Apply chain rule for log-transformed K parameters
            for i, p_name in enumerate(self._parameter_names[start:end]):
                if not (p_name.startswith("dH") or "nuisance" in p_name):
                    d_phys_d_reg[:, i] *= bm_params_prepared[p_name]

            # Collect dH values and their derivatives w.r.t. regression params
            full_dh_array = np.zeros(len(self._bm.equilibria))
            d_dh_d_reg_list = []
            for i, k_name in enumerate(self._bm.equilibria):
                dh_name = f"dH_{k_name[1:]}"
                full_dh_array[i] = phys_params.get(dh_name, 0.0)
                if dh_name in phys_param_names:
                    idx = phys_param_names.index(dh_name)
                    d_dh_d_reg_list.append(d_phys_d_reg[idx, :])
                else:
                    d_dh_d_reg_list.append(np.zeros(d_phys_d_reg.shape[1]))

            # Concentration Jacobians via the symbolic path
            n_micro = len(self._bm.micro_species)
            n_bm_params = len(self._bm.param_names)
            d_concs_d_bm_params_list = []

            for i in range(len(self._expt_list)):
                exp_jacobians = []
                for j in range(len(self._macro_arrays[i])):
                    all_concs_dict = {
                        **bm_params_prepared,
                        **dict(zip(self._bm.macro_species, self._macro_arrays[i][j, :])),
                        **dict(zip(self._bm.micro_species, self._micro_arrays[i][j, :])),
                    }
                    jac = self._bm.get_numerical_jacobian(all_concs_dict)
                    if jac is None or np.any(np.isnan(jac)):
                        jac = np.full((n_micro, n_bm_params), np.nan)
                    exp_jacobians.append(jac)
                d_concs_d_bm_params_list.append(exp_jacobians)

            # Assemble full Jacobian row by row
            for i, pt in enumerate(self._points):
                expt_idx, shot_idx = pt.expt_idx, pt.idx
                d_concs_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx]

                if isinstance(pt, SpecPoint):
                    J[i, start:end] = pt.get_d_y_d_concs() @ d_concs_d_bm

                elif isinstance(pt, ITCPoint) and pt.idx > 0:
                    d_concs_before_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx - 1]
                    num_eq = len(pt._dh_product_mask)

                    d_dC_list = []
                    delta_c = np.empty(num_eq)
                    for j in range(num_eq):
                        mask = pt._dh_product_mask[j]
                        w = pt._stoich_weight[j]
                        d_C_after = d_concs_d_bm[mask, :]
                        d_C_before = d_concs_before_d_bm[mask, :]
                        d_del_C = d_C_after - d_C_before * pt._meas_vol_dilution
                        d_dC_list.append(w * np.mean(d_del_C, axis=0))

                        C_before = pt._micro_array[shot_idx - 1, mask]
                        C_after = pt._micro_array[shot_idx, mask]
                        delta_c[j] = w * np.mean(
                            C_after - C_before * pt._meas_vol_dilution)

                    M = pt._extent_matrix
                    extents = M @ delta_c
                    d_delta_c_mat = np.vstack(d_dC_list)
                    d_extents_mat = M @ d_delta_c_mat

                    signed_dh = full_dh_array[:num_eq] * np.array(pt._dh_sign)
                    term1 = signed_dh @ d_extents_mat
                    d_dh_d_reg_mat = np.vstack(d_dh_d_reg_list[:num_eq])
                    term2 = (d_dh_d_reg_mat
                             * (np.array(pt._dh_sign) * extents)[:, None]).sum(axis=0)
                    J[i, start:end] = (term1 + term2) * pt._total_volume

                other_param_derivs = pt.get_d_y_d_other_params(
                    parameters, full_dh_array=full_dh_array)
                for param_idx, deriv_val in other_param_derivs.items():
                    J[i, param_idx] = deriv_val

            # Finite difference for fudge-factor columns
            fudge_param_indices = {
                item[1] for item in self._fudge_list if item is not None}

            if fudge_param_indices:
                y_center = self.model(parameters)
                eps = 1e-8
                for p_idx in fudge_param_indices:
                    p_orig = parameters[p_idx]
                    step = eps * max(abs(p_orig), 1.0)
                    p_new = parameters.copy()
                    p_new[p_idx] += step
                    J[:, p_idx] = (self.model(p_new) - y_center) / step
                self.model(parameters)  # restore model state

            # Normalise Jacobian rows and scale enthalpy columns
            if np.any(~np.isclose(self._y_norm_std, 0)):
                J /= self._y_norm_std[:, np.newaxis]

            for i, p_name in enumerate(self._parameter_names):
                if p_name.startswith("dH") or "nuisance_dil" in p_name:
                    J[:, i] *= self._dh_scale

            return J

        except Exception as e:
            tb_str = traceback.format_exc()
            warnings.warn(f"Jacobian calculation failed: {e}\n{tb_str}")
            return np.full((len(self._points), len(self.parameter_names)), np.nan)

    def hessian_normalized(self, parameters: np.ndarray) -> np.ndarray:
        """
        Gauss-Newton approximation to the normalised negative log-likelihood Hessian.

        Computed as ``J_w.T @ J_w`` where ``J_w = jacobian_normalized / y_std_normalized``.
        This is the optimal HMC mass matrix (precision matrix) evaluated at
        ``parameters``, exact at the MAP where residuals ≈ 0.

        Parameters
        ----------
        parameters : array-like of float
            Full regression parameter vector.

        Returns
        -------
        numpy.ndarray, shape (n_params, n_params)
            Approximate Hessian.  Falls back to the identity matrix on
            numerical failure.
        """

        try:
            J = self.jacobian_normalized(parameters)
            J_w = J / self._y_std_normalized[:, np.newaxis]
            return J_w.T @ J_w
        except Exception as e:
            warnings.warn(f"hessian_normalized failed: {e}")
            return np.eye(len(self._parameter_names))

    def generate_data(self,
                      shot_volume: float,
                      num_shots: int,
                      cell_contents: dict,
                      syringe_contents: dict,
                      cell_volume: float = 201.3,
                      noise_std: float = 0.0,
                      num_files: int = 1,
                      output_dir: str = ".",
                      file_prefix: str = "simulated_data",
                      data_type: str = "itc",
                      obs_column: str | None = None,
                      obs_microspecies: list | None = None,
                      obs_macrospecies: str | None = None) -> None:
        """
        Generate synthetic experimental data and write to CSV files.

        Constructs a temporary ``Experiment`` and ``GlobalModel`` using the
        current model spec and parameter guesses, then evaluates the model to
        produce noiseless (or noisy) simulated data.

        Parameters
        ----------
        shot_volume : float
            Volume of each injection in µL.
        num_shots : int
            Number of injections.
        cell_contents : dict
            Initial cell concentrations in mol/L (e.g. ``{"AT": 50e-6}``).
        syringe_contents : dict
            Syringe concentrations in mol/L (e.g. ``{"CT": 350e-6}``).
        cell_volume : float, default 201.3
            Cell volume in µL. Default is the MicroCal ITC200 cell volume.
        noise_std : float, default 0.0
            Standard deviation of Gaussian noise added to the signal.
            Units: µCal for ``data_type="itc"``; signal units for
            ``data_type="spec"``.
        num_files : int, default 1
            Number of replicate output files.
        output_dir : str, default "."
            Directory in which to write output CSV files.
        file_prefix : str, default "simulated_data"
            Filename prefix; files are named ``<prefix>_1.csv``,
            ``<prefix>_2.csv``, etc.
        data_type : {"itc", "spec"}, default "itc"
            Observable type to simulate.  ITC output columns:
            ``injection``, ``heat``, ``heat_stdev``, ``ignore_point``.
            Spectroscopic output columns: ``injection``, ``<obs_column>``,
            ``<obs_column>_std``, ``ignore_point``.
        obs_column : str, optional
            Column name for the spectroscopic observable.  Only used when
            ``data_type="spec"``; defaults to ``"signal"``.
        obs_microspecies : list of str, optional
            Micro-species forming the numerator of the spectroscopic signal.
            Required when ``data_type="spec"``.
        obs_macrospecies : str, optional
            Macro-species used as the denominator of the spectroscopic signal.
            Required when ``data_type="spec"``.

        Raises
        ------
        ValueError
            If ``data_type`` is not ``"itc"`` or ``"spec"``, if required
            spectroscopic arguments are missing, or if any species in
            ``cell_contents`` or ``syringe_contents`` is not in the model.
        """

        data_type = data_type.lower()
        if data_type not in ("itc", "spec"):
            raise ValueError(f"data_type must be 'itc' or 'spec', got '{data_type}'")

        if data_type == "spec":
            if obs_microspecies is None:
                raise ValueError("obs_microspecies must be provided when data_type='spec'")
            if obs_macrospecies is None:
                raise ValueError("obs_macrospecies must be provided when data_type='spec'")
            if obs_column is None:
                obs_column = "signal"

        for s in list(cell_contents.keys()) + list(syringe_contents.keys()):
            if s not in self.macro_species:
                raise ValueError(
                    f"Species '{s}' not found in model macro species: {self.macro_species}")

        injections = [0.0] + [shot_volume] * num_shots
        design_df = pd.DataFrame({"injection": injections})
        design_df_liters = design_df.copy()
        design_df_liters["injection"] *= 1e-6

        if data_type == "itc":
            design_df_liters["heat"] = 0.0
            temp_expt = Experiment(expt_data=design_df_liters,
                                   cell_contents=cell_contents,
                                   syringe_contents=syringe_contents,
                                   cell_volume=cell_volume * 1e-6)
            temp_expt.define_itc_observable(obs_column="heat", obs_std=1.0)
        else:
            design_df_liters[obs_column] = 0.0
            temp_expt = Experiment(expt_data=design_df_liters,
                                   cell_contents=cell_contents,
                                   syringe_contents=syringe_contents,
                                   cell_volume=cell_volume * 1e-6)
            temp_expt.define_spectroscopic_observable(
                obs_column=obs_column,
                obs_std=1.0,
                obs_microspecies=obs_microspecies,
                obs_macrospecies=obs_macrospecies)

        temp_gm = GlobalModel(expt_list=[temp_expt],
                              model_spec=self._model_spec,
                              verbose=False)

        current_guesses = self.parameter_guesses
        param_values = []
        for p_name in temp_gm.parameter_names:
            if p_name in current_guesses:
                param_values.append(current_guesses[p_name])
            else:
                warnings.warn(
                    f"generate_data: parameter '{p_name}' not found in "
                    f"parameter_guesses, using default 0.0")
                param_values.append(0.0)

        raw_values = temp_gm.model(np.array(param_values))
        simulated_values = raw_values * 1e6 if data_type == "itc" else raw_values

        os.makedirs(output_dir, exist_ok=True)
        obs_col_name = "heat" if data_type == "itc" else obs_column
        std_col_name = "heat_stdev" if data_type == "itc" else f"{obs_column}_std"

        for i in range(num_files):
            noisy_values = simulated_values + np.random.normal(
                0, noise_std, size=len(simulated_values))

            out_df = design_df.copy()
            out_df[obs_col_name] = np.nan
            out_df[std_col_name] = noise_std
            out_df["ignore_point"] = False

            for pt_idx, pt in enumerate(temp_gm._points):
                out_df.loc[pt.idx, obs_col_name] = noisy_values[pt_idx]

            if len(out_df) > 0 and np.isclose(out_df.iloc[0]["injection"], 0.0):
                out_df = out_df.iloc[1:]

            out_df = out_df[["injection", obs_col_name, std_col_name, "ignore_point"]]
            out_df.to_csv(os.path.join(output_dir, f"{file_prefix}_{i+1}.csv"),
                          index=False)

        print(f"Generated {num_files} simulated {data_type.upper()} "
              f"datasets in '{output_dir}'")

    def calculate_derived_params(self,
                                 estimate: np.ndarray | None = None,
                                 cov: np.ndarray | None = None,
                                 samples: np.ndarray | None = None,
                                 dof: int | None = None) -> pd.DataFrame | None:
        """
        Compute derived (physical) parameters and their uncertainties.

        Supports two modes:

        **Bayesian mode** (pass ``samples``)
            Each sample row is mapped through ``get_physical_params`` to
            produce a distribution of physical parameters.  Mean, std, and
            2.5/97.5 percentile credible intervals are reported.

        **Frequentist mode** (pass ``estimate`` and ``cov``)
            Physical parameter estimates and uncertainties are computed by
            propagating the regression covariance through the physical-parameter
            Jacobian.  Equilibrium constants are reported in log-space using
            the delta method.

        Parameters
        ----------
        estimate : numpy.ndarray, optional
            MAP or MLE regression parameter vector.  Required for frequentist
            mode.
        cov : numpy.ndarray, optional
            Covariance matrix of ``estimate``.  Required for frequentist mode.
        samples : numpy.ndarray, optional
            Posterior samples of shape ``(n_samples, n_params)``.  Required
            for Bayesian mode.
        dof : int, optional
            Degrees of freedom for the t-distribution used in frequentist
            confidence intervals.  If not provided, a z-score of 1.96 is used.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with columns ``name``, ``estimate``, ``std``,
            ``low_95``, ``high_95``, ``fixed``, ``guess``,
            ``lower_bound``, ``upper_bound``, ``prior_mean``,
            ``prior_std``.  Returns ``None`` if the model has no physical
            parameters or if neither ``samples`` nor ``estimate`` is provided.
        """

        if not hasattr(self._bm, "physical_param_names"):
            return None

        phys_names = self._bm.physical_param_names
        if not phys_names:
            return None

        results = []
        start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1

        if samples is not None:
            # Bayesian mode: map each sample to physical params
            phys_samples = np.zeros((samples.shape[0], len(phys_names)))
            for k in range(samples.shape[0]):
                row = samples[k, :]
                bm_params_prepared = {}
                for i in range(start, end):
                    p_name = self._parameter_names[i]
                    val = row[i]
                    if p_name.startswith("dH") or "nuisance" in p_name:
                        bm_params_prepared[p_name] = val
                    else:
                        bm_params_prepared[p_name] = np.exp(val)
                phys_vals = self._bm.get_physical_params(bm_params_prepared)
                for p_idx, p_name in enumerate(phys_names):
                    phys_samples[k, p_idx] = phys_vals.get(p_name, np.nan)

            est_vec = np.mean(phys_samples, axis=0)
            std_vec = np.std(phys_samples, axis=0)
            low_vec = np.percentile(phys_samples, 2.5, axis=0)
            high_vec = np.percentile(phys_samples, 97.5, axis=0)

            for i, name in enumerate(phys_names):
                results.append({
                    "name": name,
                    "estimate": est_vec[i],
                    "std": std_vec[i],
                    "low_95": low_vec[i],
                    "high_95": high_vec[i],
                    "fixed": False,
                    "guess": np.nan,
                    "lower_bound": -np.inf,
                    "upper_bound": np.inf,
                    "prior_mean": np.nan,
                    "prior_std": np.nan,
                })

        elif estimate is not None and cov is not None:
            # Frequentist mode: uncertainty propagation
            bm_params_prepared = {}
            for i in range(len(estimate)):
                p_name = self._parameter_names[i]
                val = estimate[i]
                if p_name.startswith("dH") or "nuisance" in p_name:
                    bm_params_prepared[p_name] = val
                else:
                    try:
                        bm_params_prepared[p_name] = np.exp(val)
                    except FloatingPointError:
                        bm_params_prepared[p_name] = np.inf

            phys_vals_dict = self._bm.get_physical_params(bm_params_prepared)

            J_phys = np.zeros((len(phys_names), len(estimate)))
            d_phys_d_bm = np.array(
                self._bm.get_physical_jacobian(bm_params_prepared),
                dtype=float, copy=True)
            d_phys_d_bm.flags.writeable = True

            # Apply chain rule for log-space K parameters
            for col_idx, glob_idx in enumerate(range(start, end)):
                p_name = self._parameter_names[glob_idx]
                if not (p_name.startswith("dH") or "nuisance" in p_name):
                    d_phys_d_bm[:, col_idx] *= bm_params_prepared[p_name]
            J_phys[:, start:end] = d_phys_d_bm

            Cov_phys = J_phys @ cov @ J_phys.T
            var_phys = np.diagonal(Cov_phys).copy()
            var_phys[var_phys < 0] = 0.0
            std_phys = np.sqrt(var_phys)

            def _is_log_param(name: str) -> bool:
                return not (name.startswith("dH") or "nuisance" in name)

            tcrit = 1.96
            if dof is not None and dof > 0:
                import scipy.stats
                tcrit = scipy.stats.t.ppf(0.975, dof)

            final_estimates = []
            final_stds = []
            for i, name in enumerate(phys_names):
                val = phys_vals_dict[name]
                std = std_phys[i]
                if _is_log_param(name):
                    if val > 0:
                        val_log = np.log(val)
                        std_log = std / val
                    else:
                        val_log = np.nan
                        std_log = np.nan
                    final_estimates.append(val_log)
                    final_stds.append(std_log)
                else:
                    final_estimates.append(val)
                    final_stds.append(std)

            final_estimates = np.array(final_estimates)
            final_stds = np.array(final_stds)
            low_phys = final_estimates - tcrit * final_stds
            high_phys = final_estimates + tcrit * final_stds

            for i, name in enumerate(phys_names):
                if name in self._parameter_names:
                    continue
                results.append({
                    "name": name,
                    "estimate": final_estimates[i],
                    "std": final_stds[i],
                    "low_95": low_phys[i],
                    "high_95": high_phys[i],
                    "guess": np.nan,
                    "fixed": False,
                    "lower_bound": -np.inf,
                    "upper_bound": np.inf,
                    "prior_mean": np.nan,
                    "prior_std": np.nan,
                })

        if results:
            df = pd.DataFrame(results)
            df.index = df["name"]
            return df

        return None

    # ---------------------------------------------------------------------- #
    # Properties                                                               #
    # ---------------------------------------------------------------------- #

    @property
    def y_obs(self) -> np.ndarray:
        """Raw observed values, flattened across all experiments and observables."""
        return self._y_obs

    @property
    def y_std(self) -> np.ndarray:
        """Raw observation uncertainties, aligned with ``y_obs``."""
        return self._y_std

    @property
    def y_obs_normalized(self) -> np.ndarray:
        """Normalised observed values ``(y - mean) / std``."""
        return self._y_obs_normalized

    @property
    def y_std_normalized(self) -> np.ndarray:
        """Normalised uncertainties, scaled by the experiment size scalar."""
        return self._y_std_normalized

    @property
    def parameter_names(self) -> list[str]:
        """Ordered list of all fittable parameter names."""
        return self._parameter_names

    @property
    def parameter_guesses(self) -> dict[str, float]:
        """Initial guesses for all parameters, keyed by name."""
        return self._parameter_guesses

    @property
    def parameter_df(self) -> pd.DataFrame:
        """Parameter guesses as a single-column DataFrame, indexed by name."""
        df = pd.DataFrame(
            list(self._parameter_guesses.items()), columns=["name", "guess"])
        df.set_index("name", inplace=True)
        return df

    @property
    def physical_parameter_names(self) -> list[str]:
        """Physical parameter names as reported by the binding model."""
        if hasattr(self._bm, "physical_param_names"):
            return self._bm.physical_param_names
        return []

    @property
    def macro_species(self) -> np.ndarray:
        """Macro-species names from the binding model."""
        return self._bm.macro_species

    @property
    def micro_species(self) -> np.ndarray:
        """Micro-species names from the binding model."""
        return self._bm.micro_species

    @property
    def model_spec(self) -> str | None:
        """The model specification string, if available."""
        return getattr(self._bm, "model_spec", None)

    @property
    def simplified_equations(self):
        """Simplified binding equations from the symbolic solver, if available."""
        return getattr(self._bm, "simplified_eqs", None)

    @property
    def solved_vars(self):
        """Solved variables from the symbolic solver, if available."""
        return getattr(self._bm, "solved_vars", None)

    @property
    def final_ct(self):
        """Final concentration table from the binding model, if available."""
        return getattr(self._bm, "final_ct", None)

    @property
    def as_df(self) -> pd.DataFrame:
        """
        All observation points as a DataFrame for inspection and plotting.

        Columns include experiment index, type, observable description,
        volume, injection size, all macro- and micro-species concentrations,
        and the raw and normalised observed values and uncertainties.
        """

        out: dict = {
            "expt_id": [], "expt_type": [], "expt_obs": [],
            "volume": [], "injection": [],
        }
        for k in self._bm.macro_species:
            out[k] = []
        for k in self._bm.micro_species:
            out[k] = []

        for p in self._points:
            out["expt_id"].append(p.expt_idx)
            if isinstance(p, SpecPoint):
                out["expt_type"].append(p.obs_key)
                num = "+".join(
                    s for s_idx, s in enumerate(self._bm.micro_species)
                    if p._obs_mask[s_idx])
                den = self._bm.macro_species[p._denom]
                out["expt_obs"].append(f"{num}/{den}")
            elif isinstance(p, ITCPoint):
                out["expt_type"].append("itc")
                out["expt_obs"].append("obs_heat")
            else:
                raise ValueError("Point class not recognised.")

            out["volume"].append(p._total_volume)
            out["injection"].append(p._injection_volume)
            for i, k in enumerate(self._bm.macro_species):
                out[k].append(p._macro_array[p._idx, i])
            for i, k in enumerate(self._bm.micro_species):
                out[k].append(p._micro_array[p._idx, i])

        out["y_obs"] = self.y_obs
        out["y_std"] = self.y_std
        out["y_obs_norm"] = self.y_obs_normalized
        out["y_std_norm"] = self.y_std_normalized

        return pd.DataFrame(out)

    @property
    def concentrations_df(self) -> pd.DataFrame:
        """Concentration table from the binding model, if available."""
        return self._bm.concentrations_df
