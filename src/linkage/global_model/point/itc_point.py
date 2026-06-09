from linkage.global_model.point.experimental_point import ExperimentalPoint

import numpy as np


class ITCPoint(ExperimentalPoint):
    """
    A single ITC injection heat observation point.

    The predicted heat for injection ``i`` is::

        Q_i = sum_j( dH_j * sign_j * extent_j ) * total_volume
              + sum_k( dil_k * delta_macro_k ) * injection_volume

    where the extents ``xi`` are recovered from the observed changes in
    micro-species concentrations via the stoichiometric extent-recovery
    matrix, and the second term accounts for heats of dilution of each
    titrating species.

    The full Jacobian of this observable with respect to binding-constant
    and enthalpy parameters is computed directly in
    ``GlobalModel.jacobian_normalized`` rather than here, because it depends
    on concentration states from both the current and the preceding injection
    point.
    """

    def __init__(self,
                 idx: int,
                 expt_idx: int,
                 obs_key: str,
                 micro_array: np.ndarray,
                 macro_array: np.ndarray,
                 del_macro_array: np.ndarray,
                 total_volume: float,
                 injection_volume: float,
                 dh_sign: list,
                 dh_product_mask: list,
                 dh_dilution_idx: list,
                 titrating_species_mask: np.ndarray,
                 extent_matrix: np.ndarray,
                 stoich_weight: list | None = None):
        """
        Initialise an ITC observation point.

        Parameters
        ----------
        idx : int
            Row index of this point within the experiment's data array.
        expt_idx : int
            Index of the parent experiment in ``GlobalModel._expt_list``.
        obs_key : str
            Column name of the observable in the experiment's DataFrame.
        micro_array : numpy.ndarray, shape (n_points, n_micro)
            Shared array of micro-species concentrations.
        macro_array : numpy.ndarray, shape (n_points, n_macro)
            Shared array of total macro-species concentrations.
        del_macro_array : numpy.ndarray, shape (n_points, n_macro)
            Shared array of (syringe − cell) concentration differences.
        total_volume : float
            Total cell volume (L) at this injection point.
        injection_volume : float
            Volume (L) of the injection that produced this point.
        dh_sign : list of float
            Sign convention for each reaction's enthalpy contribution:
            ``+1`` if the reaction proceeds forward (products increase),
            ``-1`` if reverse.
        dh_product_mask : list of numpy.ndarray of bool
            One boolean mask per reaction selecting the micro-species whose
            concentration change tracks the reaction extent.
        dh_dilution_idx : list of int
            Indices into the global parameter vector for the
            ``nuisance_dil_*`` heat-of-dilution parameters, one per
            titrating species in this experiment.
        titrating_species_mask : numpy.ndarray of bool
            Boolean mask of length ``n_macro`` indicating which macro-species
            are actively titrated in this experiment.
        extent_matrix : numpy.ndarray, shape (n_rxn, n_rxn)
            Stoichiometric extent-recovery matrix ``M = pinv(N)``.  Solves
            ``N @ xi = delta_c`` for the true reaction extents ``xi``, correctly
            disentangling cascaded reactions where an intermediate is both
            produced by one reaction and consumed by another.
        stoich_weight : list of float, optional
            Mean stoichiometric weight for each reaction's product species,
            derived from the analyte mass balance.  Accounts for statistical
            degeneracy in multi-site models (e.g. a two-site symmetric model
            where the same species appears with coefficient 2).  Defaults to
            1.0 for all reactions if not provided.
        """

        super().__init__(idx=idx,
                         expt_idx=expt_idx,
                         obs_key=obs_key,
                         micro_array=micro_array,
                         macro_array=macro_array,
                         del_macro_array=del_macro_array,
                         total_volume=total_volume,
                         injection_volume=injection_volume)

        self._dh_sign = dh_sign
        self._dh_product_mask = dh_product_mask
        self._dh_dilution_idx = dh_dilution_idx
        self._titrating_species_mask = titrating_species_mask
        self._extent_matrix = extent_matrix
        self._stoich_weight = stoich_weight if stoich_weight is not None \
            else [1.0] * len(dh_product_mask)

        # Scalar that accounts for dilution of the previous injection's
        # concentrations when computing concentration changes:
        # C_before_diluted = C_before * (1 - V_inj / V_total)
        self._meas_vol_dilution = 1 - self._injection_volume / self._total_volume

    def calc_value(self,
                   parameters: np.ndarray,
                   full_dh_array: np.ndarray | None = None,
                   **kwargs) -> float:
        """
        Compute the predicted injection heat for this point.

        Returns ``0.0`` for the first injection (``idx == 0``) since no
        heat is produced before any titrant has been added.

        The heat is computed as::

            Q = sum_j( dH_j * sign_j * extent_j ) * total_volume
                + sum_k( dil_k * delta_macro_k ) * injection_volume

        Reaction extents are recovered from the observed changes in
        micro-species concentrations via the stoichiometric extent-recovery
        matrix ``M``::

            extents = M @ delta_c

        where ``delta_c[j]`` is the weighted mean concentration change of
        the product species of reaction ``j``, corrected for dilution of
        the previous point's concentrations.

        Parameters
        ----------
        parameters : numpy.ndarray
            Full regression parameter vector.  Used to read heat-of-dilution
            values via ``dh_dilution_idx``.
        full_dh_array : numpy.ndarray, optional
            Pre-built array of enthalpy values (cal/mol) for every
            equilibrium, in equilibrium order, including any reparameterised
            dependent enthalpies.  Required — raises ``ValueError`` if not
            provided.

        Returns
        -------
        float
            Predicted heat in cal (same units as ``dH`` parameters).
        """

        if self._idx == 0:
            return 0.0

        if full_dh_array is None:
            raise ValueError("full_dh_array must be provided to ITCPoint.calc_value")

        n_rxn = len(self._dh_product_mask)
        delta_c = np.empty(n_rxn)
        for i in range(n_rxn):
            C_before = self._micro_array[self._idx - 1, self._dh_product_mask[i]]
            C_after = self._micro_array[self._idx, self._dh_product_mask[i]]
            delta_c[i] = self._stoich_weight[i] * np.mean(
                C_after - C_before * self._meas_vol_dilution)

        extents = self._extent_matrix @ delta_c

        total_heat = np.dot(
            full_dh_array[:n_rxn] * np.array(self._dh_sign), extents
        ) * self._total_volume

        if len(self._dh_dilution_idx) > 0:
            dil_heats = parameters[self._dh_dilution_idx]
            molar_change = self._del_macro_array[self._idx, self._titrating_species_mask]
            if dil_heats.shape == molar_change.shape:
                total_heat += np.sum(dil_heats * molar_change) * self._injection_volume

        return total_heat

    def get_d_y_d_concs(self) -> np.ndarray:
        """
        Placeholder derivative of heat with respect to micro-species concentrations.

        The full analytical derivative for ITC depends on concentration states
        from both the current and the preceding injection point, and is
        therefore computed directly inside ``GlobalModel.jacobian_normalized``
        rather than here.  This method returns zeros so that the standard
        chain-rule assembly in ``GlobalModel`` produces no contribution from
        this path; the correct ITC Jacobian rows are filled in separately.

        Returns
        -------
        numpy.ndarray, shape (n_micro,)
            Zero array.
        """

        return np.zeros(self._micro_array.shape[1], dtype=float)

    def get_d_y_d_other_params(self,
                               parameters: np.ndarray,
                               full_dh_array: np.ndarray | None = None,
                               **kwargs) -> dict:
        """
        Derivatives of injection heat with respect to heat-of-dilution parameters.

        Computes ``d(Q) / d(nuisance_dil_k)`` for each titrating species
        ``k``::

            d(Q) / d(dil_k) = delta_macro_k * injection_volume

        where ``delta_macro_k`` is the change in total concentration of
        species ``k`` caused by this injection.

        Parameters
        ----------
        parameters : numpy.ndarray
            Full regression parameter vector (values not used here; only
            indices from ``dh_dilution_idx`` are needed).
        full_dh_array : numpy.ndarray, optional
            Ignored.  Present for a consistent call signature with
            ``calc_value``.

        Returns
        -------
        dict
            Maps parameter index (int) → scalar derivative (float).
            Empty for ``idx == 0``.
        """

        deriv_dict = {}
        if self._idx == 0:
            return deriv_dict

        if len(self._dh_dilution_idx) > 0:
            molar_change = self._del_macro_array[self._idx, self._titrating_species_mask]
            for i, param_index in enumerate(self._dh_dilution_idx):
                if i < len(molar_change):
                    deriv_dict[param_index] = self._injection_volume * molar_change[i]

        return deriv_dict
