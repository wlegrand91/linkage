from linkage.global_model.point.experimental_point import ExperimentalPoint

import numpy as np


class SpecPoint(ExperimentalPoint):
    """
    A single spectroscopic observation point.

    The observable is modelled as the fraction of a macrospecies that
    exists in a specified set of microspecies states::

        signal = sum(micro_num) / macro_denom

    where ``micro_num`` is the sum of concentrations of the microspecies
    selected by ``obs_mask`` and ``macro_denom`` is the total concentration
    of the reference macrospecies.
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
                 obs_mask: np.ndarray,
                 denom: int):
        """
        Initialise a spectroscopic observation point.

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
        obs_mask : numpy.ndarray of bool
            Boolean mask of length ``n_micro`` selecting the micro-species
            that contribute to the numerator of the observable.
        denom : int
            Column index into ``macro_array`` for the macrospecies used as
            the denominator (total concentration normaliser).
        """

        super().__init__(idx=idx,
                         expt_idx=expt_idx,
                         obs_key=obs_key,
                         micro_array=micro_array,
                         macro_array=macro_array,
                         del_macro_array=del_macro_array,
                         total_volume=total_volume,
                         injection_volume=injection_volume)

        self._obs_mask = obs_mask
        self._denom = denom

    def calc_value(self, *args, **kwargs) -> float:
        """
        Compute the spectroscopic signal at this injection point.

        Returns ``sum(selected micro-species) / macro_denom``.  Returns
        ``nan`` if the denominator concentration is zero.

        Parameters
        ----------
        *args, **kwargs
            Ignored.  Present for a consistent call signature with other
            point types.

        Returns
        -------
        float
            Predicted fractional signal, or ``nan`` if the denominator is
            zero.
        """

        num = np.sum(self._micro_array[self._idx, self._obs_mask])
        den = self._macro_array[self._idx, self._denom]

        if den == 0:
            return np.nan
        return num / den

    def get_d_y_d_concs(self) -> np.ndarray:
        """
        Compute ``d(signal) / d(micro_species_concentrations)``.

        For ``signal = sum(micro_num) / macro_denom``, the derivative with
        respect to micro-species ``k`` is ``1 / macro_denom`` if ``k`` is
        selected by ``obs_mask``, and ``0`` otherwise.

        Returns
        -------
        numpy.ndarray, shape (n_micro,)
            Per-micro-species derivatives.  Returns zeros if the denominator
            is zero.
        """

        den = self._macro_array[self._idx, self._denom]
        if den == 0:
            return np.zeros(self._micro_array.shape[1], dtype=float)

        return self._obs_mask.astype(float) / den

    def get_d_y_d_other_params(self, parameters: np.ndarray, **kwargs) -> dict:
        """
        Derivatives with respect to non-concentration parameters.

        Spectroscopic points have no direct dependence on enthalpy or
        nuisance parameters — fudge-factor effects are captured implicitly
        through the concentration chain rule in ``GlobalModel.jacobian_normalized``.

        Parameters
        ----------
        parameters : numpy.ndarray
            Full regression parameter vector (unused).

        Returns
        -------
        dict
            Empty dictionary.
        """

        return {}
