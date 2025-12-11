from .experimental_point import ExperimentalPoint

import numpy as np

class SpecPoint(ExperimentalPoint):
    """
    Class holds experimental data for an individual spectroscopic 
    experimental data point and how it links to the binding model. 
    """

    def __init__(self,
                 idx,
                 expt_idx,
                 obs_key,
                 micro_array,
                 macro_array,
                 del_macro_array,
                 total_volume,
                 injection_volume,
                 obs_mask,
                 denom):
        """
        Initialize a spectroscopic data point. 
        
        Parameters
        ----------
        idx : int
            index of point in the experimental array
        expt_idx : int
            index of the experiment itself (allowing multiple experiments)
        obs_key : str
            key pointing to observable from the experiment
        micro_array : np.ndarray (float)
            array holding concentrations of all microscopic species, calculated
            elsewhere
        macro_array : np.ndarray (float)
            array holding concentrations of all macroscopic species, calculated
            elsewhere
        del_macro_array : np.ndarray (float)
            array holding change in concentrations of all macroscopic species 
            from the syringe to this condition
        total_volume : float
            total volume of cell plus titrant at this point in the titration
        injection_volume : float
            volume of last injection
        obs_mask : np.ndarray (bool or int)
            mask that grabs microscopic species from micro_array that correspond
            to the numerator when calculating the observable
        denom : int
            index of the macro species that should be used as the denominator
            for the observable calculation
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
        
    def calc_value(self,*args,**kwargs):
        """
        Calculate the observable given the estimated concentrations of all 
        species. *args and **kwargs are ignored.
        """

        num = np.sum(self._micro_array[self._idx,self._obs_mask])
        den = self._macro_array[self._idx,self._denom]

        if den == 0:
            return np.nan
        return num/den
    
    def get_d_y_d_concs(self):
        """
        Calculate the derivative of the calculated value with respect to the
        microscopic species concentrations: d(y_calc)/d(micro_concs).

        For y = sum(micro_num) / macro_den, the derivative with respect to
        a specific micro_species[k] is 1/macro_den if k is in the numerator
        mask, and 0 otherwise.

        Returns
        -------
        numpy.ndarray
            A 1D array of shape (num_micro_species,).
        """
        den = self._macro_array[self._idx, self._denom]
        if den == 0:
            return np.zeros(self._micro_array.shape[1], dtype=float)
            
        # The derivative is 1/den for species in the numerator, 0 for all others.
        deriv = self._obs_mask.astype(float) / den
        return deriv

    def get_d_y_d_other_params(self, parameters):
        """
        Calculate the derivative of the calculated value with respect to any
        "other" parameters (i.e., not binding constants). For SpecPoint,
        this is essentially zero as fudge factors are handled implicitly.

        Returns
        -------
        dict
            An empty dictionary, as there are no direct parameter dependencies.
        """
        # Spectroscopic points have no direct dependence on enthalpies or fudges.
        # The effect of fudge factors is implicitly captured by the chain rule
        # in the main GlobalModel jacobian method, via the d(concs)/d(fudge) term.
        return {}