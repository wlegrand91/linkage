
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
                         del_macro_array=del_macro_array)
        
        self._obs_mask = obs_mask
        self._denom = denom
        
    def calc_value(self,*args,**kwargs):
        """
        Calculate the observable given the estimated concentrations of all 
        species. *args and **kwargs are ignored.
        """

        num = np.sum(self._micro_array[self._idx,self._obs_mask])
        den = self._macro_array[self._idx,self._denom]

        return num/den


