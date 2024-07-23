import numpy as np

class ExperimentalPoint:
    """
    Class for storing an individual experimental data point. Should be sub-
    classed. 
    """
    
    def __init__(self,
                 idx,
                 expt_idx,
                 obs_key):
        """
        Should be sub-classed.
        """
        
        self._idx = idx
        self._expt_idx = expt_idx
        self._obs_key = obs_key

    @property
    def idx(self):
        """
        Index of point in experimental array.
        """
        return self._idx

    @property
    def expt_idx(self):
        """
        Index of experiment.
        """
        return self._expt_idx
    
    @property
    def obs_key(self):
        """
        Name of observable.
        """
        return self._obs_key



class SpecPoint(ExperimentalPoint):
    """
    Class holds experimental data for an individual spectroscopic 
    experimental data point and how it links to the binding model. 
    """

    def __init__(self,
                 idx,
                 expt_idx,
                 obs_key,
                 obs_mask,
                 denom,
                 micro_array,
                 macro_array):
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
        obs_mask : np.ndarray (bool or int)
            mask that grabs microscopic species from micro_array that correspond
            to the numerator when calculating the observable
        denom : int
            index of the macro species that should be used as the denominator
            for the observable calculation
        micro_array : np.ndarray (float)
            array holding concentrations of all microscopic species, calculated
            elsewhere
        macro_array : np.ndarray (float)
            array holding concentrations of all macroscopic species, calculated
            elsewhere
        """

        super().__init__(idx=idx,
                         expt_idx=expt_idx,
                         obs_key=obs_key)
        
        self._obs_mask = obs_mask
        self._denom = denom
        
        self._micro_array = micro_array
        self._macro_array = macro_array
        
    def calc_value(self,*args,**kwargs):
        """
        Calculate the observable given the estimated concentrations of all 
        species. *args and **kwargs are ignored.
        """

        num = np.sum(self._micro_array[self._idx,self._obs_mask])
        den = self._macro_array[self._idx,self._denom]

        return num/den


class ITCPoint(ExperimentalPoint):
    """
    Class holds experimental data for an individual ITC experimental data point
    and how it links to the thermodynamic model. 
    """

    def __init__(self,
                 idx,
                 expt_idx,
                 obs_key,
                 dh_param_start_idx,
                 dh_param_end_idx,
                 micro_array):
        """
        Initialize an ITC data point. 
        
        Parameters
        ----------
        idx : int
            index of point in the experimental array
        expt_idx : int
            index of the experiment itself (allowing multiple experiments)
        obs_key : str
            key pointing to observable from the experiment
        dh_param_start_idx : int
            index of first enthalpy parameter in guesses array
        dh_param_end_idx : int
            index of last enthalpy parameter in guesses array
        micro_array : np.ndarray (float)
            array holding concentrations of all microscopic species, calculated
            elsewhere
        """

        super().__init__(idx=idx,
                         expt_idx=expt_idx,
                         obs_key=obs_key)
        
        self._dh_param_start_idx = dh_param_start_idx
        self._dh_param_end_idx = dh_param_end_idx
        
        self._micro_array = micro_array
        
    def calc_value(self,parameters,*args,**kwargs):
        """
        Calculate the heat for this shot given the current estimated
        concentration changes and enthalpy parameters. *args and **kwargs are
        ignored. 

        Parameters
        ----------
        parameters : np.ndarray (float)
            fit parameters (guesses array)
        """

        dh_array = parameters[self._dh_param_start_idx:self._dh_param_end_idx]
        dC = self._micro_array[self._idx,:] - self._micro_array[self._idx-1,:]
        
        return np.sum(dC*dh_array)