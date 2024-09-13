import numpy as np

class ExperimentalPoint:
    """
    Class for storing an individual experimental data point. Should be sub-
    classed. 
    """
    
    def __init__(self,
                 idx,
                 expt_idx,
                 obs_key,
                 micro_array,
                 macro_array,
                 del_macro_array,
                 total_volume,
                 injection_volume):
        """
        Should be sub-classed.
        """
        
        self._idx = idx
        self._expt_idx = expt_idx
        self._obs_key = obs_key
        self._micro_array = micro_array
        self._macro_array = macro_array
        self._del_macro_array = del_macro_array
        self._total_volume = total_volume
        self._injection_volume = injection_volume
        

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

