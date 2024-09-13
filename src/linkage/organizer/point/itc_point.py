
from .experimental_point import ExperimentalPoint

import numpy as np

class ITCPoint(ExperimentalPoint):
    """
    Class holds experimental data for an individual ITC experimental data point
    and how it links to the thermodynamic model. 
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
                 dh_param_start_idx,
                 dh_param_end_idx,
                 dh_sign,
                 dh_product_mask,
                 dh_dilution_mask):
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
        dh_param_start_idx : int
            index of first enthalpy parameter in guesses array
        dh_param_end_idx : int
            index of last enthalpy parameter in guesses array
        dh_sign : list-like
            list of enthalpy signs (1 for forward, -1 for reverse) for each 
            reaction
        dh_product_mask : list-like
            list of boolean masks for pulling out products when calcuating 
            enthalpy changes
        dh_dilution_mask : np.ndarray (bool)
            mask indicating which macro species have a dilution heat associated
            with them. 
        """
        
        super().__init__(idx=idx,
                         expt_idx=expt_idx,
                         obs_key=obs_key,
                         micro_array=micro_array,
                         macro_array=macro_array,
                         del_macro_array=del_macro_array,
                         total_volume=total_volume,
                         injection_volume=injection_volume)

        # Get dh specific parameters        
        self._dh_param_start_idx = dh_param_start_idx
        self._dh_param_end_idx = dh_param_end_idx
        self._dh_sign = dh_sign
        self._dh_product_mask = dh_product_mask
        self._dh_dilution_mask = dh_dilution_mask

        # Decide how to cut parameter array into enthalpies (first block of
        # param) and heats of dilution (second block of param)
        self._dh_first = self._dh_param_start_idx
        self._dh_last = self._dh_first + len(self._dh_sign) 
        self._dil_first = self._dh_last
        self._dil_last = self._dh_param_end_idx

        # Get volume dilution scalar
        self._meas_vol_dilution = (1 - self._injection_volume/self._total_volume)

        
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

        dh_array = parameters[self._dh_first:self._dh_last]
        
        total_heat = 0.0
        
        # Get conc changes for each equilibrium. 
        for i in range(len(self._dh_product_mask)):

            # Concentration of relevant microspecies before the injection 
            C_before = self._micro_array[self._idx-1,self._dh_product_mask[i]]

            # Concentration of relevant microspecies after the injection
            C_after  = self._micro_array[self._idx,self._dh_product_mask[i]]
            
            # Concentration change in the cell itself. Scale the down the 
            # concentration before to account for the dilution effect of the 
            # shot.
            del_C = C_after - C_before*self._meas_vol_dilution

            # Treat concentration change as the *mean* of all species in 
            # dh_product_mask. For a simple reaction A + B -> C, this would be
            # the mean of the change in "C". For a more complicated reaction, 
            # this would be A + B -> C + D, this would be mean(dC,dD). 
            dC = np.mean(del_C)

            total_heat += dh_array[i]*self._dh_sign[i]*dC

        total_heat = total_heat*self._total_volume

        # Heat of dilution
        dil_heats = parameters[self._dil_first:self._dil_last]
        molar_change = self._del_macro_array[self._idx,self._dh_dilution_mask]
        total_heat += np.sum(dil_heats*molar_change)*self._injection_volume
        
        return total_heat