from linkage.global_model.point.experimental_point import ExperimentalPoint

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
                 dh_sign,
                 dh_product_mask,
                 dh_dilution_idx,
                 titrating_species_mask):
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
        dh_sign : list-like
            list of enthalpy signs (1 for forward, -1 for reverse) for each 
            reaction
        dh_product_mask : list-like
            list of boolean masks for pulling out products when calcuating 
            enthalpy changes
        dh_dilution_idx : list
            list of integer indices pointing to the dilution heat parameters in
            the main parameter vector.
        titrating_species_mask : np.ndarray (bool)
            mask indicating which macro species are being titrated in this
            experiment.
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
        self._dh_sign = dh_sign
        self._dh_product_mask = dh_product_mask
        self._dh_dilution_idx = dh_dilution_idx
        self._titrating_species_mask = titrating_species_mask

        # Get volume dilution scalar
        self._meas_vol_dilution = (1 - self._injection_volume/self._total_volume)

        
    def calc_value(self, parameters, full_dh_array=None, **kwargs):
        """
        Calculate the heat for this shot given the current estimated
        concentration changes and enthalpy parameters. 

        Parameters
        ----------
        parameters : np.ndarray (float)
            fit parameters (guesses array)
        full_dh_array : np.ndarray, optional
            A pre-constructed array containing the enthalpy value for every
            single equilibrium, respecting any reparameterization rules.
        """
        if self._idx == 0:
            return 0.0

        if full_dh_array is None:
            raise ValueError("full_dh_array must be provided to ITCPoint.calc_value")

        dh_array = full_dh_array
        
        total_heat = 0.0
        
        # Get conc changes for each equilibrium. 
        for i in range(len(self._dh_product_mask)):
            C_before = self._micro_array[self._idx-1, self._dh_product_mask[i]]
            C_after  = self._micro_array[self._idx, self._dh_product_mask[i]]
            del_C = C_after - C_before * self._meas_vol_dilution
            dC = np.mean(del_C)
            total_heat += dh_array[i] * self._dh_sign[i] * dC

        total_heat = total_heat * self._total_volume

        # Heat of dilution
        if len(self._dh_dilution_idx) > 0:
            dil_heats = parameters[self._dh_dilution_idx]
            molar_change = self._del_macro_array[self._idx, self._titrating_species_mask]
            
            if dil_heats.shape == molar_change.shape:
                total_heat += np.sum(dil_heats * molar_change) * self._injection_volume
        
        return total_heat
    
    def get_d_y_d_concs(self):
        """
        Returns a placeholder for d(heat)/d(micro_concs).

        The actual logic for this derivative is complex as it depends on both
        the current and previous concentration states (C_after and C_before).
        This is handled directly in the `GlobalModel.jacobian_normalized`
        method for simplicity and to avoid passing many parameters.
        """
        return np.zeros(self._micro_array.shape[1], dtype=float)

    def get_d_y_d_other_params(self, parameters, full_dh_array=None, **kwargs):
        """
        Calculate the derivative of the heat with respect to any "other"
        parameters, which for ITC are the heats of dilution. Derivatives for
        reaction enthalpies (dH) are handled in GlobalModel.

        Parameters
        ----------
        parameters : np.ndarray
            The full vector of fittable parameters.
        full_dh_array : np.ndarray, optional
            Ignored in this method, but kept for consistent signature with calc_value.

        Returns
        -------
        dict
            A dictionary where keys are parameter *indices* and values are
            their derivatives.
        """
        deriv_dict = {}
        if self._idx == 0:
            return deriv_dict

        # Derivatives with respect to heats of dilution (dil_params)
        if len(self._dh_dilution_idx) > 0:
            molar_change = self._del_macro_array[self._idx, self._titrating_species_mask]
            for i, param_index in enumerate(self._dh_dilution_idx):
                if i < len(molar_change):
                    deriv_dict[param_index] = self._injection_volume * molar_change[i]

        return deriv_dict

    def get_error_value(self, y_calc, base_error=0.1, proportional_error=0.01):
        """
        Calculate the expected standard deviation for this point using the robust
        ITC error model: sigma = sqrt(sigma_base^2 + (f_rel * y_calc)^2)

        Parameters
        ----------
        y_calc : float
            The calculated heat value for this point.
        base_error : float, default=0.1
            The constant noise floor (e.g. uCal).
        proportional_error : float, default=0.01
            The proportional error factor (fractional, e.g. 0.01 for 1%).

        Returns
        -------
        float
            Calculated standard deviation sigma_i.
        """
        # Ensure positive sigma
        sigma = np.sqrt(base_error**2 + (proportional_error * y_calc)**2)
        return sigma