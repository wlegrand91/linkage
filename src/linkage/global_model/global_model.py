import linkage.models
from linkage.global_model.point.spec_point import SpecPoint
from linkage.global_model.point.itc_point import ITCPoint

import numpy as np
import pandas as pd
import copy
import warnings
import traceback

class GlobalModel:
    def __init__(self, expt_list, model_name, model_spec=None):
        """
        This class integrates experimental data with thermodynamic modeling by combining 
        a list of experiments with a specified model. The integrated model includes:

        - Equilibrium constants derived from the thermodynamic model
        - Optional nuisance concentration parameters for each experiment
        - Enthalpies for model equilibria and heats of dilution for titrating species 
        (when ITC experiments are included)

        Key features:
        - Signal normalization: Transforms experimental signals (e.g., ITC heats, 
        spectroscopic channels) using (value - mean)/std across all experiments of 
        the same type, ensuring consistent scaling for residual calculations
        - Balanced weighting: Weights observations inversely to the number of data 
        points in each experiment, ensuring equal contribution regardless of 
        experiment size

        Parameters
        ----------
        expt_list : list
            List of experiments with loaded observations
        model_name : str
            Name of the thermodynamic model to use for concentration calculations
        model_spec : str, optional
            Model specification string for generic models. If None, a manually defined 
            model class is assumed.
        """

        # Store model name and experiment list
        self._model_name = model_name
        self._expt_list = copy.deepcopy(expt_list)
        self._model_spec = model_spec

        # Scaling Factor for Enthalpy Parameters (Cal -> kCal for fitting)
        self._dh_scale = 1000.0
        
        # Error Model Parameters
        self._error_model_params = {}
        for i, expt in enumerate(self._expt_list):
            # Default Robust Error Model (sigma_base=0.1 uCal, f_rel=0%)
            # Applies to 'heat' observable if present.
            self._error_model_params[i] = {
                "heat": {"sigma_base": 0.1, "f_rel": 0.0}
            }

        # Load the model
        self._load_model()

        # Load experimental data
        self._get_expt_std_scalar()
        self._get_expt_normalization()
        self._load_observables()

        self._get_enthalpy_param()
        self._get_expt_fudge()
    
    
        # Create points that allow calculation of observations
        self._build_point_map()

        # Add confirmation printout for analytical Jacobian
        if self._model_name == "GenericBindingModel":
            # Check the underlying binding model to see if it successfully created the jacobian function.
            if hasattr(self._bm, "jacobian_function") and self._bm.jacobian_function is not None:
                print("INFO: Analytical Jacobian was successfully generated for the binding model.")
            else:
                # The warning from GenericBindingModel's __init__ will provide the specific error.
                # This just serves as a high-level confirmation of failure.
                print("WARNING: Analytical Jacobian could not be generated. Fitter will use numerical methods.")
                self.jacobian_normalized = None


    def _load_model(self):
        """
        Load and initialize the thermodynamic linkage model. 
        """

        # Make a list of all classes in linkage.models
        available_models = {}
        for k in linkage.models.__dict__:
            if k.startswith("_"):
                continue

            if issubclass(type(linkage.models.__dict__[k]),type):
                available_models[k] = linkage.models.__dict__[k]

        # Make sure the model the user specified is found 
        if self._model_name not in available_models:
            err = f"model_name '{self._model_name}' not recognized. It should be one of:\n"
            for k in available_models:
                err += f"    {k}\n"
            err += "\n"
            raise ValueError(err)

        # Initialize binding model
        ModelClass = available_models[self._model_name]
        if self._model_name == "GenericBindingModel":
            if self._model_spec is None:
                raise ValueError("model_spec must be provided for GenericBindingModel")
            self._bm = ModelClass(model_spec=self._model_spec)
        else:
            if self._model_spec is not None:
                print("Warning: model_spec provided but not used for non-generic model")
            self._bm = ModelClass()

        # Record names of the model parameters
        self._parameter_names = []
        self._parameter_guesses = []
        for p in self._bm.param_names:
            self._parameter_names.append(p)
            self._parameter_guesses.append(0.0)

        # Record indexes spanning parameter guesses
        self._bm_param_start_idx = 0
        self._bm_param_end_idx = len(self._parameter_names) - 1

    def _get_expt_std_scalar(self):
        """
        Second, we normalize each experiment to the number of points in that 
        experiment. The normalization is: 

            theta = num_obs/sum(num_obs)
            y_std = y_std*(1 - theta + np.max(theta))
        """

        # Number of points contributed by each experiment
        points_per_expt = []
        for expt in self._expt_list:

            # Count the total number of points contributed by this experiment
            # num_observables times number of not-ignored points
            num_obs = len(expt.observables)
            num_not_ignore = np.sum(np.logical_not(expt.expt_data["ignore_point"]))
            points_per_expt.append(num_obs*num_not_ignore)

        # Scale y_std for each experiment by this value. It will be 1 for the 
        # experiment with the smallest number of points and will increase  
        # for experiments with more points. 
        points_per_expt = np.array(points_per_expt)
        if np.sum(points_per_expt) > 0:
            theta = points_per_expt/np.sum(points_per_expt)
            self._expt_std_scalar = 1 - theta + np.max(theta)
        else:
            self._expt_std_scalar = np.ones(len(points_per_expt))


    def _get_expt_normalization(self):
        """
        First, each unique 'obs' seen (e.g. heat, cd222, etc.) is normalized to
        all values of that obs type seen across all experiments. So, if there
        are three itc experiments, we will do a single normalization across all
        three experiments. The normalization is: 
            (value - mean(value))/std(value)
        where the mean and std are taken over all experimental values with that
        obs. 
        """

        # Create dictionary keying obs to a list of all observed values for that 
        # obs across experiments. 
        obs_values_seen = {}
        for expt in self._expt_list:

            for obs in expt.observables:

                keep = np.logical_not(expt.expt_data["ignore_point"])
                obs_values = list(expt.expt_data.loc[keep,obs])
                if obs not in obs_values_seen:
                    obs_values_seen[obs] = []

                obs_values_seen[obs].extend(obs_values)

        # Create a normalization_params dictionary that keys obs to the mean and
        # std of that obs. 
        self._normalization_params = {}
        for obs in obs_values_seen:

            values = np.array(obs_values_seen[obs])
            values = values[np.logical_not(np.isnan(values))]
            if len(values) == 0:
                mean_value = 0
                std_value = 1
            else:
                mean_value = np.mean(values)
                std_value = np.std(values)

            self._normalization_params[obs] = [mean_value,std_value]

    def _load_observables(self):

        # Flat list of observed values. Values to use for model()
        self._y_obs = []
        self._y_std = []

        # Normalized observed values. Values to use for model_normalized()
        # (value - mean)/std for y_obs
        # (value/std)*y_std_scalar for y_std
        self._y_obs_normalized = []
        self._y_std_normalized = []

        # These flat arrays allow us to reverse the normalizations:
        # (value*std + mean) for y_obs
        # (value*y_norm_std/y_std_scalar) or y_std
        self._y_norm_mean = []
        self._y_norm_std = []
        self._y_std_scalar = []

        # For each experiment
        for expt_counter, expt in enumerate(self._expt_list):

            # Add any macro_species from the model but not seen in the experiment
            not_in_expt = set(self._bm.macro_species) - set(expt.expt_concs.columns)
            for missing in not_in_expt:
                expt.add_expt_conc_column(new_column=missing)

            # For each observable
            for obs in expt.observables:

                # For each point in the observable
                for point_idx in range(len(expt.expt_data)):

                    # Grab experimental point 
                    obs_info = expt.observables[obs]
                    expt_data = expt.expt_data.loc[expt.expt_data.index[point_idx],:]

                    # Skip the point if ignored                
                    if expt_data["ignore_point"]:
                        continue

                    # Record observations and standard deviations
                    self._y_obs.append(expt_data[obs])
                    
                    # Robust Error Model Logic
                    # Check if error model is defined for this experiment/obs
                    # We use a tuple key (expt_idx, obs) or just check validity
                    # Assuming we store it as {expt_id: {obs: {sigma_base, f_rel}}} ?? 
                    # Simpler: self._error_model_params.get(expt.expt_index?? no expt has no id internally?)
                    # GlobalModel uses list index as ID.
                    
                    std_val = expt_data[obs_info["std_column"]]
                    
                    # If error model params exist for this experiment index
                    if expt_counter in self._error_model_params:
                        ep = self._error_model_params[expt_counter]
                        # Check if specific observable is covered (e.g. 'heat')
                        if obs in ep:
                            sigma_base = ep[obs].get("sigma_base", 0.0)
                            f_rel = ep[obs].get("f_rel", 0.0)
                            
                            # Calculate Robust SD: sqrt(base^2 + (f * heat)^2)
                            # Use Absolute Value of heat!
                            y_val = expt_data[obs]
                            std_val = np.sqrt(sigma_base**2 + (f_rel * y_val)**2)

                    self._y_std.append(std_val)

                    # Get mean and std of obs for normalization
                    obs_mean = self._normalization_params[obs][0]
                    obs_std = self._normalization_params[obs][1]
                    y_std_scalar = self._expt_std_scalar[expt_counter]

                    # Record information that will be used to normalize the point
                    self._y_norm_mean.append(obs_mean)
                    self._y_norm_std.append(obs_std)
                    self._y_std_scalar.append(y_std_scalar)

                    # Do normalization
                    y_obs_norm = (self._y_obs[-1] - obs_mean)/obs_std
                    y_std_norm = self._y_std[-1]/obs_std*y_std_scalar

                    # Record normalized values
                    self._y_obs_normalized.append(y_obs_norm)
                    self._y_std_normalized.append(y_std_norm)

        # Convert lists populated above into numpy arrays
        self._y_obs = np.array(self._y_obs)
        self._y_std = np.array(self._y_std)

        self._y_norm_mean = np.array(self._y_norm_mean)
        self._y_norm_std = np.array(self._y_norm_std)
        self._y_std_scalar = np.array(self._y_std_scalar)

        self._y_obs_normalized = np.array(self._y_obs_normalized)
        self._y_std_normalized = np.array(self._y_std_normalized)

    @property
    def physical_parameter_names(self):
        """
        Return the list of physical parameter names (Ks, dHs) that are derived from
        the regression parameters.
        """
        if hasattr(self._bm, "physical_param_names"):
             return self._bm.physical_param_names
        return []

    def _get_enthalpy_param(self):
        """
        Deal with enthalpy terms if needed. This method is now aware of
        reparameterization rules for dH values.
        """

        # Look for an ITC experiment
        need_enthalpies = False
        for expt in self._expt_list:
            for obs in expt.observables:    
                if expt.observables[obs]["type"] == "itc":
                    need_enthalpies = True
                    break

        if not need_enthalpies:
            return

        self._dh_param_start_idx = len(self._parameter_names)

        # Reaction enthalpies
        self._dh_sign = []
        self._dh_product_mask = []
        self._dh_name_map = {}
        
        dh_reparam_rules = {}
        if hasattr(self._bm, "reparam_rules"):
            dh_reparam_rules = {s.name: getattr(e, "name", str(e)) for s, e in self._bm.reparam_rules.items() if s.name.startswith("dH_")}

        # Create a list of all possible dH parameters, one for each equilibrium
        original_dh_names = [f"dH_{k[1:]}" for k in self._bm.equilibria]
        
        # Create the fittable list by removing dependent parameters
        dependent_dh_names = set(dh_reparam_rules.keys())
        potential_dh_params = sorted([name for name in original_dh_names if name not in dependent_dh_names])
        
        # Add only the fittable dH parameters to the master list IF NOT PRESENT
        for name in potential_dh_params:
            if name not in self._parameter_names:
                self._parameter_names.append(name)
                self._parameter_guesses.append(0.0)

        # Now, build the map from ALL original equilibria to their independent parent
        for k in self._bm.equilibria:
            dh_name = f"dH_{k[1:]}"
            self._dh_name_map[k] = dh_reparam_rules.get(dh_name, dh_name)
        
        for k in self._bm.equilibria:
            reactants, products = self._bm.equilibria[k]
            if len(products) <= len(reactants):
                self._dh_sign.append(1.0)
                key_species = products[:]
            else:
                self._dh_sign.append(-1.0)
                key_species = reactants[:]
            self._dh_product_mask.append(np.isin(self._bm.micro_species, key_species))

        # Heats of dilution
        to_dilute = sorted(list(set(s for e in self._expt_list for o in e.observables 
                                    if e.observables[o]["type"] == "itc" 
                                    for s in e.titrating_macro_species)))
        
        self._dh_dilution_idx_map = {}
        for s in to_dilute:
            param_name = f"nuisance_dil_{s}"
            if param_name not in self._parameter_names:
                self._parameter_names.append(param_name)
                self._parameter_guesses.append(0.0)
            self._dh_dilution_idx_map[s] = self._parameter_names.index(param_name)
        
        self._dh_param_end_idx = len(self._parameter_names) - 1

    def _get_expt_fudge(self):
        """
        Fudge parameters account for uncertainty in one of the total
        concentrations each experiment. This is specified by `conc_to_float`
        when the `Experiment` class is initialized. 
        """
        self._fudge_list = []
        for i, expt in enumerate(self._expt_list):
            if expt.conc_to_float:
                param_name = f"nuisance_expt_{i}_{expt.conc_to_float}_fudge"
                if param_name not in self._parameter_names:
                    self._parameter_names.append(param_name)
                    self._parameter_guesses.append(1.0)
                
                fudge_species_idx = np.where(self._bm.macro_species == expt.conc_to_float)[0][0]
                self._fudge_list.append((fudge_species_idx, self._parameter_names.index(param_name)))
            else:
                self._fudge_list.append(None)

    def _add_point(self,point_idx,expt_idx,obs):
        expt = self._expt_list[expt_idx]
        obs_info = expt.observables[obs]
        data_idx = expt.expt_data.index[point_idx]
        if expt.expt_data.loc[data_idx, "ignore_point"]: return

        point_kwargs = {"idx": point_idx, "expt_idx": expt_idx, "obs_key": obs,
                        "micro_array": self._micro_arrays[-1], "macro_array": self._macro_arrays[-1],
                        "del_macro_array": self._del_macro_arrays[-1],
                        "total_volume": float(expt.expt_concs.loc[data_idx, "volume"]),
                        "injection_volume": float(expt.expt_data.loc[data_idx, "injection"])}

        if obs_info["type"] == "spec":
            point_kwargs["obs_mask"] = np.isin(self._bm.micro_species, obs_info["microspecies"])
            point_kwargs["denom"] = np.where(self._bm.macro_species == obs_info["macrospecies"])[0][0]
            pt = SpecPoint(**point_kwargs)
        elif obs_info["type"] == "itc":
            # Identify titration indices specifically for this experiment, in macro_species order
            local_dilution_idx = []
            for s in self._bm.macro_species:
                if s in expt.titrating_macro_species:
                    if s in self._dh_dilution_idx_map:
                        local_dilution_idx.append(self._dh_dilution_idx_map[s])
                    else:
                        # Should not happen if to_dilute logic is correct
                        pass

            point_kwargs.update({
                "dh_sign": self._dh_sign,
                "dh_product_mask": self._dh_product_mask,
                "dh_dilution_idx": local_dilution_idx,
                "titrating_species_mask": np.array([s in expt.titrating_macro_species 
                                                    for s in self._bm.macro_species])
            })
            pt = ITCPoint(**point_kwargs)
        else:
            raise ValueError(f"Obs type '{obs_info['type']}' not recognized.")
        
        self._points.append(pt)

    def _build_point_map(self):
        self._ref_macro_arrays, self._macro_arrays, self._micro_arrays = [], [], []
        self._del_macro_arrays, self._expt_syringe_concs, self._points = [], [], []

        for i, expt in enumerate(self._expt_list):
            self._micro_arrays.append(np.full((len(expt.expt_data), len(self._bm.micro_species)), np.nan))
            
            macro_array = np.zeros((len(expt.expt_data), len(self._bm.macro_species)))
            for j, species in enumerate(self._bm.macro_species):
                macro_array[:,j] = expt.expt_concs[species].values
            self._ref_macro_arrays.append(macro_array)
            self._macro_arrays.append(macro_array.copy())
            print(f"DEBUG: Expt {i} Macro Array Mean: {np.mean(macro_array, axis=0)}")
            
            syringe_concs = np.array([expt.syringe_contents.get(s, 0.0) for s in self._bm.macro_species])
            self._expt_syringe_concs.append(syringe_concs)
            self._del_macro_arrays.append(syringe_concs - macro_array)
            
            for obs in expt.observables:
                for j in range(len(expt.expt_data)):
                    self._add_point(point_idx=j, expt_idx=i, obs=obs)

    def model(self,parameters):
        """
        Model output. Can be used to draw plots or as the target of a regression
        analysis against y_obs. 
        """
        start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1
    
        # Create internally scaled parameters vector (fit kCal -> physics Cal)
        parameters_internal = np.array(parameters, dtype=float)
        for i, name in enumerate(self._parameter_names):
            if name.startswith('dH') or "nuisance_dil" in name:
                parameters_internal[i] *= self._dh_scale

        # Prepare params dict (using heuristic exp/linear) from SCALED parameters
        bm_params_prepared = {}
        for i in range(start, end):
             p_name = self._parameter_names[i]
             val = parameters_internal[i]
             # Use val directly as dH/nuisance_dil are already scaled in parameters_internal
             # Other nuisance parameters (fudge factors) are not scaled, so use val directly.
             if p_name.startswith('dH') or "nuisance" in p_name:
                 bm_params_prepared[p_name] = val
             else:
                 bm_params_prepared[p_name] = np.exp(val)
        
        # Get Physical Parameters (including dHs)
        phys_params = self._bm.get_physical_params(bm_params_prepared)
        
        # Extract dH values for each equilibrium in order
        full_dh_array = np.zeros(len(self._bm.equilibria))
        for i, k_name in enumerate(self._bm.equilibria):
            dh_name = f"dH_{k_name[1:]}"
            val = phys_params.get(dh_name, None)
            
            # Fallback for legacy models where dH is managed by GlobalModel
            if val is None:
                if dh_name in self._parameter_names:
                    idx = self._parameter_names.index(dh_name)
                    val = parameters_internal[idx]
                else:
                    val = 0.0
            
            full_dh_array[i] = val

        for i in range(len(self._macro_arrays)):
            fudge_value = 1.0
            if self._fudge_list[i] is not None:
                fudge_species_idx, fudge_param_idx = self._fudge_list[i]
                fudge_value = parameters_internal[fudge_param_idx]

            self._macro_arrays[i] = self._ref_macro_arrays[i].copy()
            if self._fudge_list[i] is not None:
                self._macro_arrays[i][:,fudge_species_idx] *= fudge_value
            
            self._del_macro_arrays[i] = self._expt_syringe_concs[i] - self._macro_arrays[i]
            
            for j in range(len(self._macro_arrays[i])):
                self._micro_arrays[i][j,:] = self._bm.get_concs(param_array=parameters_internal[start:end],
                                                                macro_array=self._macro_arrays[i][j,:])

        y_calc = np.full(len(self._points), np.nan)
        for i, pt in enumerate(self._points):
            if isinstance(pt, ITCPoint):
                y_calc[i] = pt.calc_value(parameters_internal, full_dh_array=full_dh_array)
            else:
                y_calc[i] = pt.calc_value(parameters_internal)

        return y_calc

    def model_normalized(self, parameters):
        """
        Model output where each experiment is normalized using the mean and std
        of all observables of the same type seen across all experiments. 
        """
        y_calc = self.model(parameters)
        if np.all(np.isclose(self._y_norm_std,0)):
            return y_calc - self._y_norm_mean
        return (y_calc - self._y_norm_mean) / self._y_norm_std

    
    def jacobian_normalized(self, parameters):
        """
        Calculate the Jacobian of the normalized model output with respect to
        all fittable parameters. This is d(y_calc_normalized)/d(parameters).
        """
        try:
            self.model(parameters)
            num_obs, num_params = len(self._points), len(self.parameter_names)
            J = np.zeros((num_obs, num_params))
            start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1
            
            # Prepare params dict (Linearized)
            bm_params_prepared = {}
            for i in range(start, end):
                 p_name = self._parameter_names[i]
                 val = parameters[i]
                 if p_name.startswith('dH') or "nuisance_dil" in p_name:
                     bm_params_prepared[p_name] = val * self._dh_scale
                 elif "nuisance" in p_name:
                     bm_params_prepared[p_name] = val
                 else:
                     bm_params_prepared[p_name] = np.exp(val)
            
            # 1. Physical Parameters and Jacobian d(Phys)/d(Reg)
            phys_params = self._bm.get_physical_params(bm_params_prepared)
            d_phys_d_reg = self._bm.get_physical_jacobian(bm_params_prepared).astype(float) # (N_Phys, N_Reg)
            phys_param_names = self._bm.physical_param_names
            
            # Apply Chain Rule for Log Parameters to d_phys_d_reg
            # If P_log is fitted, but P_lin used in map, d/dP_log = d/dP_lin * P_lin
            for i, p_name in enumerate(self._parameter_names[start:end]):
                 if not (p_name.startswith('dH') or "nuisance" in p_name):
                      val_lin = bm_params_prepared[p_name]
                      d_phys_d_reg[:, i] *= val_lin

            # Extract independent dH values and dH derivatives
            full_dh_array = np.zeros(len(self._bm.equilibria))
            d_dh_d_reg_list = [] # List of rows from Jacobian corresponding to dH_K
            
            for i, k_name in enumerate(self._bm.equilibria):
                dh_name = f"dH_{k_name[1:]}"
                full_dh_array[i] = phys_params.get(dh_name, 0.0)
                
                # Get derivative row
                if dh_name in phys_param_names:
                    idx = phys_param_names.index(dh_name)
                    d_dh_d_reg_list.append(d_phys_d_reg[idx, :])
                else:
                    d_dh_d_reg_list.append(np.zeros(d_phys_d_reg.shape[1]))

            # 2. Concentration Jacobians (dC/dReg)
            d_concs_d_bm_params_list = []
            
            # We must pass the FULL dict (Reg Params + Concs) to get_numerical_jacobian
            # because logic in GenericBindingModel relies on it to call SymbolicBindingModel.
            bm_param_dict = bm_params_prepared # This is Reg parameters (Linearized K, dH) 
            
            for i in range(len(self._expt_list)):
                exp_jacobians = []
                for j in range(len(self._macro_arrays[i])):
                    all_concs_dict = {
                        **bm_param_dict, 
                        **dict(zip(self._bm.macro_species, self._macro_arrays[i][j,:])),
                        **dict(zip(self._bm.micro_species, self._micro_arrays[i][j,:]))
                    }
                    
                    jac = self._bm.get_numerical_jacobian(all_concs_dict)
                    if jac is None or np.any(np.isnan(jac)):
                        jac = np.full((len(self._bm.micro_species), len(self._bm.param_names)), np.nan)
                    


                    exp_jacobians.append(jac)
                d_concs_d_bm_params_list.append(exp_jacobians)

            # 3. Assemble Full Jacobian
            for i, pt in enumerate(self._points):
                expt_idx, shot_idx = pt.expt_idx, pt.idx
                d_concs_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx]

                if isinstance(pt, SpecPoint):
                    J[i, start:end] = pt.get_d_y_d_concs() @ d_concs_d_bm
                
                elif isinstance(pt, ITCPoint) and pt.idx > 0:
                    d_concs_before_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx - 1]
                    
                    # Term 1: Heat change due to Concentration change (dH * dC/dP)
                    term1 = np.zeros(len(self._bm.param_names))
                    for j in range(len(pt._dh_product_mask)):
                        mask = pt._dh_product_mask[j]
                        d_C_after = d_concs_d_bm[mask, :]
                        d_C_before = d_concs_before_d_bm[mask, :]
                        d_del_C = d_C_after - d_C_before * pt._meas_vol_dilution
                        d_dC_d_bm = np.mean(d_del_C, axis=0) # d(DeltaC)/d(P)
                        term1 += full_dh_array[j] * pt._dh_sign[j] * d_dC_d_bm
                    
                    # Term 2: Heat change due to dH parameter change (d(dH)/dP * DeltaC)
                    term2 = np.zeros(len(self._bm.param_names))
                    for j in range(len(pt._dh_product_mask)):
                        mask = pt._dh_product_mask[j]
                        C_before = pt._micro_array[shot_idx - 1, mask]
                        C_after  = pt._micro_array[shot_idx, mask]
                        del_C = C_after - C_before * pt._meas_vol_dilution
                        dC = np.mean(del_C) # Scalar Delta Conc
                        
                        # Add derivative contribution: d(dH_j)/dP * dC
                        # d_dh_d_reg_list[j] is the gradient vector for dH_j
                        term2 += d_dh_d_reg_list[j] * pt._dh_sign[j] * dC
                        
                    J[i, start:end] = (term1 + term2) * pt._total_volume
                
                other_param_derivs = pt.get_d_y_d_other_params(parameters, full_dh_array=full_dh_array)
                for param_idx, deriv_val in other_param_derivs.items():
                    J[i, param_idx] = deriv_val

            # 4. Finite Difference for Fudge Parameters (Mixed-Mode AD)
            # Calculate Y at center (current state/micro_arrays are valid from initial model call)
            # Note: self.model(parameters) was called at start. We call it again to get y_center reliably
            # and ensure state is consistent for perturbation.
            y_center = self.model(parameters)
            
            fudge_param_indices = set()
            for item in self._fudge_list:
                if item is not None:
                    fudge_param_indices.add(item[1])
            
            if fudge_param_indices:
                eps = 1e-8
                for p_idx in fudge_param_indices:
                    p_orig = parameters[p_idx]
                    step = eps * max(abs(p_orig), 1.0)
                    
                    p_new = parameters.copy()
                    p_new[p_idx] += step
                    
                    y_perturb = self.model(p_new)
                    dy = (y_perturb - y_center) / step
                    J[:, p_idx] = dy
                
                # Restore State (Micro Arrays)
                self.model(parameters)

            
            if np.any(~np.isclose(self._y_norm_std,0)):
                J /= self._y_norm_std[:, np.newaxis]

            # Apply Scaling to Jacobian Columns dY/dP_reg
            # For dH/Dil: P_reg = P_cal / 1000. => P_cal = 1000 * P_reg
            # dY/dP_reg = dY/dP_cal * dP_cal/dP_reg = dY/dP_cal * 1000
            for i, p_name in enumerate(self._parameter_names):
                if p_name.startswith('dH') or "nuisance_dil" in p_name:
                    J[:, i] *= self._dh_scale

            return J

        except Exception as e:
            tb_str = traceback.format_exc()
            warnings.warn(f"Jacobian calculation failed with error: {e}\n{tb_str}")
            return np.full((len(self._points), len(self.parameter_names)), np.nan)

    @property
    def y_obs(self):
        return self._y_obs

    @property
    def y_std(self):
        return self._y_std

    @property
    def y_obs_normalized(self):
        return self._y_obs_normalized
    
    @property
    def y_std_normalized(self):
        return self._y_std_normalized

    @property
    def parameter_names(self):
        return self._parameter_names
            
    @property
    def parameter_guesses(self):
        return self._parameter_guesses
    
    @property
    def model_name(self):
        return self._model_name
    
    @property
    def macro_species(self):
        return self._bm.macro_species
    
    @property
    def micro_species(self):
        return self._bm.micro_species
    
    @property
    def final_ct(self):
        return getattr(self._bm, "final_ct", None)
    
    @property
    def model_spec(self):
        return getattr(self._bm, "model_spec", None)
    
    @property
    def simplified_equations(self):
        return getattr(self._bm, "simplified_eqs", None)
    
    @property
    def solved_vars(self):
        return getattr(self._bm, "solved_vars", None)
    
    @property
    def as_df(self):
        out = {"expt_id":[],"expt_type":[],"expt_obs":[],"volume":[],"injection":[]}
        for k in self._bm.macro_species: out[k] = []
        for k in self._bm.micro_species: out[k] = []

        for p in self._points:
            out["expt_id"].append(p.expt_idx)
            if isinstance(p, SpecPoint):
                out["expt_type"].append(p.obs_key)
                num = "+".join([s for s_idx, s in enumerate(self._bm.micro_species) if p._obs_mask[s_idx]])
                den = self._bm.macro_species[p._denom]
                out["expt_obs"].append(f"{num}/{den}")
            elif isinstance(p, ITCPoint):
                out["expt_type"].append("itc")
                out["expt_obs"].append("obs_heat")
            else:
                raise ValueError("point class not recognized\n")

            out["volume"].append(p._total_volume)
            out["injection"].append(p._injection_volume)
            for i, k in enumerate(self._bm.macro_species):
                out[k].append(p._macro_array[p._idx,i])
            for i, k in enumerate(self._bm.micro_species):
                out[k].append(p._micro_array[p._idx,i])
            
        out["y_obs"] = self.y_obs
        out["y_std"] = self.y_std
        out["y_obs_norm"] = self.y_obs_normalized
        out["y_std_norm"] = self.y_std_normalized

        return pd.DataFrame(out)

    @property
    def concentrations_df(self):
        return self._bm.concentrations_df



    def calculate_derived_params(self, estimate=None, cov=None, samples=None, dof=None):
        """
        Calculate Derived (Physical) Parameters and their statistics.
        Can handle both Frequentist (estimate + cov) and Bayesian (samples) inputs.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['name', 'estimate', 'std', 'low_95', 'high_95', 'fixed', 'guess', ...]
                          appended to the original parameter list format.
        """
        
        # We want to report ALL physical parameters identified by the mapper
        # plus any other derived params? Just Physical for now.
        if not hasattr(self._bm, "physical_param_names"):
             return None
             
        phys_names = self._bm.physical_param_names
        if not phys_names:
            return None

        # Prepare base dataframe
        results = []

        if samples is not None:
            # Bayesian Mode: Process Samples
            # Map each sample `reg_params` -> `phys_params`
            # This might be slow for many samples if `get_physical_params` is slow.
            # But `ParameterMapper` uses compiled lambdas, should be fast.
            
            # samples shape: (N_samples, N_params)
            # We need to extract the relevant BM parameters from the samples
            start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1
            
            # Pre-allocate output
            phys_samples = np.zeros((samples.shape[0], len(phys_names)))
            
            for k in range(samples.shape[0]):
                row = samples[k, :]
                
                # Construct params dict (Linearized, because get_physical_params expects it?)
                # Wait, generic_binding_model.get_physical_params expects whatever the model expects.
                # In GlobalModel.model(), we UNLOG expected Log params.
                # So we must do the same here.
                
                bm_params_prepared = {}
                for i in range(start, end):
                     p_name = self._parameter_names[i]
                     val = row[i]
                     if p_name.startswith('dH') or "nuisance" in p_name:
                         bm_params_prepared[p_name] = val
                     else:
                         bm_params_prepared[p_name] = np.exp(val)
                
                phys_vals = self._bm.get_physical_params(bm_params_prepared)
                # Ensure ordered
                for p_idx, p_name in enumerate(phys_names):
                    phys_samples[k, p_idx] = phys_vals.get(p_name, np.nan)
                    
            # Calculate stats
            est_vec = np.mean(phys_samples, axis=0) # Or Median? standard is mean for posterior?
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
                    # Add dummy values for other columns usually in fit_df
                    "lower_bound": -np.inf, "upper_bound": np.inf, 
                    "prior_mean": np.nan, "prior_std": np.nan
                })

        elif estimate is not None and cov is not None:
            # Frequentist Mode: Propagation of Uncertainty
            
            # 1. Base Estimates
            bm_params_prepared = {}
            # Need to use the estimate vector
            for i in range(len(estimate)):
                 p_name = self._parameter_names[i]
                 val = estimate[i]
                 
                 # Decouple parameter type based on index or name
                 # BM Parameters (Equilibrium Constants) are typically Log-fitted in GlobalModel
                 # Enthalpies and Nuisance are Linear-fitted
                 
                 # Check if it is a Log-fitted parameter
                 # Typically params in [start, end] correspond to _bm.constants (Ks)
                 if i >= self._bm_param_start_idx and i <= self._bm_param_end_idx:
                     # Check if it is NOT dH (just in case dH got mixed in BM params, though unlikely in GlobalModel structure)
                     if p_name.startswith('dH') or "nuisance" in p_name:
                         bm_params_prepared[p_name] = val
                     else:
                         try:
                            bm_params_prepared[p_name] = np.exp(val)
                         except FloatingPointError:
                            bm_params_prepared[p_name] = np.inf
                 else:
                     # dH, nuisance, etc. (Linear)
                     bm_params_prepared[p_name] = val
            
            phys_vals_dict = self._bm.get_physical_params(bm_params_prepared)
            
            # 2. Jacobian J = dPhys/dReg (padded)
            # shape: (N_phys, N_total_fitted)
            J_phys = np.zeros((len(phys_names), len(estimate)))
            
            start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1
            
            # Get block from GenericBindingModel
            # d_phys_d_bm (N_phys, N_bm)
            prior_jac = self._bm.get_physical_jacobian(bm_params_prepared)
            d_phys_d_bm = np.array(prior_jac, dtype=float, copy=True)
            d_phys_d_bm.flags.writeable = True
            
            # Apply Chain Rule for Log Params
            # Same logic as jacobian_normalized
            # If P_log is fitted, dPhys/dLog = dPhys/dLin * Lin
            # Iterate over BM params (columns of d_phys_d_bm)
            for col_idx, glob_idx in enumerate(range(start, end)):
                 p_name = self._parameter_names[glob_idx]
                 if not (p_name.startswith('dH') or "nuisance" in p_name):
                      val_lin = bm_params_prepared[p_name]
                      d_phys_d_bm[:, col_idx] *= val_lin
            
            # Place in full Jacobian
            J_phys[:, start:end] = d_phys_d_bm
            
            # 3. Covariance Prop: Cov_phys = J @ Cov_reg @ J.T
            Cov_phys = J_phys @ cov @ J_phys.T
            var_phys = np.diagonal(Cov_phys).copy()
            # Handle negative variance (numerical noise)
            var_phys[var_phys < 0] = 0
            std_phys = np.sqrt(var_phys)
            
            # 4. Intervals and Log-Transform for Ks
            # The user wants "Physical Parameters" (Canonical Ks) reported in Log Space (like the fit params).
            # dPs are Linear. Ks are Log.
            
            # Helper to check if param is K (Log-Space desired)
            def is_log_param(name):
                return not (name.startswith("dH") or "nuisance" in name)

            # Transform Estimate and Standard Deviation
            final_estimates = []
            final_stds = []
            final_lows = []
            final_highs = []
            
            for i, name in enumerate(phys_names):
                val = phys_vals_dict[name] # Linear Value
                std = std_phys[i]          # Linear Std
                
                if is_log_param(name):
                     # Transform to Log Space
                     # val_log = ln(val)
                     # std_log approx std / val (Delta Method d(lnx)/dx = 1/x)
                     try:
                         val_log = np.log(val)
                         # If val is negative (impossible for K), this fails. 
                         # Assuming Linear K > 0.
                         std_log = std / np.abs(val)
                     except:
                         val_log = np.nan
                         std_log = np.nan
                     
                     final_estimates.append(val_log)
                     final_stds.append(std_log)
                else:
                     # Linear (dH)
                     final_estimates.append(val)
                     final_stds.append(std)

            # Recalculate Intervals in the transformed space
            if dof is not None and dof > 0:
                 import scipy.stats
                 tcrit = scipy.stats.t.ppf(0.975, dof)
            else:
                 tcrit = 1.96

            final_estimates = np.array(final_estimates)
            final_stds = np.array(final_stds)
            
            low_phys = final_estimates - tcrit * final_stds
            high_phys = final_estimates + tcrit * final_stds
            
            results = []
            for i, name in enumerate(phys_names):
                # Filter out parameters that are already in the regression set
                # to avoid duplicates in the final dataframe.
                # Since we ensured units match (Log/Log or Lin/Lin), the Regression entry is sufficient.
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
                    "prior_std": np.nan
                })
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            # Use canonical names (no suffix)
            df.index = df["name"]
            return df
        
        return None

    def update_parameters(self, param_df):
        """
        Update parameter configurations from a dataframe (guess, bounds, fixed status).
        Handles Scaling: Converts input Cal values to internal kCal values for dH parameters.
        """
        if not set(param_df.index).issubset(set(self._parameter_names)):
            raise ValueError("Parameter names in update do not match model parameters")

        # Create dicts for update
        guesses = param_df['guess'].to_dict()
        fixed = param_df['fixed'].to_dict()
        lower = param_df['lower_bound'].to_dict()
        upper = param_df['upper_bound'].to_dict()

        for i, name in enumerate(self._parameter_names):
            if name in guesses:
                self._parameter_guesses[i] = guesses[name]
                
            if name in fixed:
                self._parameter_fixed[i] = fixed[name]
                
            if name in lower:
                self._parameter_lower_bounds[i] = lower[name]
                
            if name in upper:
                self._parameter_upper_bounds[i] = upper[name]

    def update_error_model(self, error_params):
        """
        Update the error model parameters for experiments.
        
        Parameters
        ----------
        error_params : dict
            Dictionary keyed by experiment index (int).
            Value is another dict keyed by observable name (e.g. 'heat').
            Value is dict with keys 'sigma_base' and 'f_rel'.
            
            Example:
            {
                0: { # Experiment 0
                    "heat": {"sigma_base": 0.1, "f_rel": 0.01} 
                }
            }
        """
        self._error_model_params = error_params
        # Reload observables to apply new error model
        self._load_observables()