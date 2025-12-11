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
        theta = points_per_expt/np.sum(points_per_expt)
        self._expt_std_scalar = 1 - theta + np.max(theta)

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
                    self._y_std.append(expt_data[obs_info["std_column"]])

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

    def _get_enthalpy_param(self):
        """
        Deal with enthalpy terms if needed.
        ... (docstring unchanged) ...
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

        for k in self._bm.equilibria:
            reactants = self._bm.equilibria[k][0]
            products = self._bm.equilibria[k][1]

            if len(products) <= len(reactants):
                self._dh_sign.append(1.0)
                key_species = products[:]
            else:
                self._dh_sign.append(-1.0)
                key_species = reactants[:]

            self._dh_product_mask.append(np.isin(self._bm.micro_species,
                                                    key_species))

        for s in self._bm.param_names:
            self._parameter_names.append(f"dH_{s[1:]}")
            self._parameter_guesses.append(0.0)

        # Heats of dilution
        to_dilute = []
        for expt in self._expt_list:
            for obs in expt.observables:
                if expt.observables[obs]["type"] == "itc":
                    to_dilute.extend(expt.titrating_macro_species)
        to_dilute = list(set(to_dilute))

        dh_dilution_mask = []
        for s in self._bm.macro_species:
            if s in to_dilute:
                dh_dilution_mask.append(True)
                self._parameter_names.append(f"nuisance_dil_{s}")
                self._parameter_guesses.append(0.0)
            else:
                dh_dilution_mask.append(False)

        self._dh_dilution_mask = np.array(dh_dilution_mask,dtype=bool)
        self._dh_param_end_idx = len(self._parameter_names)  - 1

    def _get_expt_fudge(self):
        """
        Fudge parameters account for uncertainty in one of the total
        concentrations each experiment. This is specified by `conc_to_float`
        when the `Experiment` class is initialized. 
        """
        self._fudge_list = []
        for expt_counter, expt in enumerate(self._expt_list):
            if expt.conc_to_float:
                param_name = f"nuisance_expt_{expt_counter}_{expt.conc_to_float}_fudge"
                self._parameter_names.append(param_name)
                self._parameter_guesses.append(1.0)
                fudge_species_index = np.where(self._bm.macro_species == expt.conc_to_float)[0][0]
                fudge_value_index = len(self._parameter_names) - 1
                self._fudge_list.append((fudge_species_index,fudge_value_index))
            else:
                self._fudge_list.append(None)

    def _add_point(self,point_idx,expt_idx,obs):
        expt = self._expt_list[expt_idx]
        obs_info = expt.observables[obs]
        data_idx = expt.expt_data.index[point_idx]
        total_volume = float(expt.expt_concs.loc[data_idx, "volume"])
        injection_volume = float(expt.expt_data.loc[data_idx, "injection"])

        if expt.expt_data.loc[data_idx, "ignore_point"]:
            return

        point_kwargs = {"idx": point_idx,
                        "expt_idx": expt_idx,
                        "obs_key": obs,
                        "micro_array": self._micro_arrays[-1],
                        "macro_array": self._macro_arrays[-1],
                        "del_macro_array": self._del_macro_arrays[-1],
                        "total_volume": total_volume,
                        "injection_volume": injection_volume}

        if obs_info["type"] == "spec":
            obs_mask = np.isin(self._bm.micro_species, obs_info["microspecies"])
            denom = np.where(self._bm.macro_species == obs_info["macrospecies"])[0][0]
            point_kwargs["obs_mask"] = obs_mask
            point_kwargs["denom"] = denom
            pt = SpecPoint(**point_kwargs)
        elif obs_info["type"] == "itc":
            point_kwargs["dh_param_start_idx"] = self._dh_param_start_idx
            point_kwargs["dh_param_end_idx"] = self._dh_param_end_idx + 1
            point_kwargs["dh_sign"] = self._dh_sign
            point_kwargs["dh_product_mask"] = self._dh_product_mask
            point_kwargs["dh_dilution_mask"] = self._dh_dilution_mask
            pt = ITCPoint(**point_kwargs)
        else:
            raise ValueError(f"The obs type '{obs_info['type']}' is not recognized\n")

        self._points.append(pt)

    def _build_point_map(self):
        self._ref_macro_arrays = []
        self._macro_arrays = []
        self._micro_arrays = []
        self._del_macro_arrays = []
        self._expt_syringe_concs = []
        self._points = []

        for expt_counter, expt in enumerate(self._expt_list):
            self._micro_arrays.append(np.ones((len(expt.expt_data),
                                            len(self._bm.micro_species)),
                                            dtype=float)*np.nan)
            
            macro_array = np.zeros((len(expt.expt_data), len(self._bm.macro_species)))
            for i, species in enumerate(self._bm.macro_species):
                macro_array[:,i] = expt.expt_concs[species].values
            self._ref_macro_arrays.append(macro_array)
            self._macro_arrays.append(self._ref_macro_arrays[-1].copy())
            
            syringe_concs = []
            for s in self._bm.macro_species:
                if s in expt.syringe_contents:
                    syringe_concs.append(expt.syringe_contents[s])
                else:
                    syringe_concs.append(0.0)
            syringe_concs = np.array(syringe_concs, dtype=float)
            
            self._expt_syringe_concs.append(syringe_concs)
            self._del_macro_arrays.append(syringe_concs - macro_array)
                    
            for obs in expt.observables:
                for i in range(len(expt.expt_data)):
                    self._add_point(point_idx=i,
                                expt_idx=expt_counter,
                                obs=obs)
            
    def model_normalized(self, parameters):
        """
        Model output where each experiment is normalized...
        ... (docstring unchanged) ...
        """
        y_calc = self.model(parameters)
        y_calc_norm = (y_calc - self._y_norm_mean)/self._y_norm_std
        return y_calc_norm

    def model(self,parameters):
        """
        Model output. Can be used to draw plots or as the target of a regression
        analysis against y_obs. 
        ... (docstring unchanged) ...
        """
        start = self._bm_param_start_idx
        end = self._bm_param_end_idx+1
    
        for i in range(len(self._macro_arrays)):
            if self._fudge_list[i] is None:
                fudge_value = 1.0
            else:
                fudge_species_index = self._fudge_list[i][0]
                fudge_value = parameters[self._fudge_list[i][1]]
            
            self._macro_arrays[i] = self._ref_macro_arrays[i].copy()
            if self._fudge_list[i] is not None:
                self._macro_arrays[i][:,fudge_species_index] *= fudge_value

            self._del_macro_arrays[i] = self._expt_syringe_concs[i] - self._macro_arrays[i]

            for j in range(len(self._macro_arrays[i])):
                self._micro_arrays[i][j,:] = self._bm.get_concs(param_array=parameters[start:end],
                                                                macro_array=self._macro_arrays[i][j,:])

        y_calc = np.ones(len(self._points))*np.nan
        for i in range(len(self._points)):
            y_calc[i] = self._points[i].calc_value(parameters)

        return y_calc

    def jacobian_normalized(self, parameters):
        """
        Calculate the Jacobian of the normalized model output with respect to
        all fittable parameters. This is d(y_calc_normalized)/d(parameters).
        This callable is suitable for use with scipy.optimize.least_squares.
        Parameters
        ----------
        parameters : np.ndarray
            Array of all current parameter values.
        Returns
        -------
        J : np.ndarray
            The Jacobian matrix of shape (num_observations, num_parameters).
        """

        # ++++++++++++++++++++++++++++++ START OF FIX ++++++++++++++++++++++++++++++
        # This function MUST NOT raise an exception. If it fails for any reason
        # (e.g., a numerical error from a bad guess), it should return a NaN matrix.
        # This allows the sampler's test call to succeed and lets the sampler
        # reject the bad step instead of crashing.
        try:
            # Run the model once to populate all concentration arrays consistently.
            self.model(parameters)

            num_obs = len(self._points)
            num_params = len(self.parameter_names)
            J = np.zeros((num_obs, num_params))

            # Build a list of Jacobians, one for each point, in a stateless way.
            start, end = self._bm_param_start_idx, self._bm_param_end_idx + 1
            bm_param_dict = dict(zip(self._bm.param_names, np.exp(parameters[start:end])))

            d_concs_d_bm_params_list = []
            for i in range(len(self._expt_list)):
                exp_jacobians = []
                for j in range(len(self._macro_arrays[i])):
                    
                    current_concs_dict = bm_param_dict.copy()
                    macro_concs = dict(zip(self._bm.macro_species, self._macro_arrays[i][j,:]))
                    current_concs_dict.update(macro_concs)
                    micro_concs = dict(zip(self._bm.micro_species, self._micro_arrays[i][j,:]))
                    current_concs_dict.update(micro_concs)
                    
                    if np.isnan(micro_concs[self._bm._c_species_name]):
                        jac = np.full((len(self._bm.micro_species), len(self._bm.param_names)), np.nan)
                    else:
                        jac = self._bm.get_numerical_jacobian(current_concs_dict)

                    if jac is None:
                        jac = np.full((len(self._bm.micro_species), len(self._bm.param_names)), np.nan)
                    
                    exp_jacobians.append(jac)
                d_concs_d_bm_params_list.append(exp_jacobians)

            for point_idx, pt in enumerate(self._points):
                expt_idx, shot_idx = pt.expt_idx, pt.idx
                d_concs_after_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx]

                if isinstance(pt, SpecPoint):
                    d_y_d_concs = pt.get_d_y_d_concs()
                    J[point_idx, start:end] = d_y_d_concs @ d_concs_after_d_bm
                
                elif isinstance(pt, ITCPoint) and pt.idx > 0:
                    d_concs_before_d_bm = d_concs_d_bm_params_list[expt_idx][shot_idx - 1]
                    d_heat_d_bm = np.zeros(len(self._bm.param_names))
                    dh_array = parameters[pt._dh_first:pt._dh_last]
                    
                    for i in range(len(pt._dh_product_mask)):
                        mask = pt._dh_product_mask[i]
                        d_C_after_d_bm = d_concs_after_d_bm[mask, :]
                        d_C_before_d_bm = d_concs_before_d_bm[mask, :]
                        d_del_C_d_bm = d_C_after_d_bm - d_C_before_d_bm * pt._meas_vol_dilution
                        d_dC_d_bm = np.mean(d_del_C_d_bm, axis=0)
                        d_heat_d_bm += dh_array[i] * pt._dh_sign[i] * d_dC_d_bm
                    
                    J[point_idx, start:end] = d_heat_d_bm * pt._total_volume

                other_param_derivs = pt.get_d_y_d_other_params(parameters)
                for param_idx, deriv_val in other_param_derivs.items():
                    J[point_idx, param_idx] = deriv_val

                if self._fudge_list[expt_idx] is not None:
                    pass 

            J[:, start:end] *= np.exp(parameters[start:end])
            J_normalized = J / self._y_norm_std[:, np.newaxis]
            return J_normalized

        except Exception as e:
            # If any failure occurs, log it and return a NaN matrix of the correct shape.
            tb_str = traceback.format_exc()
            warnings.warn(f"Jacobian calculation failed with error: {e}\n{tb_str}")
            num_obs = len(self._points)
            num_params = len(self.parameter_names)
            return np.full((num_obs, num_params), np.nan)
        # +++++++++++++++++++++++++++++++ END OF FIX +++++++++++++++++++++++++++++++


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
        if self._model_name == "GenericBindingModel":
            return self._bm.final_ct
        return None
    
    @property
    def model_spec(self):
        if self._model_name == "GenericBindingModel":
            return self._model_spec
        return None
    
    @property
    def simplified_equations(self):
        if self._model_name == "GenericBindingModel":
            return self._bm.simplified_eqs
        return None
    
    @property
    def solved_vars(self):
        if self._model_name == "GenericBindingModel":
            return self._bm.solved_vars
        return None
    
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