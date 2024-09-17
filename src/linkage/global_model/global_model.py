import linkage.models
from linkage.global_model.point.spec_point import SpecPoint
from linkage.global_model.point.itc_point import ITCPoint

import numpy as np
import pandas as pd

import copy

class GlobalModel:
    """
    This class brings together a list of experiments and a thermodynamic model
    and generates an integrated model. This model will have the following 
    parameters:
    
    + equilibrium constants from model
    + a nuisance concentration parameter for each experiment (if requested)
    + enthalpies for each of the model equilibria and heats of dilution for 
      titrating species (if at least one ITC experiment is passed in)

    This class also:
    
    + Regularizes the signal from experimental types. For example,
      the heats from across all itc experiments will be transformed by
      (heat - mean(all_heats))/std(all_heats), where all_heats comes from all 
      itc experiments loaded. The same transformation will be done to each
      spectroscopic channel. This puts all experiment types on the same scale
      when a residual is calculated. 

    + Weights each observation in each experiment by the number of
      points in that experiment. This means that an experiment with more points 
      will have the same weight overall contribution to the regression as an
      experiment with fewer points. 
    """

    def __init__(self,
                 expt_list,
                 model_name):
        """
        Initialize a global fit.
        
        Parameters
        ----------
        expt_list : list
            list of experiments with loaded observations
        model_name : str
            name of model to use to calculate concentrations
        """

        # Store model name and experiment list
        self._model_name = model_name
        self._expt_list = copy.deepcopy(expt_list)
    
        # Load the model
        self._load_model()

        # Load experimental data. The final output of this 
        self._get_expt_std_scalar()
        self._get_expt_normalization()
        self._load_observables()

        self._get_enthalpy_param()
        self._get_expt_fudge()
        
        # Create points that allow calculation of observations
        self._build_point_map()

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
        self._bm = available_models[self._model_name]()

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
        
        Enthalpy change over a titration step is determined by change in
        the concentration of microscopic species from the equilibrium. 
        Ideally, there is a single species on one side of the reaction, 
        so we can simply measure the change in the concentration of that
        species. This block of code figures out which side of the 
        equilibrium has fewer species and declares that the "product" for
        accounting purposes. dh_sign records whether this is the right
        side of the reaction (forward) with +1 or the left side of the 
        reaction (backward) with -1. By applying dh_sign, the final 
        enthalpy is always correct relative to the reaction definition. 
        """

        # Look for an ITC experiment
        need_enthalpies = False
        for expt in self._expt_list:
            for obs in expt.observables:    
                if expt.observables[obs]["type"] == "itc":
                    need_enthalpies = True
                    break

        # If we do not need enthalpies, return without doing anything
        if not need_enthalpies:    
            return 

        # Index of first enthalpy
        self._dh_param_start_idx = len(self._parameter_names)
        
        # ------------------------------------------------------------------
        # Reaction enthalpies

        self._dh_sign = []
        self._dh_product_mask = []
        
        # Create an enthalpy term (with associated dh_sign and dh_product_mask)
        # for each equilibrium. 
        for k in self._bm.equilibria:

            # Get products and reactants of this equilibrium
            reactants = self._bm.equilibria[k][0]
            products = self._bm.equilibria[k][1]

            # Figure out if products or reactants side has fewer species
            if len(products) <= len(reactants):
                self._dh_sign.append(1.0)
                key_species = products[:]
            else:
                self._dh_sign.append(-1.0)
                key_species = reactants[:]

            # Create a mask that lets us grab the species we need to track 
            # from the _micro_array array. 
            self._dh_product_mask.append(np.isin(self._bm.micro_species,
                                                    key_species))

        # Record enthalpies as parameters
        for s in self._bm.param_names:
            self._parameter_names.append(f"dH_{s[1:]}")
            self._parameter_guesses.append(0.0)

        # ------------------------------------------------------------------
        # Heats of dilution. 

        # Figure out which species are being diluted when they go into the
        # cell from the syringe. 
        to_dilute = []
        for expt in self._expt_list:
            for obs in expt.observables:
                if expt.observables[obs]["type"] == "itc":
                    to_dilute.extend(expt.titrating_macro_species)
        to_dilute = list(set(to_dilute))
        
        # Add heat of dilution parameters to the parameter array. Construct
        # the dilution_mask to indicate which macro species these 
        # correspond to. 
        dh_dilution_mask = []
        for s in self._bm.macro_species:
            if s in to_dilute:
                dh_dilution_mask.append(True)
                self._parameter_names.append(f"nuisance_dil_{s}")
                self._parameter_guesses.append(0.0)
            else:
                dh_dilution_mask.append(False)

        self._dh_dilution_mask = np.array(dh_dilution_mask,dtype=bool)
                
        # Last enthalpy index is last entry
        self._dh_param_end_idx = len(self._parameter_names)  - 1
    
    def _get_expt_fudge(self):
        """
        Fudge parameters account for uncertainty in one of the total
        concentrations each experiment. This is specified by `conc_to_float`
        when the `Experiment` class is initialized. 
        """

        # Fudge parameters will be last parameters in the guess array   
        self._fudge_list = []
        for expt_counter, expt in enumerate(self._expt_list):
            
            # If an experiment has a conc_to_float specified, create a parameter
            # and initialize it. 
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

        # Information about observable and experimental data
        expt = self._expt_list[expt_idx]
        obs_info = expt.observables[obs]
        
        data_idx = expt.expt_data.index[point_idx]
        total_volume = float(expt.expt_concs.loc[data_idx,"volume"])
        injection_volume = float(expt.expt_data.loc[data_idx,"injection"])

        if expt.expt_data.loc[data_idx,"ignore_point"]:
            return
            
        point_kwargs = {"idx":point_idx,
                        "expt_idx":expt_idx,
                        "obs_key":obs,
                        "micro_array":self._micro_arrays[-1],
                        "macro_array":self._macro_arrays[-1],
                        "del_macro_array":self._del_macro_arrays[-1],
                        "total_volume":total_volume,
                        "injection_volume":injection_volume}

        if obs_info["type"] == "spec":

            obs_mask = np.isin(self._bm.micro_species,obs_info["microspecies"])
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
            obs_type = obs_info["type"]
            err = f"The obs type '{obs_type}' is not recognized\n"
            raise ValueError(err)

        self._points.append(pt)

    def _build_point_map(self):

        # Lists of arrays that can be referenced by all points in the 
        # experiments. There is an entry for each experiment. The values in 
        # these arrays are set globally. 
        self._ref_macro_arrays = []
        self._macro_arrays = []
        self._micro_arrays = []
        self._del_macro_arrays = []
        self._expt_syringe_concs = []

        # List of all points
        self._points = []

        for expt_counter, expt in enumerate(self._expt_list):

            # Each experiment has:
            
            # 1. An array of microscopic species concentrations
            self._micro_arrays.append(np.ones((len(expt.expt_data),
                                                len(self._bm.micro_species)),
                                               dtype=float)*np.nan)

            # 2. An array of macroscopic species concentrations
            macro_array = np.array(expt.expt_concs.loc[:,self._bm.macro_species],
                                   dtype=float).copy()
            self._ref_macro_arrays.append(macro_array)
            self._macro_arrays.append(self._ref_macro_arrays[-1].copy())

            # 3. An array of the change in macro species relative to syringe. 
            syringe_concs = []
            for s in self._bm.macro_species:
                if s in expt.syringe_contents:
                    syringe_concs.append(expt.syringe_contents[s])
                else:
                    syringe_concs.append(0.0)
            
            syringe_concs = np.array(syringe_concs,dtype=float)
            self._expt_syringe_concs.append(syringe_concs)
            self._del_macro_arrays.append(syringe_concs - macro_array)
                    
            # For each observable
            for obs in expt.observables:

                # Go through each experimental point
                for i in range(len(expt.expt_data)):

                    # Add that point to the list of all points. The final list 
                    # of points will exactly match the values in y_obs, y_std,
                    # etc.
                    self._add_point(point_idx=i,
                                    expt_idx=expt_counter,
                                    obs=obs)
            
    def model_normalized(self,parameters):
        """
        Model output where each experiment is normalized to its experimental
        mean, standard deviation, and number of experimental points. This 
        is useful for regression because each point contributes the same amount
        to the regression. 
        
        Parameters
        ----------
        parameters : np.ndarray
            array of parameter values corresponding to the parameters in 
            self.parameter_names
        
        Returns
        -------
        y_calc_norm : np.ndarray
            array of outputs calculated across conditions. pairs with 
            self.y_obs_normalized and and self.y_std_normalized

        Note
        ----
        This should be regressed against self.y_obs_normalized and 
        self.y_std_normalized, *not* self.y_obs and self.y_std. The resulting 
        parameter estimates should then reproduce self.y_obs if passed back into
        self.model. 
        """

        # Run model un-normalized (which updates self._y_calc)
        y_calc = self.model(parameters)

        # Now normalize y_calc_norm
        y_calc_norm = (y_calc - self._y_norm_mean)/self._y_norm_std

        # Return
        return y_calc_norm

    def model(self,parameters):
        """
        Model output. Can be used to draw plots or as the target of a regression
        analysis against y_obs. 

        Parameters
        ----------
        parameters : np.ndarray
            array of parameter values corresponding to the parameters in 
            self.parameter_names
        
        Returns
        -------
        y_calc : np.ndarray
            array of outputs calculated across conditions. pairs with self.y_obs
            and self.y_std
        """

        # Grab binding parameters from guesses. 
        start = self._bm_param_start_idx
        end = self._bm_param_end_idx+1
    
        # For each experiment, update the macro_arrays (which might change due
        # to a fudge factor) and then update micro_arrays (which might change 
        # due to change in macro_array and/or changes in model parameters)
        for i in range(len(self._macro_arrays)):

            # Figure if/how to fudge one of the macro array concentrations
            if self._fudge_list[i] is None:
                fudge_species_index = 0
                fudge_value = 1.0
            else:
                fudge_species_index = self._fudge_list[i][0]
                fudge_value = parameters[self._fudge_list[i][1]]
            
            # Get reference macro array without any fudge factor
            self._macro_arrays[i] = self._ref_macro_arrays[i].copy()

            # Fudge the macro array
            self._macro_arrays[i][:,fudge_species_index] *= fudge_value

            # Update del_macro_array
            self._del_macro_arrays[i] = self._expt_syringe_concs[i] - self._macro_arrays[i]

            # For each titration step in this experiment (row of concs in 
            # marco_arrays[i]), update _micro_arrays[i] with the binding model
            for j in range(len(self._macro_arrays[i])):

                self._micro_arrays[i][j,:] = self._bm.get_concs(param_array=parameters[start:end],
                                                                macro_array=self._macro_arrays[i][j,:])

        # For each point, calculate the observable given the estimated microscopic
        # and macroscopic concentrations
        y_calc = np.ones(len(self._points))*np.nan
        for i in range(len(self._points)):
            y_calc[i] = self._points[i].calc_value(parameters)

        return y_calc

    @property
    def y_obs(self):
        """
        Vector of observed values.
        """
        return self._y_obs

    @property
    def y_std(self):
        """
        Vector of standard deviations of observed values.
        """
        return self._y_std

    @property
    def y_obs_normalized(self):
        """
        Vector of observed values where each experiment is normalized by
        (obs - mean(obs))/std(obs). Pairs with y_calc_normalized and 
        y_std_normalized.
        """
        return self._y_obs_normalized
    
    @property
    def y_std_normalized(self):
        """
        Vector of standard deviations of observed values normalized by 
        (y_std*expt_std_scalar)/std(obs). Pairs with y_obs_normalized. 
        """
        return self._y_std_normalized

    @property
    def parameter_names(self):
        """
        Names of all fit parameters, in stable order.
        """
        return self._parameter_names
            
    @property
    def parameter_guesses(self):
        """
        Parameter values from last run of the model.
        """
        return self._parameter_guesses
    
    @property
    def model_name(self):
        """
        Name of the underlying linkage model used in the analysis.
        """
        return self._model_name
    
    @property
    def macro_species(self):
        """
        Names of all macrospecies, in stable order expected by the linkage 
        model. 
        """
        return self._bm.macro_species
    
    @property
    def micro_species(self):
        """
        Names of all microspecies, in stable order expected by the linkage 
        model. 
        """
        return self._bm.micro_species
    
    @property
    def as_df(self):

        out = {"expt_id":[],
               "expt_type":[],
               "expt_obs":[],
               "volume":[],
               "injection":[]}

        for k in self._bm.macro_species:
            out[k] = []

        for k in self._bm.micro_species:
            out[k] = []

        for p in self._points:
            
            out["expt_id"].append(p.expt_idx)

            if issubclass(type(p),SpecPoint):
                out["expt_type"].append(p.obs_key)

                num = "+".join(self._bm.micro_species[p._obs_mask])
                den = self._bm.macro_species[p._denom]
                out["expt_obs"].append(f"{num}/{den}")
            
            elif issubclass(type(p),ITCPoint):
                out["expt_type"].append("itc")

                out["expt_obs"].append("obs_heat")

            else:
                err = "point class not recognized\n"
                raise ValueError(err)

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

        