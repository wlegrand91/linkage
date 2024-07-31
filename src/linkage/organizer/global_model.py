import linkage.models
from linkage.experiment.point.spec_point import SpecPoint
from linkage.experiment.point.itc_point import ITCPoint

import numpy as np
import pandas as pd

import copy

class GlobalModel:
    """
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

        # Indexes for slicing out binding model
        self._bm_param_start_idx = None
        self._bm_param_end_idx = None

        # Indexes for slicing out and processing
        self._dh_param_start_idx = None
        self._dh_param_end_idx = None
        self._dh_sign = None
        self._dh_product_mask = None

        # list of all parameter names in the same order as guesses
        self._all_parameter_names = []
        self._parameter_guesses = []
        
        # Initialize class
        self._load_model()
        self._sync_model_and_expt()
        self._count_expt_points()
        self._get_enthalpy_param()
        self._process_expt_fudge()
        self._calc_expt_normalization()
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

        # First binding model parameter index is 0
        self._bm_param_start_idx = 0
        
        # Record names of the model parameters
        for p in self._bm.param_names:
            self._all_parameter_names.append(p)
            self._parameter_guesses.append(0.0)

        # Last binding model parameter index is last value
        self._bm_param_end_idx = len(self._all_parameter_names) - 1
            
    def _sync_model_and_expt(self):
        """
        Make sure that all experiments have concentrations for all macrospecies
        used by the model. If an experiment is missing a macro species, set the
        concentration of that macrospecies to 0.0 over the whole experiment. 
        """
        
        for expt in self._expt_list:
            not_in_expt = set(self._bm.macro_species) - set(expt.expt_concs.columns)
            for missing in not_in_expt:
                expt.add_expt_conc_column(new_column=missing)

    def _count_expt_points(self):
        """
        Count the number of data points for a given experiment.
        """
        
        # Count all points for each experiment 
        self._points_per_expt = []
        for expt in self._expt_list:
            num_obs = len(expt.observables)
            num_not_ignore = np.sum(np.logical_not(expt.expt_data["ignore_point"]))
            self._points_per_expt.append(num_obs*num_not_ignore)

    def _get_enthalpy_param(self):
        """
        Decide if we need to include enthalpies to fit ITC data.
        """

        # Look for an ITC experiment
        need_enthalpies = False
        for expt in self._expt_list:
            for obs in expt.observables:    
                if expt.observables[obs]["type"] == "itc":
                    need_enthalpies = True
                    break

        # If we need enthalpies
        if need_enthalpies:
            
            # Index of first enthalpy
            self._dh_param_start_idx = len(self._all_parameter_names)
            
            # Enthalpy change over a titration step is determined by change in
            # the concentration of microscopic species from the equilibrium. 
            # Ideally, there is a single species on one side of the reaction, 
            # so we can simply measure the change in the concentration of that
            # species. This block of code figures out which side of the 
            # equilibrium has fewer species and declares that the "product" for
            # accounting purposes. dh_sign records whether this is the right
            # side of the reaction (forward) with +1 or the left side of the 
            # reaction (backward) with -1. By applying dh_sign, the final 
            # enthalpy is always correct relative to the reaction definition. 
            self._dh_sign = []
            self._dh_product_mask = []
            
            # For each equilibrium
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

            # Names for all enthalpies
            for s in self._bm.param_names:
                self._all_parameter_names.append(f"dH_{s[1:]}")
                self._parameter_guesses.append(0.0)

            # Heats of dilution
            for k in self._bm.macro_species:
                self._all_parameter_names.append(f"nuisance_dil_{k}")
                self._parameter_guesses.append(0.0)

            # Last enthalpy index is last entry
            self._dh_param_end_idx = len(self._all_parameter_names)  - 1
    

    def _process_expt_fudge(self):

        # Fudge parameters will be last parameters in the guess list    
        self._fudge_list = []
        for expt_counter, expt in enumerate(self._expt_list):
            
            if expt.conc_to_float:

                param_name = f"nuisance_expt_{expt_counter}_{expt.conc_to_float}_fudge"
                self._all_parameter_names.append(param_name)
                self._parameter_guesses.append(1.0)
                
                fudge_species_index = np.where(self._bm.macro_species == expt.conc_to_float)[0][0]
                fudge_value_index = len(self._all_parameter_names) - 1
                
                self._fudge_list.append((fudge_species_index,fudge_value_index))
                        
            else:
                self._fudge_list.append(None)
    
    def _calc_expt_normalization(self):
        """
        Figure out how to normalize. Each unique 'obs' seen (e.g. heat, cd222, 
        etc.) is normalized to all values of that obs type seen across all 
        experiments. So, if there are three itc experiments, we will do a single
        normalization across all three experiments. The normalization is 
        (value - mean(value))/stdev(value) where the mean and stdev are taken 
        over all experimental values with that obs. 
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
        # stdev of that obs. This allows other methods to normalize data on 
        # the fly. 
        self._normalization_params = {}
        for obs in obs_values_seen:
            
            self._normalization_params[obs] = {}
            
            values = np.array(obs_values_seen[obs])
            values = values[np.logical_not(np.isnan(values))]
            if len(values) == 0:
                mean_value = 0
                stdev_value = 1
            else:
                mean_value = np.mean(values)
                stdev_value = np.std(values)
            
            self._normalization_params[obs]["mean"] = mean_value
            self._normalization_params[obs]["stdev"] = stdev_value


    def _add_point(self,point_idx,expt_idx,obs):

        # Information about observable and experimental data
        expt = self._expt_list[expt_idx]
        obs_info = expt.observables[obs]
        expt_data = expt.expt_data.loc[expt.expt_data.index[point_idx],:]
        
        if expt_data["ignore_point"]:
            return
            
        if obs_info["type"] == "spec":

            den_index = np.where(self._bm.macro_species == obs_info["macrospecies"])[0][0]

            pt = SpecPoint(idx=point_idx,
                           expt_idx=expt_idx,
                           obs_key=obs,
                           micro_array=self._micro_arrays[-1],
                           macro_array=self._macro_arrays[-1],
                           del_macro_array=self._del_macro_arrays[-1],
                           obs_mask=np.isin(self._bm.micro_species,
                                               obs_info["microspecies"]),
                           denom=den_index)
            
        elif obs_info["type"] == "itc":

            meas_vol_dilution = expt.expt_concs.loc[expt.expt_data.index[point_idx],
                                                    "meas_vol_dilution"]
            
            pt = ITCPoint(idx=point_idx,
                          expt_idx=expt_idx,
                          obs_key=obs,
                          micro_array=self._micro_arrays[-1],
                          macro_array=self._macro_arrays[-1],
                          del_macro_array=self._del_macro_arrays[-1],
                          meas_vol_dilution=meas_vol_dilution,
                          dh_param_start_idx=self._dh_param_start_idx,
                          dh_param_end_idx=self._dh_param_end_idx + 1,
                          dh_sign=self._dh_sign,
                          dh_product_mask=self._dh_product_mask)

        else:
            obs_type = obs_info["type"]
            err = f"The obs type '{obs_type}' is not recognized\n"
            raise ValueError(err)

        # Record point, observations, and standard deviation
        self._points.append(pt)
        self._y_obs.append(expt_data[obs])
        self._y_stdev.append(expt_data[obs_info["stdev_column"]])

        # Get mean and stdev of obs for normalization
        obs_mean = self._normalization_params[obs]["mean"]
        obs_stdev = self._normalization_params[obs]["stdev"]

        # Record point normalization
        self._y_norm_mean.append(obs_mean)
        self._y_norm_stdev.append(obs_stdev)
        self._y_obs_normalized.append((expt_data[obs] - obs_mean)/obs_stdev)
        self._y_stdev_normalized.append(self._y_stdev[-1]/obs_stdev)


    def _build_point_map(self):

        # Lists of arrays that can be referenced by the individual points
        self._micro_arrays = []
        self._macro_arrays = []
        self._del_macro_arrays = []
        self._expt_syringe_concs = []

        # Points
        self._points = []

        # Observed values
        self._y_obs = []
        self._y_stdev = []

        # Normalized observed values (and how to do it)
        self._y_obs_normalized = []
        self._y_stdev_normalized = []

        self._y_norm_mean = []
        self._y_norm_stdev = []
        
        for expt_counter, expt in enumerate(self._expt_list):

            # Each experiment has:
            # 
            # 1. An array of microscopic species concentrations
            self._micro_arrays.append(np.ones((len(expt.expt_data),
                                                len(self._bm.micro_species)),
                                               dtype=float)*np.nan)

            # 2. An array of macroscopic species concentrations
            macro_array = np.array(expt.expt_concs.loc[:,self._bm.macro_species],
                                   dtype=float)
            self._macro_arrays.append(macro_array)

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

                    # And try to add it
                    self._add_point(point_idx=i,
                                    expt_idx=expt_counter,
                                    obs=obs)
                    
        # Convert lists populated above into numpy arrays
        self._y_obs = np.array(self._y_obs)
        self._y_stdev = np.array(self._y_stdev)

        self._y_norm_mean = np.array(self._y_norm_mean)
        self._y_norm_stdev = np.array(self._y_norm_stdev)
        
        self._y_obs_normalized = np.array(self._y_obs_normalized)
        self._y_stdev_normalized = np.array(self._y_stdev_normalized)

        # Create calc arrays that are all nan at this point
        self._y_calc = np.ones(len(self._y_obs),dtype=float)*np.nan
        self._y_calc_normalized = np.ones(len(self._y_obs),dtype=float)*np.nan
    
    def model_normalized(self,guesses):
        """
        Model where each experiment is normalized to its mean and standard 
        deviation. Should be regressed against self.y_obs_normalized and 
        self.y_stdev_normalized, *not* self.y_obs and self.y_stdev. 

        When this method is run, self.y_calc will be updated properly, 
        allowing self.y_calc and self.y_obs to be directly compared, even after
        regressing against normalized values. 
        """

        # Run model un-normalized (which updates self._y_calc)
        y_calc = self.model(guesses)

        # Now normalize y_calc_norm
        self._y_calc_norm = (y_calc - self._y_norm_mean)/self._y_norm_stdev

        # Return
        return self._y_calc_norm

    def model(self,guesses):
        """
        """
    
        # For each block of macro species
        for i in range(len(self._macro_arrays)):
        
            # Figure out if we are fudging a macro array species concentration
            if self._fudge_list[i] is not None:
                fudge_species_index = self._fudge_list[i][0]
                fudge_value = guesses[self._fudge_list[i][1]]
            else:
                fudge_species_index = 0
                fudge_value = 1.0

            # For each macro concentration in this experiment
            for j in range(len(self._macro_arrays[i])):

                # Build a vector with macro concentrations, possibly with one 
                # fudged
                this_macro_array = self._macro_arrays[i][j,:]
                this_macro_array[fudge_species_index] *= fudge_value

                # Grab binding parameters from guesses. 
                this_param_array = np.float_power(10,guesses[self._bm_param_start_idx:self._bm_param_end_idx+1])

                # Update microscopic species concentrations
                self._micro_arrays[i][j,:] = self._bm.get_concs(param_array=this_param_array,
                                                                macro_array=this_macro_array)

            # Update del_macro_array
            this_macro_array = self._macro_arrays[i].copy()
            this_macro_array[fudge_species_index] *= fudge_value
            self._del_macro_arrays[i] = self._expt_syringe_concs[i] - this_macro_array

        # For each point, calculate the observable given the estimated microscopic
        # and macroscopic concentrations
        for i in range(len(self._points)):
            self._y_calc[i] = self._points[i].calc_value(guesses)
        
        return self._y_calc

    @property
    def y_calc(self):
        """
        Vector of calculated values.
        """
        return self._y_calc
    
    @property
    def y_obs(self):
        """
        Vector of observed values.
        """
        return self._y_obs

    @property
    def y_stdev(self):
        """
        Vector of standard deviations of observed values.
        """
        return self._y_stdev

    @property
    def y_calc_normalized(self):
        """
        Vector of calculated values where each experiment is normalized by
        (obs - mean(obs))/stdev(obs). Pairs with y_obs_normalized.
        """
        return self._y_calc_normalized

    @property
    def y_obs_normalized(self):
        """
        Vector of observed values where each experiment is normalized by
        (obs - mean(obs))/stdev(obs). Pairs with y_calc_normalized and 
        y_stdev_normalized.
        """
        return self._y_obs_normalized
    
    @property
    def y_stdev_normalized(self):
        """
        Vector of standard deviations of observed values normalized by 
        (obs - mean(obs))/stdev(obs). Pairs with y_obs_normalized. 
        """
        return self._y_stdev_normalized

    @property
    def parameter_names(self):
        """
        Names of all fit parameters, in stable order.
        """
        return self._all_parameter_names
            
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
        return self._micro_species
    
    @property
    def as_df(self):

        out = {"expt_id":[],
               "expt_type":[],
               "expt_obs":[]}

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

                out["expt_obs"].append("heat")

            else:
                err = "point class not recognized\n"
                raise ValueError(err)

            for i, k in enumerate(self._bm.macro_species):
                out[k].append(p._macro_array[p._idx,i])

            for i, k in enumerate(self._bm.micro_species):
                out[k].append(p._micro_array[p._idx,i])
            
        out["y_obs"] = self.y_obs
        out["y_calc"] = self.y_calc
        out["y_stdev"] = self.y_stdev

        return pd.DataFrame(out)

        