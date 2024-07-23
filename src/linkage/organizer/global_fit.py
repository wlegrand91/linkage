import linkage.models
from linkage.experiment.experimental_point import SpecPoint
from linkage.experiment.experimental_point import ITCPoint

import numpy as np

import copy

class GlobalFit:

    def __init__(self,expt_list,model_name="SixStateEDTA"):
        """
        Initialize a global fit.
        
        Parameters
        ----------
        expt_list : list
            list of experiments with loaded observations
        model_name : str
            name of model to use for the analysis
        """

        # Store model name and experiment list
        self._model_name = model_name
        self._expt_list = copy.deepcopy(expt_list)

        # Indexes for slicing out binding model and enthalpy parameters
        self._bm_param_start_idx = None
        self._bm_param_end_idx = None
        self._dh_param_start_idx = None
        self._dh_param_end_idx = None

        # list of all parameter names in the same order as guesses
        self._all_parameter_names = []
        
        # Initialize class
        self._load_model()
        self._sync_model_and_expt()
        self._count_expt_points()
        self._get_enthalpy_param()
        self._process_expt_fudge()
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

        # Last binding model parameter index is last value
        self._bm_param_end_idx = len(self._all_parameter_names) - 1
            
    def _sync_model_and_expt(self):
        """
        Make sure that all experiments have all concentrations required for the
        model. 
        """
        
        # If an experiment is missing a macro species, set the concentration of
        # that molecule to 0.0 over whole titration. 
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
                if expt.observables[obs]["obs_type"] == "itc":
                    need_enthalpies = True
                    break

        # If we need enthalpies
        if need_enthalpies:
            
            # Index of first enthalpy
            self._dh_param_start_idx = len(self._all_parameter_names)
            
            # Names for all enthalpies
            for s in self._bm.micro_species:
                self._all_parameter_names.append(f"dH_{s}")

            # Last enthalpy index is last entry
            self._dh_param_end_idx = len(self._all_parameter_names) - 1
    

    def _process_expt_fudge(self):

        # Fudge parameters will be last parameters in the guess list    
        self._fudge_list = []
        for expt_counter, expt in enumerate(self._expt_list):
            
            if expt.conc_to_float:

                param_name = f"expt_{expt_counter}_{expt.conc_to_float}_fudge"
                self._all_parameter_names.append(param_name)
                
                fudge_species_index = np.where(self._bm.macro_species == expt.conc_to_float)[0][0]
                fudge_value_index = len(self._all_parameter_names) - 1
                
                self._fudge_list.append((fudge_species_index,fudge_value_index))
                        
            else:
                self._fudge_list.append(None)
        
    def _build_point_map(self):

        # Lists of arrays that can be referenced by the individual points
        self._micro_arrays = []
        self._macro_arrays = []

        self._points = []
        self._y_obs = []
        self._y_stdev = []
        
        for expt_counter, expt in enumerate(self._expt_list):

            # Each experiment will have a an array of microscopic species concentrations
            self._micro_arrays.append(np.zeros((len(expt.expt_data),
                                                len(self._bm.micro_species)),
                                               dtype=float))

            # ... and an array of macroscopic species concentrations
            macro_array = np.array(expt.expt_concs.loc[:,self._bm.macro_species],
                                   dtype=float)
            self._macro_arrays.append(macro_array)
                    
            for obs in expt.observables:
                
                obs_info = expt.observables[obs]
            
                for i in range(len(expt.expt_data)):
                    
                    expt_data = expt.expt_data.loc[expt.expt_data.index[i],:]
                    
                    if expt_data["ignore_point"]:
                        continue
                           
                    if obs_info["obs_type"] == "spec":

                        pt = SpecPoint(idx=i,
                                       expt_idx=expt_counter,
                                       obs_key=obs,
                                       obs_mask=np.isin(self._bm.micro_species,
                                                        obs_info["observable_species"]),
                                       denom=obs_info["denominator"],
                                       micro_array=self._micro_arrays[-1],
                                       macro_array=self._macro_arrays[-1])
                        
                    elif obs_info["obs_type"] == "itc":
                        
                        pt = ITCPoint(idx=i,
                                      expt_idx=expt_counter,
                                      obs_key=obs,
                                      dh_param_start_idx=self._dh_param_start_idx,
                                      dh_param_end_idx=self._dh_param_end_idx,
                                      micro_array=self._micro_arrays[-1])
        
                    else:
                        obs_type = obs_info["obs_type"]
                        err = f"The obs type '{obs_type}' is not recognized\n"
                        raise ValueError(err)
        
                    self._points.append(pt)
                    self._y_obs.append(expt_data[obs])
                    self._y_stdev.append(self._points_per_expt[expt_counter])

        self._y_calc = np.ones(len(self._y_obs),dtype=float)*np.nan
        self._y_obs = np.array(self._y_obs)
        self._y_stdev = np.array(self._y_stdev)/np.sum(self._y_stdev)
    
    def total_model(self,guesses):
    
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

                # Grab parameters from guesses
                this_param_array = guesses[self._bm_param_start_idx:self._bm_param_end_idx+1]

                # Update microscopic species concentrations
                self._micro_arrays[i][j,:] = self._bm.get_concs(param_array=this_param_array,
                                                                macro_array=this_macro_array)

        # For each point, calculate the observable
        for i in range(len(self._points)):
            self._y_calc[i] = self._points[i].calc_value(guesses)
        
        return self._y_calc

    @property
    def y_calc(self):
        return self._y_calc
    
    @property
    def y_obs(self):
        return self._y_obs

    @property
    def y_stdev(self):
        return self._y_stdev

    @property
    def parameter_names(self):
        return self._all_parameter_names
            
    
        