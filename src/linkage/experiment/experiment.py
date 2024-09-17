from linkage.experiment.titrator import titrator
from linkage.experiment.titrator import sync_cell_and_syringe

import numpy as np
import pandas as pd

import copy
import warnings

def _load_dataframe(expt_data):

    # If this is a string, try to load it as a file
    if type(expt_data) is str:

        filename = expt_data

        ext = filename.split(".")[-1].strip().lower()

        if ext in ["xlsx","xls"]:
            df = pd.read_excel(filename)
        elif ext == "csv":
            df = pd.read_csv(filename,sep=",")
        elif ext == "tsv":
            df = pd.read_csv(filename,sep="\t")
        else:
            # Fall back -- try to guess delimiter
            df = pd.read_csv(filename,sep=None,engine="python")

    # If this is a pandas dataframe, work in a copy of it.
    elif type(expt_data) is pd.DataFrame:
        df = expt_data.copy()

    # Otherwise, fail
    else:
        err = f"\n\n'expt_data' {expt_data} not recognized. Should be the\n"
        err += "filename of a spreadsheet or a pandas dataframe.\n"
        raise ValueError(err)
    
    return df


def _preprocess_df(expt_data):

    if "injection" not in expt_data.columns:
        err = "expt_data should be a dataframe or spreadsheet with a 'injection' column\n"
        raise ValueError(err)

    if "ignore_point" not in expt_data.columns:
        expt_data["ignore_point"] = np.zeros(len(expt_data),dtype=bool)

    # If there is no "0" injections start -- like in ITC, for example -- add this with 
    # np.nan for all non-injection values
    if not np.isclose(expt_data.loc[expt_data.index[0],"injection"],0):
        new_row = dict([(c,[np.nan]) for c in expt_data.columns])
        new_row["injection"] = [0.0]
        new_row["ignore_point"] = [True]
        new_row = pd.DataFrame(new_row)
        expt_data = pd.concat((new_row,expt_data),ignore_index=True)

    return expt_data


class Experiment:

    def __init__(self,
                 expt_data,
                 cell_contents,
                 syringe_contents,
                 cell_volume,
                 conc_to_float=None,
                 constant_volume=False):
        """
        Load an experimental dataset.
        
        Parameters
        ----------
        expt_data : str or pandas.DataFrame
            DataFrame with experimental data. If passed in as a string, try to 
            read from file. Otherwise, use as dataframe. DataFrame must have 
            an 'injection' column holding injection volumes for experiments.
        cell_contents : dict
            dictionary keying macrospecies from the binding model to their 
            concentrations in the cell prior to any injection
        syringe_contents : dict
            dictionary keying macrospecies from the binding model to their
            concentrations in the syringe. 
        cell_volume : float
            volume of the cell into which the titration is done. the units of
            this volume should match the units of the 'injection' column. 
        conc_to_float : str, optional
            name of the most macrospecies with the most uncertain concentration
            in the experiment (usually a macromolecule). If specified, this 
            concentration will be allowed to float in the fit (look for a 
            "fudge" parameter). 
        constant_volume : bool, default=False
            if True, an injection with volume X uL is treated as "suck out X uL
            and send to waste, then add X uL from syringe." if False, each 
            injection increases the total volume. 
        """

        # Process and clean up input experimental data
        expt_data = _load_dataframe(expt_data=expt_data)
        self._expt_data = _preprocess_df(expt_data=expt_data)
                        
        # Process and clean up cell and syringe data
        out = sync_cell_and_syringe(cell_contents,syringe_contents)
        _, titrating_species, cell_contents, syringe_contents = out
        self._initial_cell_contents = copy.deepcopy(cell_contents)
        self._syringe_contents = copy.deepcopy(syringe_contents)
        self._titrating_macro_species = copy.deepcopy(titrating_species)
            
        # Calculate the total concentrations over the experiment
        self._expt_concs = titrator(cell_contents=cell_contents,
                                    syringe_contents=syringe_contents,
                                    injection_array=np.array(self._expt_data["injection"]),
                                    cell_volume=cell_volume,
                                    constant_volume=constant_volume)
        
        # Make sure conc_to_float is sane. 
        if conc_to_float is not None:
            if conc_to_float not in self._expt_concs.columns:
                err = "conc_to_float is not a macrospecies in the experiment\n"
                raise ValueError(err)

        self._conc_to_float = conc_to_float
        self._observables = {}

    def _define_generic_observable(self,
                                   obs_column,
                                   obs_std):
        """
        Define observable, making sure obs_column and obs_std columns are 
        sane. If obs_std is a float, make a new column named {obs_column}_std
        with the value loaded in. 
        """
            
        if obs_column not in self._expt_data.columns:
            err = f"obs_column '{obs_column}' should be one of the columns in the experimental data\n"
            raise ValueError(err)

        if obs_column == "injection":
            err = "obs_column cannot be 'injection'\n"
            raise ValueError(err)
    
        # Deal with uncertainty
        if obs_std in self._expt_data.columns:
            obs_std_column = obs_std
        else:
            obs_std_column = f"{obs_column}_std"
            
            try:
                obs_std = float(obs_std)
            except Exception as e:
                err = "obs_std should be either the name of a column or a single value\n"
                raise ValueError(err) from e
            
            self._expt_data.loc[:,obs_std_column] = obs_std

        if obs_column in self._observables:
            w = f"obs_column '{obs_column}' was already adding. Overwriting\n"
            warnings.warn(w)

        # Make sure any missing data is ignored. 
        set_to_ignore = np.isnan(self._expt_data[obs_column])
        self._expt_data.loc[set_to_ignore,"ignore_point"] = True

        return obs_column, obs_std_column

    def define_itc_observable(self,
                              obs_column,
                              obs_std):
        """
        Define an ITC observable for this experiment. 
        
        Parameters
        ----------
        obs_column : str
            name of column in initial dataframe that has the experimental
            observable. 
        obs_std : str or float, optional
            If str, use as name of column in initial dataframe that has the
            standard deviation on each observation.  If float, use this single
            value as the standard deviation on all observations.  
        """

        obs_column, obs_std_column = self._define_generic_observable(obs_column=obs_column,
                                                                       obs_std=obs_std)
        
        self._observables[obs_column] = {"type":"itc",
                                         "std_column":obs_std_column}


    def define_spectroscopic_observable(self,
                                        obs_column,
                                        obs_std,
                                        obs_microspecies,
                                        obs_macrospecies):
        """
        Define a spectroscopic observable for this experiment. 
        
        Parameters
        ----------
        obs_column : str
            name of column in initial dataframe that has the experimental
            observable. 
        obs_std : str or float, optional
            If str, use as name of column in initial dataframe that has the
            standard deviation on each observation.  If float, use this single
            value as the standard deviation on all observations.  
        obs_microspecies : list-like
            list of microscopic species from binding model that contribute to 
            this spectroscopic signal. 
        obs_macrospecies: str
            name of the macroscopic species from the binding model to use as the
            denominator for the observable. (This must be somewhere in the
            cell_contents or syringe_contents dictionaries as well.)

        Example
        -------
        Consider the binding reaction where the spectroscopic signal of "A" 
        changes when "B" binds:

        AT = A + AB
        BT = B + AB

        obs_microspecies would be ["AB"] and obs_macrospecies would be "AT".     
        """

        obs_column, obs_std_column = self._define_generic_observable(obs_column=obs_column,
                                                                     obs_std=obs_std)

        if issubclass(type(obs_microspecies),str):
            obs_microspecies = [obs_microspecies]

        if not hasattr(obs_microspecies,'__iter__'):
            err = "obs_microspecies should be a list of species contributing to signal\n"
            raise ValueError(err)

        expt_macrospecies = self._expt_concs.columns
        expt_macrospecies = [m for m in expt_macrospecies if m != "injection"]
        if obs_macrospecies not in expt_macrospecies:
            err = f"obs_macrospecies must be one of: {','.join(expt_macrospecies)}\n"
            raise ValueError(err)

        self._observables[obs_column] = {"type":"spec",
                                         "std_column":obs_std_column,
                                         "microspecies":obs_microspecies,
                                         "macrospecies":obs_macrospecies}
 
    def add_expt_conc_column(self,new_column,conc_vector=None):

        if conc_vector is None:
            conc_vector = np.zeros(len(self._expt_concs))

        if new_column not in self._expt_concs.columns:
            self._expt_concs[new_column] = conc_vector

    @property
    def expt_data(self):
        return self._expt_data
    
    @property
    def expt_concs(self):
        return self._expt_concs

    @property
    def observables(self):
        return self._observables

    @property
    def conc_to_float(self):
        return self._conc_to_float

    @property
    def syringe_contents(self):
        return self._syringe_contents
    
    @property
    def initial_cell_contents(self):
        return self._initial_cell_contents
    
    @property
    def titrating_macro_species(self):
        return self._titrating_macro_species