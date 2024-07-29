from linkage.experiment.titrator import titrator
from linkage.experiment.titrator import sync_cell_and_syringe

import numpy as np
import pandas as pd

import copy

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
                 conc_to_float=None,
                 cell_volume=1800,
                 constant_volume=False):

        # Process and clean up input experimental data
        expt_data = _load_dataframe(expt_data=expt_data)
        self._expt_data = _preprocess_df(expt_data=expt_data)
                
        # Process and clean up cell and syringe data
        _, cell_contents, syringe_contents = sync_cell_and_syringe(cell_contents,
                                                                   syringe_contents)
        self._initial_cell_contents = copy.deepcopy(cell_contents)
        self._syringe_contents = copy.deepcopy(syringe_contents)

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
    
    def add_observable(self,
                       column_name,
                       obs_type,
                       observable_species=None,
                       denominator=None):

        if column_name not in self._expt_data:
            err = "column_name should be one of the columns in the experimental data\n"
            raise ValueError(err)

        if column_name == "injection":
            err = "column_name cannot be injection\n"
            raise ValueError(err)

        if obs_type not in ["spec","itc"]:
            err = "obs_type should be 'spec' or 'itc'\n"
            raise ValueError(err)

        if obs_type == "spec":
            
            if observable_species is None:
                err = "observable_species must be specified for a 'spec' experiment\n"
                raise ValueError(err)

            if issubclass(type(observable_species),str):
                observable_species = [observable_species]

            if not hasattr(observable_species,'__iter__'):
                err = "observable_species should be a list of species contributing to signal\n"
                raise ValueError(err)
            
            if denominator is None:
                err = "denominator must be specified for a 'spec' experiment.\n"
                raise ValueError(err)

            macrospecies = self._expt_concs.columns
            macrospecies = [m for m in macrospecies if m != "injection"]
            if denominator not in macrospecies:
                err = f"denominator must be one of: {','.join(macrospecies)}\n"
                raise ValueError(err)

        self._observables[column_name] = {"obs_type":obs_type,
                                          "observable_species":observable_species,
                                          "denominator":denominator}
 
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