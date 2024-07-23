from linkage.experiment.titrator import titrator

import numpy as np
import pandas as pd

class Experiment:

    def __init__(self,
                 expt_data,
                 cell_contents,
                 syringe_contents,
                 conc_to_float=None,
                 cell_volume=1800,
                 constant_volume=False):

        if not issubclass(type(expt_data),pd.DataFrame):
            expt_data = pd.read_excel(expt_data)

        self._expt_data = expt_data
        if "injection" not in self._expt_data.columns:
            err = "expt_data should be a dataframe or spreadsheet with a 'injection' column\n"
            raise ValueError(err)

        # If there is no "0" injections start -- like in ITC, for example -- add this with 
        # np.nan for all non-injection values
        if not np.isclose(self._expt_data.loc[self._expt_data.index[0],"injection"],0):
            new_row = dict([(c,[np.nan]) for c in self._expt_data.columns])
            new_row["injection"] = [0.0]
            new_row = pd.DataFrame(new_row)
            self._expt_data = pd.concat((new_row,self._expt_data),ignore_index=True)
            
        if not issubclass(type(cell_contents),dict):
            err = "cell_contents should be a dictionary with initial cell concs\n"
            raise ValueError(err)

        if not issubclass(type(syringe_contents),dict):
            err = "syringe_contents should be a dictionary with initial cell concs\n"
            raise ValueError(err)

        # Calculate the total concentrations over the experiment
        self._expt_concs = titrator(cell_contents=cell_contents,
                                    syringe_contents=syringe_contents,
                                    cell_volume=cell_volume,
                                    injection_array=np.array(self._expt_data["injection"]),
                                    constant_volume=constant_volume)
        
        self._macrospecies = np.array(self._expt_concs.columns[2:])
        self._conc_array = np.array(self._expt_concs.loc[:,self._macrospecies])

        if conc_to_float is not None:
            self._floating_conc = True
            
            self._mask = self._macrospecies == conc_to_float
            if np.sum(self._mask) != 1:
                err = "conc_to_float should be one of the macrospecies in the dataset\n"
                raise ValueError(err)

        else:
            self._mask = np.ones(len(self._macrospecies),dtype=bool)
            self._floating_conc = False


        self._observable_columns = {}
        
    @property
    def expt_data(self):
        return self._expt_data
    
    @property
    def expt_concs(self):
        return self._expt_concs
    
    @property
    def macrospecies(self):
        return self._macrospecies

    @property
    def conc_array(self):

        return self._conc_array
        
    def get_scaled_conc_array(self,conc_float_scalar=1.0):
        """
        Get the concentration array with the species indicated in conc_to_float 
        scaled by conc_float_scalar.
        """

        conc_array = self._conc_array.copy()
        conc_array[:,self._mask] = conc_array[:,self._mask]*conc_float_scalar

        return conc_array

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

            if denominator not in self.macrospecies:
                err = f"denominator must be one of: {','.join(self.macrospecies)}\n"
                raise ValueError(err)

        self._observable_columns[column_name] = {"obs_type":obs_type,
                                                 "observable_species":observable_species,
                                                 "denominator":denominator}
 
    @property
    def observable_columns(self):
        return self._observable_columns

    @property
    def floating_conc(self):
        return self._floating_conc
