import pandas as pd
import numpy as np

import copy

def titrator(cell_contents,
             syringe_contents,
             cell_volume=1800,
             injection_array=None,
             constant_volume=False):
    """
    Simulate a titrator injecting constant volumes into a cell. Calculate the
    total concentrations of all species at each point. 
    
    Parameters
    ----------
    cell_contents : dict
        dictionary holding species in cell where keys are species names
        and values are concentrations. If a species is not seen in this
        dictionary but is seen in the syringe dictionary, it is assumed to have
        a concentration of 0.0. 
    syringe_contents : dict
        dictionary holding initial species in syringe where keys are species
        names and values are concentrations. If a species is not seen in this
        dictionary but is seen in the cell dictionary, it is assumed to have
        a concentration of 0.0. 
    cell_volume : float, default=1800
        volume of cell at start
    injection_array : array
        injections as an array. if a point is collected before a injection is injected,
        the first entry in this array should be 0. 
    constant_volume : bool, default=False
        if true, assume that the titrator pulls out each injection_size before injecting, 
        keeping the total cell volume the same over the titration. 
    
    Returns
    -------
    out : pandas.DataFrame
        pandas dataframe with the concentrations of all species in the cell
        versus injections in the experiment
    """
    
    # List of all species in syringe and cell
    species = list(set(syringe_contents.keys()).union(set(cell_contents.keys())))
    species.sort()
    
    # Copy syringe and cell dictionaries because we are about to modify
    # them.
    syringe_contents = copy.deepcopy(syringe_contents)
    cell_contents = copy.deepcopy(cell_contents)
    
    # If a species is only in the syringe or cell, set it to zero in the
    # other pool
    for s in species:
        if s not in syringe_contents:
            syringe_contents[s] = 0.0
        if s not in cell_contents:
            cell_contents[s] = 0.0
    
    # Ensure injection_array is a numpy array of floats
    injection_array = np.array(injection_array,dtype=float)
    
    # Create output dictionary and populate with initial state
    out = {}
    out["injection"] = []
    out["volume"] = []
    for s in species:
        out[s] = []
        
    # For each injection
    for i in range(len(injection_array)):

        # Get starting volume
        if len(out["injection"]) == 0:
            start_vol = cell_volume
        else:
            start_vol = out["volume"][-1]

        # Get volume after this injection is injected
        if constant_volume:
            new_vol = start_vol
        else:
            new_vol = start_vol + injection_array[i]

        # Record current volume and injection
        out["injection"].append(injection_array[i])
        out["volume"].append(new_vol)

        # Update cell concs based on injected titrant
        for s in species:

            prev_conc = cell_contents[s]
            
            if constant_volume:
                a = (cell_volume - injection_array[i])*prev_conc
            else:
                a = start_vol*prev_conc

            b = injection_array[i]*syringe_contents[s]
            cell_contents[s] = (a + b)/new_vol
            out[s].append(cell_contents[s])
            
    # Convert out to dataframe and return
    return pd.DataFrame(out)
