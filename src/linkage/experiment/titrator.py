import pandas as pd
import numpy as np

import copy

def sync_cell_and_syringe(cell_contents,
                          syringe_contents):
    """
    Make sure the same species are in both cell_contents and syringe_contents. 
    Any species present in one but not the other are assigned concentrations
    of 0.0.

    Parameters
    ----------
    cell_contents : dict
        dictionary with species as keys and cell concentrations as values 
    syringe_contents : dict
        dictionary with species as keys and syringe concentrations as values 
    
    Returns
    -------
    species : list
        sorted list of all species seen 
    titrating_species : list
        list of all species that titrate, meaning they are in the syringe and 
        have a different concentration from the concentration of that species
        in the cell
    cell_contents : dict
        cell_contents dict with concs of zero for missing species
    syringe_contents : dict
        syringe_contents dict with concs of zero for missing species
    """
    if not issubclass(type(cell_contents),dict):
        err = "cell_contents should be a dictionary with initial cell concs\n"
        raise ValueError(err)

    if not issubclass(type(syringe_contents),dict):
        err = "syringe_contents should be a dictionary with initial syringe concs\n"
        raise ValueError(err)

    titrating_species = []
    for s in syringe_contents:

        if s in cell_contents:
            if syringe_contents[s] == cell_contents[s]:
                continue
        titrating_species.append(s)

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

    return species, titrating_species, cell_contents, syringe_contents

def _titr_constant_volume(cell_contents,
                          syringe_contents,
                          injection_array,
                          cell_volume,
                          out):

    # For each injection
    for i in range(len(injection_array)):

        # Record current volume and injection
        out["injection"].append(injection_array[i])
        out["volume"].append(cell_volume)
        out["meas_vol_dilution"].append(1)

        # Update cell concs based on injected titrant
        for s in cell_contents.keys():

            prev_conc = cell_contents[s]
            
            a = (cell_volume - injection_array[i])*prev_conc
            b = injection_array[i]*syringe_contents[s]

            cell_contents[s] = (a + b)/cell_volume
            out[s].append(cell_contents[s])

    return out

def _titr_increase_volume(cell_contents,
                          syringe_contents,
                          injection_array,
                          cell_volume,
                          out):
    
    # For each injection
    current_volume = cell_volume
    meas_vol_dilution = 1
    for i in range(len(injection_array)):

        meas_vol_dilution = (1 - 2*injection_array[i]/cell_volume)

        # Get volume after this injection is injected
        new_volume = current_volume + injection_array[i]
            
        # Record current volume and injection
        out["injection"].append(injection_array[i])
        out["volume"].append(new_volume)
        out["meas_vol_dilution"].append(meas_vol_dilution)

        # Update cell concs based on injected titrant
        for s in cell_contents.keys():

            a = current_volume*cell_contents[s]
            b = injection_array[i]*syringe_contents[s]

            cell_contents[s] = (a + b)/new_volume
            out[s].append(cell_contents[s])

        # Update volume
        current_volume = new_volume

    return out
            


def titrator(cell_contents,
             syringe_contents,
             injection_array,
             cell_volume=1800,
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

    # Make sure all species are present in the cell and syringe dictionaries
    species, _, cell_contents, syringe_contents = sync_cell_and_syringe(cell_contents,
                                                                        syringe_contents)
    
    # Ensure injection_array is a numpy array of floats
    injection_array = np.array(injection_array,dtype=float)
    
    # Create output dictionary and populate with initial state
    out = {}
    out["injection"] = []
    out["volume"] = []
    out["meas_vol_dilution"] = []
    for s in species:
        out[s] = []
    
    if constant_volume:
        out = _titr_constant_volume(cell_contents,
                                    syringe_contents,
                                    injection_array,
                                    cell_volume,
                                    out)
    else:
        out = _titr_increase_volume(cell_contents,
                                    syringe_contents,
                                    injection_array,
                                    cell_volume,
                                    out)
            
    # Convert out to dataframe and return
    return pd.DataFrame(out)
