
from linkage.models.base import BindingModel

import numpy as np

import warnings

class CaEDTA(BindingModel):
    """
    species:
        ET = E + EC
        CT = C + EC
    
    equilibria:
        E + C -> EC; KE
    """
    
    def __init__(self):
        super().__init__()

    def get_concs(self,param_array,macro_array):

        KE = param_array[0]
        CT, ET = macro_array

        if CT == 0 or ET == 0:
            return np.array([CT,ET,0.0],dtype=float)

        # Simple quadratic
        # a = 1
        b = -(CT + ET + 1/KE)
        c = ET*CT
        s = np.sqrt(b**2 - 4*c)
        roots = np.array([(-b + s)/2, (-b - s)/2])

        # Get real root that has concentration <= CT and <= ET
        mask = np.logical_and.reduce((np.isreal(roots),
                                      roots>=0,
                                      roots<=CT,
                                      roots<=ET))
        solution = np.unique(roots[mask])
        
        # No real root between 0 and CT. Return np.nan for all concentrations
        if len(solution) == 0:
            warnings.warn("no roots found\n")
            return np.nan*np.ones(3,dtype=float)
        
        # Multiple real roots between 0 and CT, 0 and ET. 
        if len(solution) > 1:
            
            # Check whether the all roots are numerically close and thus 
            # arise from float imprecision. If really have multiple roots, 
            # return np.nan for all concentrations
            close_mask = np.isclose(solution[0],solution)
            if np.sum(close_mask) != len(solution):
                warnings.warn("multiple roots found\n")
                return np.nan*np.ones(3,dtype=float)

        # Get species
        EC = np.real(solution[0])
        C = CT - EC
        E = ET - EC

        return np.array([C,E,EC],dtype=float)

    @property
    def param_names(self):
        return np.array(["KE"])

    @property
    def macro_species(self):
        return np.array(["CT","ET"])
    
    @property
    def micro_species(self):
        return np.array(["C", "E", "EC"])
    
    @property
    def reactants(self):
        pass