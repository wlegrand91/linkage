
from linkage.models.base import BindingModel

import numpy as np

class CaEDTA(BindingModel):
    """
    equilibria:
        E + C -> EC; KE

    species:
        ET = E + EC
        CT = C + EC
    """
    
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

        # EC is the real root between 0->ET and 0->CT
        EC = self._get_real_root(roots=roots,
                                 upper_bounds=[ET,CT])
        
        # Get species
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
