
from linkage.models.base import BindingModel

import numpy as np

import warnings

class ReceptorCompetitor(BindingModel):
    """
    equilibria:
        ma -> m + a; K
        rma -> r + ma; KA
        rs -> r + s; KI

    species:
        mt = m + ma + rma
        st = s
        at = a + ma + rma
        rt = r + rma + rs
        
    """
    
    def get_concs(self,param_array,macro_array):

        K, KA, KI = param_array
    
        mt, st, at, rt = macro_array

        if rt > 0.01*np.min([mt,st,at]):
            w = "we assumed rt was small relative to other species in the derivation\n"
            warnings.warn(w)

        # For quadratic
        a = 1
        b = -(mt + at + K)
        c = mt*at
        sqrt_term = np.sqrt(b**2 - 4*a*c)
        denom = 2*a
        roots = np.array([(-b +sqrt_term)/denom, (-b - sqrt_term)/denom])

        # ma is the concentration of activator
        ma = self._get_real_root(roots=roots,
                                 upper_bounds=[mt,at])
        
        m = mt - ma
        s = st
        a = at - ma

        r = rt/(1 + ma/KA + st/KI)
        rma = r*ma/KA
        rs = r*st/KI

        return np.array([m,s,a,r,ma,rma,rs],dtype=float)

    @property
    def param_names(self):
        return np.array(["K","KA","KI"])

    @property
    def macro_species(self):
        return np.array(["mt","st","at","rt"])
    
    @property
    def micro_species(self):
        return np.array(["m", "s", "a", "r", "ma","rma","rs"])
