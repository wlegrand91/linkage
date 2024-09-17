
from linkage.models.base import BindingModel

import numpy as np

class HeadToHead(BindingModel):
    """
    equilibria:
        ma -> m + a; K1
        sa -> s + a; K2

    species:
        mt = m + ma
        st = s + sa
        at = a + ma + sa
    """
    
    def get_concs(self,param_array,macro_array):

        K = param_array[0]
        mt, st, at = macro_array

        # if at == zero, ma == 0 --> root below won't work. mt -> m, st -> s,
        # 0 -> ma, 0 -> sa, 0 -> a
        if at == 0:
            return np.array([mt,0,st,0,0],dtype=float)
        
        mst = mt + st

        # no m or s around, so only species is at -> a
        if mst == 0:
            return np.array([0,0,0,0,at])
        
        # Fraction of the molecules that are m or s
        fx_m = mt/mst
        fx_s = st/mst

        # For quadratic
        a = 1
        b = -(mst + at + K)
        c = mst*at

        sqrt_term = np.sqrt(b**2 - 4*a*c)
        denom = 2*a
        roots = np.array([(-b +sqrt_term)/denom, (-b - sqrt_term)/denom])

        # msa is the concentration of a bound to either m or s
        msa = self._get_real_root(roots=roots,
                                  upper_bounds=[mst,at])
        
        ma = fx_m*msa
        sa = fx_s*msa
        m = mt - ma
        s = st - sa
        a = at - ma - sa

        return np.array([m,ma,s,sa,a],dtype=float)

    @property
    def param_names(self):
        return np.array(["K1","K2"])

    @property
    def macro_species(self):
        return np.array(["mt","st","at"])
    
    @property
    def micro_species(self):
        return np.array(["m", "ma", "s", "sa", "a"])
