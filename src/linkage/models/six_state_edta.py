"""
"""

from linkage.models.base import BindingModel

import numpy as np


class SixStateEDTA(BindingModel):
    """
    equilibria:
        E + C -> EC; KE
        A -> I; KI
        A + C -> AC1; K1
        A + 2*C -> AC2; K2
        A + 3*C -> AC3; K3
        A + 4*C -> AC4; K4

    species:
        ET = E + EC
        AT = I + A + 2*AC1 + AC2 + 2*AC3 + AC4
        CT = C + EC + 2*AC1 + 2*AC2 + 6*AC3 + 4*AC4
    """

    def _get_free_c(self,KI,KE,K1,K2,K3,K4,AT,CT,ET):
        """
        The `_get_free_c` function finds the polynomial root to get the free
        calcium concentration. The `get_concs` function calculates all species
        concentrations given $K_{E}$, $K_{1}$, $K_{2}$, $K_{3}$, and $K_{4}$,
        as well as the total concentrations of S100A4 dimer, calcium, and EDTA.
    
        Get the free calcium concentration given the equilibrium constants and 
        total concentrations in the system. Private function. Should generally be 
        called by get_concs.
        """
    
        # Coefficients based on equilibrium constants and total concs
        alpha = -CT*(1 + KI)
        beta = (1 + KI) + KE*(ET - CT)*(1 + KI) + K1*(2*AT - 2*CT)
        gamma = KE*(1 + K1*(2*ET + 2*AT - 2*CT)) + K1*(2 + K2*(2*AT - CT))
        delta = KE*K1*(2 + K2*(ET + 2*AT - CT)) + K1*K2*(1 + K3*(6*AT - 2*CT))
        epsilon = KE*K1*K2*(1 + K3*(2*ET + 6*AT -2*CT)) + K1*K2*K3*(2 + K4*(4*AT - CT))
        zeta = KE*K1*K2*K3*(2 + K4*(ET + 4*AT - CT)) + K1*K2*K3*K4
        eta = KE*K1*K2*K3*K4
    
        # Get roots of polynomial with these coefficients
        coef = np.array([alpha,beta,gamma,delta,epsilon,zeta,eta])
        P = np.polynomial.Polynomial(coef=coef)
        roots = P.roots()

        # Get real root between 0 and CT    
        root = self._get_real_root(roots=roots,
                                   upper_bounds=[CT])

        return root

        
    def get_concs(self,param_array,macro_array):
        """
        Get the concentrations of all species in solution given the model 
        parameters and concentrations of macro species. 
        
        Parameters
        ----------
        param_array : numpy.ndarray
            array of five equilibrium constants (KI, KE, K1, K2, K3, K4)
        macro_array : nump.ndarray
            array of total concentrations (A_total, C_total, E_total)
        
        Returns
        -------
        concs : numpy.ndarray
            array of species concentrations (A_free, C_free, E_free, AC1, AC2, 
            AC3, AC4, EC).
        """

        KI, KE, K1, K2, K3, K4 = param_array
        AT, CT, ET = macro_array
        
        # Get the free calcium concentration
        C = self._get_free_c(KI,KE,K1,K2,K3,K4,AT,CT,ET)
        
        # Calcium polynomial
        C1 = C
        C2 = (C**2)
        C3 = (C**3)
        C4 = (C**4)
        
        # Get the free A
        den = 1 + KI + 2*K1*C1 + K1*K2*C2 + 2*K1*K2*K3*C3 + K1*K2*K3*K4*C4
        A = AT/den
        
        # Get the free E
        den = 1 + KE*C
        E = ET/den
        
        # Use free A, free C, and free E with the equilibrium constants to
        # get other species concentrations
        I = KI*A
        EC = KE*E*C
        AC1 = K1*A*C
        AC2 = AC1*K2*C
        AC3 = AC2*K3*C
        AC4 = AC3*K4*C
        
        return np.array([I, A, C, E, AC1, AC2, AC3, AC4, EC])
    


    @property
    def param_names(self):
        return np.array(["KI","KE","K1","K2","K3","K4"])

    @property
    def macro_species(self):
        return np.array(["AT","CT","ET"])
    
    @property
    def micro_species(self):
        return np.array(["I", "A", "C", "E", "AC1", "AC2", "AC3", "AC4", "EC"])

