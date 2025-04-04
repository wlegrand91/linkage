"""
"""

from linkage.models.base import BindingModel
import numpy as np
from scipy.optimize import fsolve


class EightCycleA4(BindingModel):
    
    '''
    equilibria:
        E + C -> EC; KE
        A + C -> AC1; K1
        AC1 + C -> AC2; K2
        AC2 + C -> AC3; K3
        AC3 + C -> AC4; K4
        I + C -> IC1; KI1
        IC1 + C -> IC2; KI2
        A -> I; KT1
        AC1 -> IC1; KT2
        AC2 -> IC2; KT3

    species:
        ET = E + EC
        AT = I + 2*IC1 + IC2 + A + 2*AC1 + AC2 + 2*AC3 + AC4
        CT = C + EC + 2*AC1 + 2*AC2 + 6*AC3 + 4*AC4 + 2*IC1 + 2*IC2
    ''' 

    def _get_free_c(self, KE, K1, K2, K3, K4, KI1, KI2, KT1, KT2, KT3, AT, CT, ET):

        def equation(C):

            return (4*AT*C**4*K1*K2*K3*K4/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1) 
                    + 6*AT*C**3*K1*K2*K3/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
                    + 2*AT*C**2*K1*K2*KT3/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
                    + 2*AT*C**2*K1*K2/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
                    + 2*AT*C*K1*KT2/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
                    + 2*AT*C*K1/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
                    + C*ET*KE/(C*KE + 1) + C)

        try:
            # Initial guess
            C0 = CT / 2

            # Solve the equation
            result = fsolve(equation, C0, full_output=True)

            if result[2] != 1:  # Check if solution was found
                print("Failed to find solution")
                return np.nan

            root = result[0][0]  # First element of solution array

            # Check if root is physical (between 0 and CT)
            if not (0 <= root <= CT) or not np.isfinite(root):
                return np.nan

            return root

        except Exception as e:
            print(f"Error in root finding: {e}")
            return np.nan

    def get_concs(self, param_array, macro_array):
        """
        Get the concentrations of all species in solution given the model parameters
        and concentrations of macro species.

        Parameters
        ----------
        param_array : numpy.ndarray
            array of five equilibrium constants (KI, KE, K1, K2, K3, K4)
            Note: Values are in log space but named to match equilibria notation
        macro_array : numpy.ndarray
            array of total concentrations (A_total, C_total, E_total)

        Returns
        -------
        concs : numpy.ndarray
            array of species concentrations (A_free, C_free, E_free, AC1, AC2,
            AC3, AC4, EC).
        """
        # Check parameters for valid values
        if np.any(np.isnan(param_array)):
            return np.full(11, 0)

        if not np.all(np.isfinite(param_array)):
            return np.full(11, 0)

        if np.any(param_array == 0):
            return np.full(11, 0)

        KE, K1, K2, K3, K4, KI1, KI2, KT1, KT2, KT3 = np.exp(param_array)

        AT, CT, ET = macro_array

        C = self._get_free_c(
            KE, K1, K2, K3, K4, KI1, KI2, KT1, KT2, KT3, AT, CT, ET
        )

        # Is C a NaN
        if not np.isfinite(C):
            return np.full(11, 0)   

        # Microspecies equations
        E = ET/(C*KE + 1)
        A = AT/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2*KT3 + C**2*K1*K2 + 2*C*K1*KT2 + 2*C*K1 + KT1 + 1)
        AC1 = A*C*K1
        AC2 = AC1*C*K2
        AC3 = AC2*C*K3
        AC4 = AC3*C*K4
        EC = C*E*KE
        I = KT1*A
        IC1 = KT2*AC1
        IC2 = KT3*AC2

        return np.array([A, C, E, AC1, AC2, AC3, AC4, EC, I, IC1, IC2])

    @property
    def param_names(self):
        return np.array(
            ["KE", "K1", "K2", "K3", "K4", "KI1", "KI2", "KT1", "KT2", "KT3"]
        )

    @property
    def macro_species(self):
        return np.array(["AT", "CT", "ET"])

    @property
    def micro_species(self):
        return np.array(
            ["A", "C", "E", "AC1", "AC2", "AC3", "AC4", "EC", "I", "IC1", "IC2"]
        )
