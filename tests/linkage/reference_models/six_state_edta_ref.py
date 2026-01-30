"""
"""

from linkage.models.base import BindingModel
import numpy as np
from scipy.optimize import fsolve


class SixStateEDTATest(BindingModel):
    """
    equilibria:
        E + C -> EC; KE
        A -> I; KI
        A + C -> AC1; K1
        AC1 + C -> AC2; K2
        AC2 + C -> AC3; K3
        AC3 + C -> AC4; K4

    species:
        ET = E + EC
        AT = I + A + 2*AC1 + AC2 + 2*AC3 + AC4
        CT = C + EC + 2*AC1 + 2*AC2 + 6*AC3 + 4*AC4
    """

    def _get_free_c(self, KI, KE, K1, K2, K3, K4, AT, CT, ET):
        def equation(C):
            return (4*AT*C**4*K1*K2*K3*K4/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2 + 2*C*K1 + KI + 1)
                    + 6*AT*C**3*K1*K2*K3/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2 + 2*C*K1 + KI + 1)
                    + 2*AT*C**2*K1*K2/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2 + 2*C*K1 + KI + 1)
                    + 2*AT*C*K1/(C**4*K1*K2*K3*K4 + 2*C**3*K1*K2*K3 + C**2*K1*K2 + 2*C*K1 + KI + 1)
                    + C*ET*KE/(C*KE + 1) + C
                    - CT)

        try:
            # Initial guess
            C0 = CT / 2

            # Solve the equation
            result = fsolve(equation, C0, full_output=True)

            if result[2] != 1:  # Check if solution was found
            #     print("Failed to find solution")
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
        KI, KE, K1, K2, K3, K4 = np.exp(param_array)
        AT, CT, ET = macro_array
    
        C = self._get_free_c(KI, KE, K1, K2, K3, K4, AT, CT, ET)

        # Add this check
        if not np.isfinite(C):
            return np.full(9, 0)

        # Rest of concentration calculations 
        C1 = C
        C2 = C**2
        C3 = C**3
        C4 = C**4

        den = (
            1
            + KI
            + 2 * K1 * C1
            + K1 * K2 * C2
            + 2 * K1 * K2 * K3 * C3
            + K1 * K2 * K3 * K4 * C4
        )
        A = AT / den

        den = 1 + KE * C
        E = ET / den

        I = KI * A
        EC = KE * E * C
        AC1 = K1 * A * C
        AC2 = AC1 * K2 * C
        AC3 = AC2 * K3 * C
        AC4 = AC3 * K4 * C

        return np.array([I, A, C, E, AC1, AC2, AC3, AC4, EC])

    @property
    def param_names(self):
        return np.array(["KI", "KE", "K1", "K2", "K3", "K4"])

    @property
    def macro_species(self):
        return np.array(["AT", "CT", "ET"])

    @property
    def micro_species(self):
        return np.array(["I", "A", "C", "E", "AC1", "AC2", "AC3", "AC4", "EC"])