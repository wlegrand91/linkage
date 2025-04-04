from linkage.models.base import BindingModel
import numpy as np
import warnings


class CaEDTATest(BindingModel):
    """
    equilibria:
        E + C -> EC; KE

    species:
        ET = E + EC
        CT = C + EC
    """

    def get_concs(self, param_array, macro_array):
        """
        Get the concentrations of all species in solution given the model
        parameters and concentrations of macro species.

        Parameters
        ----------
        param_array : numpy.ndarray
            array of equilibrium constant (KE)
        macro_array : numpy.ndarray
            array of total concentrations (C_total, E_total)

        Returns
        -------
        concs : numpy.ndarray
            array of species concentrations (C, E, EC)
        """
        # Apply exponential transform with scaling
        exp_params = np.exp(param_array)
        k_scale = max(1.0, np.max(exp_params))  # Prevent downscaling
        print(f"Original KE: {exp_params[0]}, Scaled KE: {exp_params[0]/k_scale}")
        KE = exp_params[0] / k_scale
        
        CT, ET = macro_array
        
        # Scale concentrations with minimum scale
        conc_scale = max(1e-6, max(CT, ET))  # Set minimum scale
        if conc_scale > 0:
            CT, ET = CT/conc_scale, ET/conc_scale

        # Early return for boundary cases
        if CT == 0 or ET == 0:
            return np.array([CT, ET, 0.0], dtype=float)

        # Simple quadratic in EC
        a = 1
        b = -(CT + ET + 1 / KE)
        c = ET * CT

        try:
            s = np.sqrt(b**2 - 4 * a * c)
            if not np.isfinite(s):
                raise ValueError("Non-finite discriminant in quadratic solution")
            
            roots = np.array([(-b + s) / (2 * a), (-b - s) / (2 * a)])
            
            # EC is the real root between 0->ET and 0->CT
            EC = self._get_real_root(roots=roots, upper_bounds=[ET, CT])
            
            if not np.isfinite(EC):
                raise ValueError("Non-finite root found")

            # Get species
            C = CT - EC
            E = ET - EC
            
            # Check results are physical
            if not (0 <= EC <= min(ET, CT)):
                raise ValueError(f"EC={EC} outside valid range [0, min({ET}, {CT})]")
            if not (0 <= C <= CT):
                raise ValueError(f"C={C} outside valid range [0, {CT}]")
            if not (0 <= E <= ET):
                raise ValueError(f"E={E} outside valid range [0, {ET}]")

            # Rescale concentrations back
            concentrations = np.array([C, E, EC], dtype=float)
            return concentrations * conc_scale

        except Exception as e:
            # Provide diagnostic information
            w = f"\nQuadratic solution failed: {str(e)}\n"
            w += f"Parameters: KE={KE:.2e}\n"
            w += f"Concentrations: CT={CT:.2e}, ET={ET:.2e}\n"
            w += f"Coefficients: a={a:.2e}, b={b:.2e}, c={c:.2e}\n"
            warnings.warn(w)
            raise

    @property
    def param_names(self):
        return np.array(["KE"])

    @property
    def macro_species(self):
        return np.array(["CT", "ET"])

    @property
    def micro_species(self):
        return np.array(["C", "E", "EC"])