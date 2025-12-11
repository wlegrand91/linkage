import numpy as np
import pandas as pd
from sympy import Poly, lambdify, diff, Matrix
import warnings
from bindingpolytools import BindingPolynomial


class GenericBindingModel():


    def __init__(self, model_spec, debug=False):


        """
        Solves for species concentrations in a system of chemical equilibrium.

        This class uses the `BindingPolynomial` class from the `BindingPolyTools`
        library, which uses the SymPy library to symbolically derive the binding
        polynomial equation from a set of user-defined equilibrium and mass
        conservation equations. This derived polynomial is then used to perform
        fast numerical calculations, solving for the concentrations of all chemical
        species under specified conditions.

        The core methodology relies on algebraically reducing the entire system of
        equilibria and mass balance equations into a single polynomial for one
        unknown free concentration (referred to internally as `_c_symbol`). Once the
        root of this polynomial is found numerically, all other species'
        concentrations are determined by back-substitution.

        Analytical Jacobian Calculation:
        -------------------------------
        In addition to solving for concentrations, the class performs a one-time
        symbolic derivation of the Jacobian matrix. This matrix represents the
        sensitivity of each species' concentration to changes in the equilibrium
        constants (i.e., d[species]/d[K]).

        The derivation uses SymPy to apply the Implicit Function Theorem and the
        chain rule to the symbolic equations. The resulting analytical Jacobian is
        then compiled into a highly efficient numerical function. This provides a
        significant speed and accuracy advantage over traditional numerical
        differentiation (e.g., finite difference) methods. The numerical Jacobian
        can be retrieved for any set of conditions, making it ideal for use in
        sensitivity analysis or gradient-based optimization for parameter fitting.

        Model Specification and Limitations:
        -----------------------------------
        Formatting examples for the `model_spec` input can be found in the
        `linkage/src/linkage/model_specs` folder. 
        The specification can also be defined as a docstring
        in a script or Jupyter notebook for on-the-fly model changes.

        The `model_spec` string must define a system whose species dependency
        graph is acyclic. This technical constraint means that the network of
        reactions can be solved through sequential substitution, which is a
        requirement for the symbolic engine to derive the necessary polynomial.

        In less mathematical terms, the model's structure must not contain any
        circular dependencies.

        Examples of supported reaction topologies:
        - Sequential Binding: A linear chain of reactions, such as a protein
            binding multiple ligands in a stepwise fashion (e.g., P -> PL -> PL2).
        - Competitive Binding: A central hub species binding to multiple,
            non-interacting competitors (e.g., L1 <- P -> L2). This forms a valid
            star-shaped or tree-like structure.

        Examples of unsupported (cyclic) topologies that will fail:
        - Reaction Rings: A system where species A binds B, B binds C, and C
            in turn binds A. It is impossible to solve for [A] without first
            knowing [C], which requires knowing [B], which requires knowing [A],
            creating a circular dependency.
        - Coupled Systems: Any system that cannot be algebraically simplified
            and would require a numerical solver for a system of simultaneous
            non-linear equations.
        """


        if model_spec is None:
            raise ValueError("No model specification provided")

        self._model_spec = model_spec
        self._debug = debug

        poly_tool = BindingPolynomial(model_spec, debug=self._debug)

        self._equilibria = poly_tool._equilibria
        self._constants = poly_tool._constants
        self._micro_species = poly_tool._micro_species
        self._macro_species = poly_tool._macro_species

        self.symbols_dict = poly_tool.symbols
        self._c_symbol = poly_tool._c_symbol
        self._c_species_name = poly_tool._c_species_name
        self._ct_macrospecies_name = poly_tool._ct_macrospecies_name

        self.simplified_eqs = poly_tool.simplified_eqs
        self.solved_vars = poly_tool.solved_vars
        
        # Renamed for consistency with the GlobalModel API
        self.final_ct = poly_tool.binding_polynomial 
        
        self._setup_numerical_model()
        
        # Perform one-time symbolic derivation of the Jacobian
        self._setup_symbolic_jacobian()
        
        self._concentrations_df = pd.DataFrame(columns=self._micro_species, dtype=float)
        self._last_concs_dict = None # For Jacobian calculation
    
    def _log(self, message):
        if self._debug:
            print(message)

    def _setup_numerical_model(self):
        self._log("\nPreparing symbolic model for numerical evaluation")

        try:
            self.poly_obj = Poly(self.final_ct, self._c_symbol)
            self.symbolic_coeffs = self.poly_obj.all_coeffs()

            self._param_symbols_ordered = sorted(
                [s for s in self.final_ct.free_symbols if s != self._c_symbol], 
                key=lambda s: s.name
            )

            self._lambdified_coeffs_funcs = []
            for coeff_expr in self.symbolic_coeffs:
                args_for_lambdify = [s for s in self._param_symbols_ordered if s in coeff_expr.free_symbols]

                if not args_for_lambdify:
                    self._lambdified_coeffs_funcs.append(lambda **kwargs: float(coeff_expr))
                else:
                    self._lambdified_coeffs_funcs.append(lambdify(args_for_lambdify, coeff_expr, "numpy"))

            self._log(f"Successfully lambdified {len(self.symbolic_coeffs)} polynomial coefficients.")

        except Exception as e:
            raise RuntimeError(f"Failed to process the symbolic polynomial and lambdify its coefficients. Error: {e}")

    def _setup_symbolic_jacobian(self):
        """
        Derives a symbolic Jacobian d(micro_species)/d(constants) once using
        SymPy and the Implicit Function Theorem, then lambdifies it into a
        fast numerical function.
        """
        self._log("\nDeriving symbolic Jacobian")
        self.jacobian_function = None
        self._jacobian_input_symbols = None

        try:
            F = self.final_ct
            param_symbols = [self.symbols_dict[c] for c in self._constants]

            # 1. Calculate partial derivatives of the polynomial F(c, params)
            dF_dc = diff(F, self._c_symbol)
            dF_dparams = [diff(F, p) for p in param_symbols]

            # 2. Apply Implicit Function Theorem: dc/dp = -(dF/dp) / (dF/dc)
            dc_dparams = [-dF_dp / dF_dc for dF_dp in dF_dparams]

            # 3. Build the full Jacobian matrix using the chain rule
            jacobian_rows = []
            for species_name in self._micro_species:
                species_sym = self.symbols_dict[species_name]

                # Find the symbolic expression for the current species
                if species_sym == self._c_symbol:
                    species_expr = self._c_symbol
                elif species_sym in self.solved_vars:
                    species_expr = self.solved_vars[species_sym]
                elif species_sym in self.simplified_eqs:
                    species_expr = self.simplified_eqs[species_sym]
                else: 
                    raise ValueError(f"Cannot find symbolic expression for {species_name}")

                row = []
                for i, param_sym in enumerate(param_symbols):
                    # Total derivative: d(species)/d(param) = (∂S/∂c)*(dc/dp) + (∂S/∂p)_direct
                    chain_rule_part = diff(species_expr, self._c_symbol) * dc_dparams[i]
                    direct_part = diff(species_expr, param_sym)
                    total_deriv = chain_rule_part + direct_part
                    row.append(total_deriv)
                jacobian_rows.append(row)

            symbolic_jacobian = Matrix(jacobian_rows)
            
            # 4. Lambdify the symbolic matrix for fast numerical evaluation
            self._jacobian_input_symbols = sorted(list(symbolic_jacobian.free_symbols), key=lambda s: s.name)
            self.jacobian_function = lambdify(self._jacobian_input_symbols, symbolic_jacobian, "numpy")
            self._log(f"Successfully created lambdified Jacobian function.")

        except Exception as e:
            self.jacobian_function = None
            self._jacobian_input_symbols = None
            warnings.warn(f"Failed to derive symbolic Jacobian. Falling back to numerical methods. Error: {e}")

    def get_numerical_jacobian(self, concs_dict):
        """
        Calculates the numerical Jacobian d(micro_species)/d(constants) at a
        specific point in concentration space.

        Parameters
        ----------
        concs_dict : dict
            A dictionary of all current species concentrations and parameter values.
            Keys must be strings (e.g., "K1", "P", "L", "PL").

        Returns
        -------
        numpy.ndarray
            The numerical Jacobian matrix, or None if the symbolic function is not available.
        """
        if self.jacobian_function is None:
            return None
        
        try:
            # Prepare arguments for the lambdified function in the correct order
            args = [concs_dict[s.name] for s in self._jacobian_input_symbols]
            return self.jacobian_function(*args)
        except Exception as e:
            self._log(f"Failed to evaluate numerical Jacobian: {e}")
            return None

    def _get_free_c(self, **param_dict_num_values):
        CT_val = param_dict_num_values.get(self._ct_macrospecies_name)
        if CT_val == 0:
            return 0.0

        numerical_coeffs = []
        for i, lamb_func in enumerate(self._lambdified_coeffs_funcs):
            sym_coeff = self.symbolic_coeffs[i]
            arg_names = [s.name for s in self._param_symbols_ordered if s in sym_coeff.free_symbols]
            kwargs_for_func = {name: param_dict_num_values[name] for name in arg_names}
            numerical_coeffs.append(lamb_func(**kwargs_for_func))
        
        coeffs_for_polyroots = [float(c) for c in reversed(numerical_coeffs)]
        
        try:
            roots = np.polynomial.polynomial.polyroots(coeffs_for_polyroots)
            return self._get_real_root(roots, upper_bounds=[CT_val])
        except Exception as e:
            self._log(f"numpy.polyroots failed: {e}")
            return np.nan

    def get_concs(self, param_array, macro_array):
        param_dict = dict(zip(self._constants, np.exp(param_array)))
        param_dict.update(dict(zip(self._macro_species, macro_array)))
        
        C_free_val = self._get_free_c(**param_dict)
        if np.isnan(C_free_val):
            self._last_concs_dict = None
            return np.full(len(self._micro_species), np.nan)

        concs_dict = {self._c_species_name: C_free_val}
        
        subs_dict = {self.symbols_dict[name]: val for name, val in param_dict.items()}
        subs_dict[self._c_symbol] = C_free_val

        for base_var_sym, expr in self.solved_vars.items():
            val = float(expr.subs(subs_dict))
            concs_dict[base_var_sym.name] = val
            subs_dict[base_var_sym] = val

        for complex_sym, expr in self.simplified_eqs.items():
            val = float(expr.subs(subs_dict))
            concs_dict[complex_sym.name] = val
            
        # Store the current state for the Jacobian calculation
        self._last_concs_dict = {**param_dict, **concs_dict}

        # Record concentrations to internal dataframe
        df_row = pd.DataFrame([concs_dict], columns=self._micro_species)
        self._concentrations_df = pd.concat([self._concentrations_df, df_row], ignore_index=True)

        return np.array([concs_dict.get(name, 0.0) for name in self._micro_species])

    def _get_real_root(self, roots_complex, upper_bounds=[]):
        real_roots = np.real(roots_complex[np.isreal(roots_complex)])
        positive_roots = real_roots[real_roots >= -1e-14]
        positive_roots[positive_roots < 0] = 0

        if len(positive_roots) == 0: return np.nan
        
        valid_roots = positive_roots
        if upper_bounds:
            min_upper_bound = np.min(upper_bounds)
            valid_roots = valid_roots[valid_roots <= min_upper_bound * 1.001]

        if len(valid_roots) == 0: return np.nan

        return np.min(valid_roots)
    
    @property
    def equilibria(self):
        return self._equilibria
    
    @property
    def param_names(self):
        return np.array(self._constants)
    
    @property
    def macro_species(self):
        return np.array(self._macro_species)
    
    @property
    def micro_species(self):
        return np.array(self._micro_species)

    @property
    def concentrations_df(self):
        return self._concentrations_df