
import numpy as np

import re
import warnings

def _parse_rxn_side(rxn_side):

    molec = [m.strip() for m in rxn_side.split("+") if m.strip() != ""]
    
    if len(molec) == 0:
        err = f"no molecules found in '{rxn_side}'\n"
        raise ValueError(err)

    final_molec = []
    for m in molec:
        if len(m.split()) != 1:
            err = f"bad molecule '{m}' in '{rxn_side}'\n"
            raise ValueError(err)
        
        with_stoich = m.split("*")
        if len(with_stoich) == 1:
            final_molec.append(m)
        elif len(with_stoich) == 2:
            try:
                stoich = int(with_stoich[0])
                for _ in range(stoich):
                    final_molec.append(with_stoich[1])
            except:
                err = f"could not coerce '{with_stoich[0]}' in '{rxn_side}' into an integer\n"
                raise ValueError(err)
        else: 
            err = "could not parse stoichiometry of '{m}' in '{rxn_side}'\n"
            raise ValueError(err)

    final_molec.sort()

    return final_molec
                
def _parse_equilibria_field(line):

    first_split = line.split(";")
    if len(first_split) != 2:
        err = f"could not split '{line}' on ';'\n"
        raise ValueError(err)

    K = first_split[1].strip()
    if K == "":
        err = f"no equilibrium constant on '{line}'\n"
        raise ValueError(err)

    if K[0] != "K":
        err = f"equilibrium constant on '{line}' should start with 'K'\n"
        raise ValueError(err)
    
    if len(K.split()) != 1:
        err = f"equilibrium constant on '{line}' cannot have spaces\n"
        raise ValueError(err)

    second_split = first_split[0].split("->")
    if len(second_split) != 2:
        err = f"could not split '{line}' on '->'\n"
        raise ValueError(err)

    reactants = _parse_rxn_side(second_split[0])
    products = _parse_rxn_side(second_split[1])

    return K, reactants, products

def _parse_species_field(line):

    first_split = line.split("=")
    if len(first_split) != 2:
        raise ValueError(f"could not split '{line}' on '='\n")

    macro_species = _parse_rxn_side(first_split[0])
    if len(macro_species) != 1:
        err = f"could not parse macrospecies entry '{first_split[0]} on '{line}'\n"
        raise ValueError(err)
    macro_species = macro_species[0]

    micro_species = _parse_rxn_side(first_split[1])
    
    species, counts = np.unique(micro_species,return_counts=True)
    to_sort = list(zip(list(species),list(counts)))
    to_sort.sort()

    micro_species_names = [m[0] for m in to_sort]
    micro_species_stoich = [m[1] for m in to_sort]
        
    return macro_species, micro_species_names, micro_species_stoich

def _finalize_microspecies(equilibria,species):
    
    # Get all micro species seen in the equilibria
    micro_species_in_equilibria = []
    for K in equilibria:
        micro_species_in_equilibria.extend(equilibria[K][0])
        micro_species_in_equilibria.extend(equilibria[K][1])

    # get all micro species seen in the species definitions
    micro_species_in_species = []
    for s in species:
        micro_species_in_species.extend(species[s][0])

    # Convert species found into sets
    micro_species_in_equilibria = set(micro_species_in_equilibria)
    micro_species_in_species = set(micro_species_in_species)

    # Check for mismatch between the species in the equilibria and species
    # definitions
    if not micro_species_in_equilibria == micro_species_in_species:

        err = "mismatch in species between equilibria and species\n"
        
        only_in_eq = micro_species_in_equilibria - micro_species_in_species
        if len(only_in_eq) > 0:
            err += f"only in equilibria: {','.join(only_in_eq)}\n"

        only_in_sp = micro_species_in_species - micro_species_in_equilibria
        if len(only_in_sp) > 0:
            err += f"only in species: {','.join(only_in_sp)}\n"

        raise ValueError(err)
    
    micro_species = list(micro_species_in_equilibria)
    micro_species.sort()

    return micro_species

def _finalize_species(species,micro_species):

    macro_species = list(species.keys())
    macro_species.sort()

    # Check for species that appears as both a macrospecies and a microspecies
    micro_and_macro = set(macro_species).intersection(set(micro_species))
    if len(micro_and_macro): 
        err = "micro and macrospecies names are not unique\n"
        err += f"found in both: {','.join(micro_and_macro)}\n"
        raise ValueError(err)

    # Convert species into a dictionary that keys macro species name to 
    # a list like [(micro_species_1,micro_species_1_stoich),
    #              (micro_species_2,micro_species_2_stoich),...] 
    for s in species:
        species[s] = list(zip(*species[s]))

    return species, macro_species


def _parse_linkage_docstring(docstring):
    
    # Figure out how much leader is in front due to python indent
    # in class definition
    not_whitespace = re.compile("[^\\s]")
    front_space = None
    for line in docstring.splitlines():
        line = line.split("#")[0]
        if line.strip() in ["equilibria:","species:"]:
            front_space = not_whitespace.search(line).start()
            break
        
    if front_space is None:
        err = "Could not parse reaction description. No 'equilbria:' or \n"
        err += "'species:' line found.\n"
        raise ValueError(err)
    
    equilibria = {}
    species = {}
    
    this_field = None
    for line in docstring.splitlines():

        # Split on comment lines
        line = line.split("#")[0]

        # Blank line --> no field
        if line.strip() == "":
            this_field = None
            continue

        # If indent level is correct for a new entry
        if not_whitespace.match(line[front_space:]):
            
            # entry looks like 'blah:', not 'blah blah:' or 'blah'
            if len(line.split()) == 1 and line.strip()[-1] == ":":    
                this_field = line.strip()[:-1]
                continue

            # otherwise, no field
            else:
                this_field = None

        # If we are in an equilibria field
        if this_field == "equilibria":
            
            K, reactants, products = _parse_equilibria_field(line)
            if K in equilibria:
                err = f"equilibrium constant '{K}' appears more than once\n"
                raise ValueError(err)
                
            equilibria[K] = [reactants,products]
        
        # If we are in a species field
        elif this_field == "species":

            macro_species, micro_species_names, micro_species_stoich = _parse_species_field(line)
            if macro_species in species:
                err = f"macro species '{macro_species}' appears more than once\n"
                raise ValueError(err)
                
            species[macro_species] = [micro_species_names,micro_species_stoich]

        # Not doing anything in this case
        else:
            continue

    # Make sure we found at least one equilibria field
    if len(equilibria) == 0:
        err = "no 'equilibria:' entries found\n"
        raise ValueError(err)
    
    # Make sure we found at least one species field
    if len(species) == 0:
        err = "no 'species:' entries found\n"
        raise ValueError(err)

    # Get unique list of micro_species, checking for consistency/sanity
    micro_species = _finalize_microspecies(equilibria=equilibria,
                                           species=species)

    # Get unique list of macro_species, checking for consistency/sanity
    species, macro_species = _finalize_species(species=species,
                                               micro_species=micro_species)

    
    return equilibria, species, micro_species, macro_species
    

class BindingModel:
    """
    Binding model. Must be subclassed to be used.

    Subclass must define the get_concs method and three properties: param_names,
    macro_species, and micro_species. See the docstrings on those functions for
    details.  

    The docstring for the subclass must describe the thermodynamic model. It
    should have two sets of fields: equilibria and species. These are parsed
    like yaml, where the fields should have ':' at the end and sub-fields are
    indented. All other content in the docstring is ignored. Example:

    ```
    This is some non-yaml stuff from the docstring. 

    equilibria: 
        X + Y -> Z; K1      # note about reaction 1
        W + X -> Y + Z; K2  # note about reaction 2

    # comment about the reaction
    species:
        AT = X + Y
        BT = W + Z

    More non-yaml stuff.
    ```

    Rules for defining the equilibria
    ---------------------------------

    1. Equilibria lines contain reactants (separated by '+'), a reaction
       (specified by '->'), products (separated by '+'), and an equilibrium
       constant (separated by ';'). 
    2. Species lines contain the macro species (must be single entry), 
       the definition (separated by '='), and the component micro species
       (separated by '+'). 
    3. Equilibrium constants and macrospecies must be unique. 
    4. Equilibria can only contain microspecies, not macrospecies. 
    5. Equilibrium constants must start with 'K'. 
    6. Names of species and equilibrium constants cannot include spaces. 
    7. White space in entries is ignored.
    8. Anything after # is ignored. 
    """

    def __init__(self):
        """
        Initialize the class. 
        """
        
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(self.__doc__)

        param_from_class = set(self.param_names)
        param_from_docstring = set(equilibria.keys())
        if param_from_class != param_from_docstring:
            err = "param names from docstring and class definition must match\n"
            err += f"from docstring: {','.join(param_from_docstring)}\n"
            err += f"from class: {','.join(param_from_class)}\n"
            raise AssertionError(err)
        
        micro_from_class = set(self.micro_species)
        micro_from_docstring = set(micro_species)
        if micro_from_class != micro_from_docstring:
            err = "microspecies from docstring and class definition must match\n"
            err += f"from docstring: {','.join(micro_from_docstring)}\n"
            err += f"from class: {','.join(micro_from_class)}\n"
            raise AssertionError(err)

        macro_from_class = set(self.macro_species)
        macro_from_docstring = set(macro_species)
        if macro_from_class != macro_from_docstring:
            err = "macrospecies from docstring and class definition must match\n"
            err += f"from docstring: {','.join(macro_from_docstring)}\n"
            err += f"from class: {','.join(macro_from_class)}\n"
            raise AssertionError(err)

        self._equilibria = equilibria
        self._species = species

    def _get_real_root(self,roots,upper_bounds=[]):
        """
        Get the real root between 0 and upper_bounds. 

        Parameters
        ----------
        roots : numpy.ndarray
            numpy array with roots to check
        upper_bounds : list-like
            list of upper bounds against which to check root.
        """

        # Check for realness
        to_check = [np.isreal(roots)]

        # Check to see if root >= 0 (isclose catches values that ended up just 
        # slightly less than zero due to numerical imprecision)
        to_check.append(np.logical_or(roots > 0,np.isclose(roots,0)))

        # Check to see if root <= lowest upper bound. (isclose for numerical
        # imprecision)
        if len(upper_bounds) > 0:
            min_upper = np.min(upper_bounds)
            to_check.append(np.logical_or(roots < min_upper,
                                          np.isclose(roots,min_upper)))
        
        # Get all roots that meet all criteria
        mask = np.logical_and.reduce(to_check)
        solution = np.unique(roots[mask])
        
        # No root matches condition. Warn and return np.nan. 
        if len(solution) == 0:
            warnings.warn("no roots found\n")
            return np.nan 
        
        # Multiple roots match conditions. Warn and return np.nan
        if len(solution) > 1:
            
            # Check whether the all roots are numerically close and thus 
            # arise from float imprecision. If really have multiple roots, 
            # return np.nan for all concentrations
            close_mask = np.isclose(solution[0],solution)
            if np.sum(close_mask) != len(solution):
                warnings.warn("multiple roots found\n")
                return np.nan
        
        # Return real component
        return np.real(solution[0])
                
    
    def get_concs(self,param_array,macro_array):
        """
        Function must be defined by subclass. It must take only param_array and
        macro_array and return the concentrations of all microspecies given 
        the parameters and macro species concentrations. 
        
        1. param_array should be an array of floats holding parameters in the
           order defined by the param_names property.
        2. marco_array should be an array of floats holding the macrospecies 
           inputs in the order defined by the macro_species property. This can 
           either be an L x N array where 'L' is the number of conditions and
           'N' is the number of macrospecies or simply an 'N'-length array.
        3. The function should return an array of floats holding the calculated
           micro species in the order defined by the micro_species property. 
           This can either be an L x M array where 'L' is the number of 
           conditions and 'M' is the number of microspecies or simply an 
           'M'-length array.
        """

        err = "This method must be redefined by the subclass\n"
        raise NotImplementedError(err)

        return np.array([],dtype=float)

    @property
    def param_names(self):
        """
        Parameters defining the model in the order they are read from 
        `param_array` in the `get_concs` method. These parameters must also 
        match the equilibria defined in the docstring. 
        """
        
        err = "This property must be redefined by the subclass\n"
        raise NotImplementedError(err)

        return np.array(["K1","K2"])

    @property
    def macro_species(self):
        """
        Macrospecies within the model in the order they are read from 
        `macro_array` in the `get_concs` method. These macrospecies must also 
        match the species defined in the docstring. 
        """

        err = "This property must be redefined by the subclass\n"
        raise NotImplementedError(err)

        return np.array(["CT","ET"])
    
    @property
    def micro_species(self):
        """
        Microspecies from the model in the order they are returned by the 
        `get_concs` method. These microspecies must also match the species
        defined in the docstring.  
        """

        err = "This property must be redefined by the subclass\n"
        raise NotImplementedError(err)

        return np.array(["C", "E", "EC"])
    
    @property
    def equilibria(self):
        return self._equilibria
    
    @property
    def species(self):
        """
        Dictionary keying macro_species to list of tuples representing their
        microspecies and the stoichiometries of those microspecies.
        """
        return self._species