
import pytest

from linkage.models.base import _parse_rxn_side
from linkage.models.base import _parse_equilibria_field
from linkage.models.base import _parse_species_field
from linkage.models.base import _finalize_microspecies
from linkage.models.base import _finalize_species
from linkage.models.base import _parse_linkage_docstring


import numpy as np

def test__parse_rxn_side():

    output = _parse_rxn_side("A + B")
    assert np.array_equal(output,["A","B"])

    output = _parse_rxn_side("B + A")
    assert np.array_equal(output,["A","B"])

    output = _parse_rxn_side("B + A + B")
    assert np.array_equal(output,["A","B","B"])

    output = _parse_rxn_side("A + 2*B")
    assert np.array_equal(output,["A","B","B"])

    # some wacky stoich that is legal
    output = _parse_rxn_side("A + B + 2*A + A")
    assert np.array_equal(output,["A","A","A","A","B"])

    # no reaction specified
    with pytest.raises(ValueError):
        output = _parse_rxn_side("")

    # bad species definition (missing + or space in name)
    with pytest.raises(ValueError):
        output = _parse_rxn_side("A + B C")

    # Bad stoich (not integer)
    with pytest.raises(ValueError):
        output = _parse_rxn_side("A + 7.2*B")

    # Bad stoich (flipped)
    with pytest.raises(ValueError):
        output = _parse_rxn_side("A + B*7")
    
    with pytest.raises(ValueError):
        output = _parse_rxn_side("A + 7*7*B")

    
def test__parse_equilibria_field():
    
    # basic run
    K, react, prod = _parse_equilibria_field("A + B -> C + D; K1")
    assert K == "K1"
    assert np.array_equal(react,["A","B"])
    assert np.array_equal(prod,["C","D"])

    # check sorting of products and reactants
    K, react, prod = _parse_equilibria_field("B + A -> D + C; K4")
    assert K == "K4"
    assert np.array_equal(react,["A","B"])
    assert np.array_equal(prod,["C","D"])

    # no ;
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C: K4")

    # multiple no ;
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C; K4;")

    # no K
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C; ")

    # K with disallowed name
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C; AK")

    # K with spaces
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C; K 1")

    # no ->
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A + D + C; K1")

    # multiple ->
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> D + C -> E + F; K1")

    # bad reactant
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B C + A -> D + C; K1")

    # bad product
    with pytest.raises(ValueError):
        K, react, prod = _parse_equilibria_field("B + A -> 7*8*D + C; K1")

    # good, but relatively complicateds
    K, react, prod = _parse_equilibria_field("B + 3*A -> D + D + C; Ktest")
    assert K == "Ktest"
    assert np.array_equal(react,["A","A","A","B"])
    assert np.array_equal(prod,["C","D","D"])


def test__parse_species_field():
    
    # basic test
    macro, micro, stoich = _parse_species_field("AT = A1 + A2 + A3")
    assert macro == "AT"
    assert np.array_equal(micro,["A1","A2","A3"])
    assert np.array_equal(stoich,[1,1,1])

    # Send in some wacky stoichiometries
    macro, micro, stoich = _parse_species_field("AT = A1 + A2 + 3*D + A2")
    assert macro == "AT"
    assert np.array_equal(micro,["A1","A2","D"])
    assert np.array_equal(stoich,[1,2,3])

    # Make sure it is really grabbing names
    macro, micro, stoich = _parse_species_field("XT = A1 + A2 + 11*Q + A2")
    assert macro == "XT"
    assert np.array_equal(micro,["A1","A2","Q"])
    assert np.array_equal(stoich,[1,2,11])

    # no =
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("XT A1 + A2 ")

    # multiple =
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("XT = A1 = A2 ")

    # no macrospecies
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field(" = A1 + A2")

    # bad macrospecies
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("X1 X2 = A1 + A2")

    # too many macrospecies
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("X1 + X2 = A1 + A2")

    # bad micro species
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("XT = A1  A2")

    # bad micro species
    with pytest.raises(ValueError):
        macro, micro, stoich = _parse_species_field("XT = A1 + 11.2*A2")


def test__finalize_microspecies():
    
    equilibria = {"K1":[["A","B"],["C","D"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D"],[1,1]]}
    micro_species = _finalize_microspecies(equilibria=equilibria,
                                           species=species)
    assert np.array_equal(micro_species,["A","B","C","D"])

    # test sorting
    equilibria = {"K1":[["B","A"],["C","D"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["D","C"],[1,1]]}
    micro_species = _finalize_microspecies(equilibria=equilibria,
                                           species=species)
    assert np.array_equal(micro_species,["A","B","C","D"])

    # extra in reactants
    equilibria = {"K1":[["A","B"],["C","D","E"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D"],[1,1]]}
    with pytest.raises(ValueError):
        micro_species = _finalize_microspecies(equilibria=equilibria,
                                            species=species)
        
    # extra in products
    equilibria = {"K1":[["A","B","E"],["C","D"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D"],[1,1]]}
    with pytest.raises(ValueError):
        micro_species = _finalize_microspecies(equilibria=equilibria,
                                            species=species)
    
    # fix by adding to species
    equilibria = {"K1":[["A","B"],["C","D","E"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D","E"],[1,1]]}
    micro_species = _finalize_microspecies(equilibria=equilibria,
                                           species=species)
    assert np.array_equal(micro_species,["A","B","C","D","E"])

    # extra in species
    equilibria = {"K1":[["A","B"],["C","D"]]}
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D","E"],[1,1]]}
    with pytest.raises(ValueError):
        micro_species = _finalize_microspecies(equilibria=equilibria,
                                            species=species)
    

def test__finalize_species():
    
    species = {"AT":[["A","B"],[1,1]],
               "CT":[["C","D"],[1,1]]}
    micro_species = ["A","B","C","D"]

    out_species, macro_species = _finalize_species(species=species,
                                                   micro_species=micro_species)
    
    assert len(out_species) == 2
    assert np.array_equal(out_species["AT"],[("A",1),("B",1)])
    assert np.array_equal(out_species["CT"],[("C",1),("D",1)])
    assert np.array_equal(macro_species,["AT","CT"])

    species = {"AT":[["A","B"],[1,12]],
               "XT":[["C","Q"],[12,1]]}
    micro_species = ["A","B","C","D"]

    out_species, macro_species = _finalize_species(species=species,
                                                   micro_species=micro_species)
    
    assert len(out_species) == 2
    assert np.array_equal(out_species["AT"],[("A",1),("B",12)])
    assert np.array_equal(out_species["XT"],[("C",12),("Q",1)])
    assert np.array_equal(macro_species,["AT","XT"])


    species = {"AT":[["A","B","AT"],[1,1,1]],
               "CT":[["C","D"],[1,1]]}
    micro_species = ["AT","A","B","C","D"]

    # microspecies same as macrospecies    
    with pytest.raises(ValueError):
        out_species, macro_species = _finalize_species(species=species,
                                                        micro_species=micro_species)


def test__parse_linkage_docstring():

    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
        XT = A + 2*B + C
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    assert np.array_equal(list(equilibria.keys()),["K1"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert len(species.keys()) == 1
    assert list(species.keys())[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT"])

    # no correct fields
    test_str = \
    """
    equilibriax:
        A + 2*B -> C; K1
    
    speciesx:
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    # other way to be bad
    test_str = \
    """
    equilibria :
        A + 2*B -> C; K1
    
    species :
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    # another  way to be bad
    test_str = \
    """
    equilibria test:
        A + 2*B -> C; K1
    
    species test:
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    # should fail because no equilibria entry
    test_str = \
    """
    equilibria:
    #    A + 2*B -> C; K1
    
    species:
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    # should fail because no species entry
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
    #    XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

     # should fail because of messed up indentation
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
     species:
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

     # should fail because of messed up indentation
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
    XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

     # should be fine because 
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
         XT = A + 2*B + C
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)
    assert np.array_equal(list(equilibria.keys()),["K1"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert len(species.keys()) == 1
    assert list(species.keys())[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT"])

    # should be fine
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    # 
    species:
        XT = A + 2*B + C
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)
    assert np.array_equal(list(equilibria.keys()),["K1"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert len(species.keys()) == 1
    assert list(species.keys())[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT"])

      # should be fine, lots of valid comments
    test_str = \
    """
    # comment!
    equilibria:#comment
        A + 2*B -> C; K1 #comment here
    # 
    species:     #comment
        XT = A + 2*B + C#comment

    ######
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)
    assert np.array_equal(list(equilibria.keys()),["K1"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert len(species.keys()) == 1
    assert list(species.keys())[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT"])

    # should fail because of duplicated K1
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
        A + 2*B -> C; K1
    
    species:
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    # should work because reaction now has diff names
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
        A + 2*B -> C; K2
    
    species:
        XT = A + 2*B + C
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)
    keys = list(equilibria.keys())
    keys.sort()
    assert np.array_equal(keys,["K1","K2"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert np.array_equal(equilibria["K2"][0],["A","B","B"])
    assert np.array_equal(equilibria["K2"][1],["C"])
    assert len(species.keys()) == 1
    assert list(species.keys())[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT"])

    # should fail because of duplicated XT
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
        XT = A + 2*B + C
        XT = A + 2*B + C
    """

    with pytest.raises(ValueError):
        equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    ## should work because of YT
    test_str = \
    """
    equilibria:
        A + 2*B -> C; K1
    
    species:
        XT = A + 2*B + C
        YT = A + B + B + B + 2*C
    """

    equilibria, species, micro_species, macro_species = _parse_linkage_docstring(test_str)

    assert np.array_equal(list(equilibria.keys()),["K1"])
    assert np.array_equal(equilibria["K1"][0],["A","B","B"])
    assert np.array_equal(equilibria["K1"][1],["C"])
    assert len(species.keys()) == 2
    keys = list(species.keys())
    keys.sort()
    assert keys[0] == "XT"
    assert np.array_equal(species["XT"],[("A",1),
                                         ("B",2),
                                         ("C",1)])
    assert keys[1] == "YT"
    assert np.array_equal(species["YT"],[("A",1),
                                         ("B",3),
                                         ("C",2)])
    
    assert np.array_equal(micro_species,["A","B","C"])
    assert np.array_equal(macro_species,["XT","YT"])