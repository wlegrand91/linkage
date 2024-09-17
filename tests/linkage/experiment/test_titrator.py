import pytest
import numpy as np
import pandas as pd

from linkage.experiment.titrator import titrator
from linkage.experiment.titrator import sync_cell_and_syringe
from linkage.experiment.titrator import _titr_constant_volume
from linkage.experiment.titrator import _titr_increase_volume

import copy

def test_titrator():
    
    base_kwargs = {"cell_contents":{"X":0,"Y":10},
                   "syringe_contents":{"X":10,"Y":10},
                   "cell_volume":100,
                   "injection_array":np.ones(10,dtype=float),
                   "constant_volume":False}
    
    kwargs = copy.deepcopy(base_kwargs)
    df = titrator(**kwargs)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 10
    assert np.allclose(np.array(df["volume"]),100 + np.arange(1,11))

    kwargs = copy.deepcopy(base_kwargs)
    kwargs["constant_volume"] = True
    df = titrator(**kwargs)
    assert issubclass(type(df),pd.DataFrame)
    assert len(df) == 10
    assert np.allclose(np.array(df["volume"]),100*np.ones(10))


def test_sync_cell_and_syringe():

    cell = {"A":10,"B":2}
    syringe = {"A":10,"C":20}

    species, titr, cell, syringe = sync_cell_and_syringe(cell_contents=cell,
                                                         syringe_contents=syringe)
    
    assert np.array_equal(species,["A","B","C"])
    assert np.array_equal(titr,["C"])

    assert cell["A"] == 10
    assert cell["B"] == 2
    assert cell["C"] == 0

    assert syringe["A"] == 10
    assert syringe["B"] == 0
    assert syringe["C"] == 20

    with pytest.raises(ValueError):
        species, titr, cell, syringe = sync_cell_and_syringe(cell_contents=1,
                                                             syringe_contents=syringe)
        
    with pytest.raises(ValueError):
        species, titr, cell, syringe = sync_cell_and_syringe(cell_contents=cell,
                                                             syringe_contents=-1)
        

def test__titr_constant_volume():

    cell = {"A":1,"B":2,"C":0}
    syringe = {"A":10,"B":0,"C":20}
    injection_array = np.ones(3,dtype=float)
    injection_array[0] = 0.0
    injection_array[2] = 10
    cell_volume = 100

    out = {}
    out["injection"] = []
    out["volume"] = []
    out["meas_vol_dilution"] = []
    out["A"] = []
    out["B"] = []
    out["C"] = []

    out = _titr_constant_volume(cell_contents=cell,
                                syringe_contents=syringe,
                                injection_array=injection_array,
                                cell_volume=cell_volume,
                                out=out)
    
    
    assert np.array_equal(out["injection"],injection_array)
    assert np.array_equal(out["volume"],[100,100,100])
    assert np.allclose(out["A"],
                       [1,
                        ((100 - 1)*1 + 10*1)/100,
                        ((100 - 10)*1.09 + 10*10)/100])
    assert np.allclose(out["B"],
                       [2,
                        ((100 - 1)*2)/100,
                        ((100 - 10)*1.98)/100])
    assert np.allclose(out["C"],
                       [0,
                        20/100,
                        ((100 - 10)*20/100 + 20*10)/100])
    

def test__titr_increase_volume():

    cell = {"A":1,"B":2,"C":0}
    syringe = {"A":10,"B":0,"C":20}
    injection_array = np.ones(3,dtype=float)
    injection_array[0] = 0.0
    injection_array[2] = 10
    cell_volume = 100

    out = {}
    out["injection"] = []
    out["volume"] = []
    out["meas_vol_dilution"] = []
    out["A"] = []
    out["B"] = []
    out["C"] = []

    out = _titr_increase_volume(cell_contents=cell,
                                syringe_contents=syringe,
                                injection_array=injection_array,
                                cell_volume=cell_volume,
                                out=out)
    
    assert np.array_equal(out["injection"],injection_array)
    assert np.array_equal(out["volume"],[100,101,111])
    

    assert np.allclose(out["A"],
                       [1,
                        (100*1 + 1*10)/101,
                        ((100*1 + 1*10)/101*101 + 10*10)/111])
                        
    assert np.allclose(out["B"],
                       [2,
                        (100*2)/101,
                        ((100*2)/101*101)/111])
    
    assert np.allclose(out["C"],
                       [0,
                        1*20/101,
                       (101*1*20/101 + 20*10)/111])


