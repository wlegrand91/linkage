import pytest

from linkage.experiment.experiment import _load_dataframe
from linkage.experiment.experiment import _preprocess_df
from linkage.experiment.experiment import Experiment

import pandas as pd
import numpy as np

import os
import shutil

def test__load_dataframe(tmp_path):

    cwd = os.getcwd()
    os.chdir(tmp_path)

    out = {"test":[1,2,3]}
    df = pd.DataFrame(out)

    out_df = _load_dataframe(df)
    assert not out_df is df # return copy
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    df.to_csv("junk.csv")
    out_df = _load_dataframe("junk.csv")
    assert not out_df is df
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    df.to_csv("junk.tsv",sep="\t")
    out_df = _load_dataframe("junk.tsv")
    assert not out_df is df
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    df.to_csv("junk.txt")
    out_df = _load_dataframe("junk.txt")
    assert not out_df is df
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    df.to_excel("junk.xlsx")
    out_df = _load_dataframe("junk.xlsx")
    assert not out_df is df
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    shutil.copy("junk.xlsx","junk.xls")
    out_df = _load_dataframe("junk.xls")
    assert not out_df is df
    assert len(out_df) == 3
    assert np.allclose(out_df["test"],[1,2,3])

    with pytest.raises(FileNotFoundError):
        out_df = _load_dataframe("junk_not_real.xslx")

    with pytest.raises(ValueError):
        out_df = _load_dataframe(1.0)


    os.chdir(cwd)


def test__preprocess_df():

    df = pd.DataFrame({"injection":[1,2,3]})
    out_df = _preprocess_df(df)
    assert np.array_equal(out_df.columns,["injection","ignore_point"])

    df = pd.DataFrame({"out":[1,2,3]})
    with pytest.raises(ValueError):
        out_df = _preprocess_df(df)

    df = pd.DataFrame({"injection":[1,2],
                       "ignore_point":[False,False]})
    out_df = _preprocess_df(df)
    assert len(out_df) == 3
    assert np.array_equal(out_df.columns,["injection","ignore_point"])
    assert np.array_equal(out_df["injection"],[0,1,2])
    assert np.array_equal(out_df["ignore_point"],[True,False,False])    


    df = pd.DataFrame({"injection":[1,2],
                       "ignore_point":[False,False],
                       "obs":[1,1]})
    out_df = _preprocess_df(df)
    assert len(out_df) == 3
    assert np.array_equal(out_df.columns,["injection","ignore_point","obs"])
    assert np.array_equal(out_df["injection"],[0,1,2])
    assert np.array_equal(out_df["ignore_point"],[True,False,False])    
    assert np.array_equal(out_df["obs"],[np.nan,1,1],equal_nan=True)    


def test_Experiment():

    expt_data = pd.DataFrame({"injection":[0,1,1]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    assert issubclass(type(e._expt_data),pd.DataFrame)
    assert e._initial_cell_contents["A"] == 10
    assert e._initial_cell_contents["B"] == 0
    assert e._syringe_contents["A"] == 0
    assert e._syringe_contents["B"] == 10
    assert issubclass(type(e._expt_concs),pd.DataFrame)
    assert np.array_equal(e._expt_concs.columns,["injection","volume","meas_vol_dilution","A","B"])
    assert e._conc_to_float is None
    assert issubclass(type(e._observables),dict)
    assert len(e._observables) == 0

    # check properties
    assert e.expt_data is e._expt_data
    assert e.expt_concs is e._expt_concs
    assert e.observables is e._observables
    assert e.conc_to_float is e._conc_to_float
    assert e.syringe_contents is e._syringe_contents
    assert e.initial_cell_contents is e._initial_cell_contents

    assert np.array_equal(e._expt_concs["volume"],[100,101,102])

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float="A",
                   cell_volume=cell_volume,
                   constant_volume=True)
    
    assert e._conc_to_float == "A"
    assert np.array_equal(e._expt_concs["volume"],[100,100,100])

    with pytest.raises(ValueError):
        e = Experiment(expt_data=expt_data,
                    cell_contents=cell_contents,
                    syringe_contents=syringe_contents,
                    conc_to_float="NOT_THERE",
                    cell_volume=cell_volume,
                    constant_volume=True)
        
def test_Experiment_add_observable():

    expt_data = pd.DataFrame({"injection":[0,1,1],
                              "obs":[1,2,3]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    e.add_observable(column_name="obs",
                     obs_type="spec",
                     observable_species=["anything"],
                     denominator="A")
    assert e._observables["obs"]["obs_type"] == "spec"
    assert np.array_equal(e._observables["obs"]["observable_species"],["anything"])
    assert e._observables["obs"]["denominator"] == "A"
    
    # Bad observable column
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="not_a_column",
                         obs_type="spec",
                         observable_species=["anything"],
                         denominator="A")
        
    # Bad denominator
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="obs",
                         obs_type="spec",
                         observable_species=["anything"],
                         denominator="not_a_species")
                
    # No observable species
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="obs",
                         obs_type="spec",
                         observable_species=None,
                         denominator="A")
    
    # Bad observable species
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="obs",
                         obs_type="spec",
                         observable_species=1,
                         denominator="A")
    

    # Observable species as list
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    e.add_observable(column_name="obs",
                     obs_type="spec",
                     observable_species="anything",
                     denominator="A")
    assert e._observables["obs"]["obs_type"] == "spec"
    assert np.array_equal(e._observables["obs"]["observable_species"],["anything"])
    assert e._observables["obs"]["denominator"] == "A"


    # Disallowed column name
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="injection",
                         obs_type="spec",
                         observable_species=["anything"],
                         denominator="A")
        

    # Send in ITC experiment
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    e.add_observable(column_name="obs",
                     obs_type="itc",
                     observable_species=None,
                     denominator=None)
    assert e._observables["obs"]["obs_type"] == "itc"
    assert e._observables["obs"]["observable_species"] is None
    assert e._observables["obs"]["denominator"] is None

    # Observable not recognized
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    with pytest.raises(ValueError):
        e.add_observable(column_name="obs",
                         obs_type="not_really_osbervable",
                         observable_species=["anything"],
                         denominator="A")
        

def test_add_expt_column():

    expt_data = pd.DataFrame({"injection":[0,1,1]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    e.add_expt_conc_column(new_column="new_column")
    assert np.array_equal(e._expt_concs.columns,
                          ["injection","volume","meas_vol_dilution","A","B","new_column"])
    assert np.array_equal(e._expt_concs["new_column"],np.zeros(3))

    e.add_expt_conc_column(new_column="blah",conc_vector=np.array([3,2,1]))
    assert np.array_equal(e._expt_concs.columns,
                          ["injection","volume","meas_vol_dilution","A","B","new_column","blah"])
    assert np.array_equal(e._expt_concs["blah"],[3,2,1])

    