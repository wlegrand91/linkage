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
    assert np.array_equal(e._expt_concs.columns,["injection","volume","A","B"])
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
    assert np.array_equal(e.titrating_macro_species,["B"])
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
        
    # make sure we recognize that only B titrates even if we have "A" in the
    # cell and titrant. 
    expt_data = pd.DataFrame({"injection":[0,1,1]})
    cell_contents = {"A":10}
    syringe_contents = {"A":10,"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    assert e.expt_data is e._expt_data
    assert e.expt_concs is e._expt_concs
    assert e.observables is e._observables
    assert e.conc_to_float is e._conc_to_float
    assert e.syringe_contents is e._syringe_contents
    assert e.initial_cell_contents is e._initial_cell_contents
    assert np.array_equal(e.titrating_macro_species,["B"])
    assert np.array_equal(e._expt_concs["volume"],[100,101,102])


def test_Experiment__define_generic_observable():

    expt_data = pd.DataFrame({"injection":[0,1,1],
                              "obs":[1,2,3],
                              "obs_std":[0.1,0.2,0.3]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   cell_volume=cell_volume,
                   conc_to_float=conc_to_float,
                   constant_volume=constant_volume)
    
    obs_column, obs_std_column = e._define_generic_observable(obs_column="obs",
                                                                obs_std="obs_std")
    assert obs_column == "obs"
    assert obs_std_column == "obs_std"

    with pytest.raises(ValueError):
        e._define_generic_observable(obs_column="not_a_column",
                                     obs_std="obs_std")
        
    with pytest.raises(ValueError):
        e._define_generic_observable(obs_column="injection",
                                     obs_std="obs_std")
        
    with pytest.raises(ValueError):
        e._define_generic_observable(obs_column="obs",
                                     obs_std="not_a_column")

    assert np.array_equal(e._expt_data["obs_std"],[0.1,0.2,0.3])
    obs_column, obs_std_column = e._define_generic_observable(obs_column="obs",
                                                                obs_std=1.5)

    assert obs_column == "obs"
    assert obs_std_column == "obs_std"
    assert np.array_equal(e._expt_data["obs_std"],[1.5,1.5,1.5])

    # add twice, which should throw a warning
    e = Experiment(expt_data=expt_data,
                cell_contents=cell_contents,
                syringe_contents=syringe_contents,
                cell_volume=cell_volume,
                conc_to_float=conc_to_float,
                constant_volume=constant_volume)
    
    # add an itc observable (have to add completely because _define_generic 
    # does not actually update _observable dict)
    e.define_itc_observable(obs_column="obs",obs_std="obs_std")
    
    # should now warn because already added obs_column = "obs" to data
    with pytest.warns():
        obs_column, obs_std_column = e._define_generic_observable(obs_column="obs",
                                                                    obs_std="obs_std")
        
    # Send in a nan and make sure it is set to be ignored
    expt_data = pd.DataFrame({"injection":[0,1,1],
                              "obs":[1,np.nan,3],
                              "obs_std":[0.1,0.2,0.3]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   cell_volume=cell_volume,
                   conc_to_float=conc_to_float,
                   constant_volume=constant_volume)
    e._define_generic_observable(obs_column="obs",
                                 obs_std="obs_std")
    assert np.array_equal(e._expt_data["ignore_point"],
                          [False,True,False])


def test_Experiment_define_spectroscopic_observable():

    expt_data = pd.DataFrame({"injection":[0,1,1],
                              "obs":[1,2,3],
                              "obs_std":[0.1,0.2,0.3]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   cell_volume=cell_volume,
                   conc_to_float=conc_to_float,
                   constant_volume=constant_volume)
    
    e.define_spectroscopic_observable(obs_column="obs",
                                      obs_std="obs_std",
                                      obs_microspecies=["anything"],
                                      obs_macrospecies="A")
    assert e._observables["obs"]["type"] == "spec"
    assert np.array_equal(e._observables["obs"]["microspecies"],["anything"])
    assert e._observables["obs"]["macrospecies"] == "A"
    
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    # Bad observable column
    with pytest.raises(ValueError):
        e.define_spectroscopic_observable(obs_column="not_a_column",
                                        obs_std="obs_std",
                                        obs_microspecies=["anything"],
                                        obs_macrospecies="A")
        
    # Bad obs_std column
    with pytest.raises(ValueError):
        e.define_spectroscopic_observable(obs_column="obs",
                                        obs_std="not_a_column",
                                        obs_microspecies=["anything"],
                                        obs_macrospecies="A")
        
    # good obs_std, but as float
    e.define_spectroscopic_observable(obs_column="obs",
                                      obs_std=1.5,
                                      obs_microspecies=["anything"],
                                      obs_macrospecies="A")
    assert np.array_equal(e._expt_data["obs_std"],[1.5,1.5,1.5])


    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)

    # Bad obs_macrospecies column
    with pytest.raises(ValueError):
        e.define_spectroscopic_observable(obs_column="obs",
                                        obs_std="obs_std",
                                        obs_microspecies=["anything"],
                                        obs_macrospecies="not_in_cell_or_syringe")
                
    # obs_microspecies as a single value
    e.define_spectroscopic_observable(obs_column="obs",
                                      obs_std=1.5,
                                      obs_microspecies="anything",
                                      obs_macrospecies="A")
    assert np.array_equal(e._observables["obs"]["microspecies"],["anything"])

    # obs_microspecies as multiple values
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    e.define_spectroscopic_observable(obs_column="obs",
                                      obs_std=1.5,
                                      obs_microspecies=["anything","else"],
                                      obs_macrospecies="A")
    assert np.array_equal(e._observables["obs"]["microspecies"],["anything","else"])


    # bad obs_microspecies
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    with pytest.raises(ValueError):
        e.define_spectroscopic_observable(obs_column="obs",
                                          obs_std="obs_std",
                                          obs_microspecies=1.5,
                                          obs_macrospecies="A")


def test_Experiment_define_itc_observable():

    expt_data = pd.DataFrame({"injection":[0,1,1],
                              "obs":[1,2,3],
                              "obs_std":[0.1,0.2,0.3]})
    cell_contents = {"A":10}
    syringe_contents = {"B":10}
    conc_to_float = None
    cell_volume = 100
    constant_volume = False

    # This function is a light wrapper for _define_generic_observable. Don't 
    # check it's validation, but do make sure we assign values correctly. 
    e = Experiment(expt_data=expt_data,
                   cell_contents=cell_contents,
                   syringe_contents=syringe_contents,
                   conc_to_float=conc_to_float,
                   cell_volume=cell_volume,
                   constant_volume=constant_volume)
    
    e.define_itc_observable(obs_column="obs",
                            obs_std="obs_std")
    
    assert len(e._observables["obs"]) == 2
    assert e._observables["obs"]["type"] == "itc"
    assert e._observables["obs"]["std_column"] == "obs_std"
    

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
                          ["injection","volume","A","B","new_column"])
    assert np.array_equal(e._expt_concs["new_column"],np.zeros(3))

    e.add_expt_conc_column(new_column="blah",conc_vector=np.array([3,2,1]))
    assert np.array_equal(e._expt_concs.columns,
                          ["injection","volume","A","B","new_column","blah"])
    assert np.array_equal(e._expt_concs["blah"],[3,2,1])

    