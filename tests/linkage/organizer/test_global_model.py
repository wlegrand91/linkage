

import pytest

import linkage
from linkage.organizer.global_model import GlobalModel
from linkage.organizer.point.spec_point import SpecPoint
from linkage.organizer.point.itc_point import ITCPoint

from linkage.experiment.experiment import Experiment

import numpy as np
import pandas as pd
import copy

def test_GlobalModel_integrated(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    guesses = simulated_itc["guesses"]

    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    gf.model(guesses)


def test_GlobalModel__load_model(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    assert gf._model_name == "SixStateEDTA"
    assert issubclass(type(gf._bm),linkage.models.six_state_edta.SixStateEDTA)

    this_expt_list = copy.deepcopy(base_expt_list)
    with pytest.raises(ValueError):
        gf = GlobalModel(model_name="not_a_model",
                         expt_list=this_expt_list) 

    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    assert gf._model_name == "SixStateEDTA"
    assert issubclass(type(gf._bm),
                      linkage.models.six_state_edta.SixStateEDTA)
    
    param_names = linkage.models.six_state_edta.SixStateEDTA().param_names
    assert np.array_equal(param_names,
                          gf._parameter_names[:len(param_names)])
    assert np.array_equal(np.zeros(len(param_names)),
                          gf._parameter_guesses[:len(param_names)])

    assert gf._bm_param_start_idx == 0
    assert gf._bm_param_end_idx == len(param_names) - 1
    
def test_GlobalModel__get_expt_std_scalar(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # 100 points in first experiment; 150 points in ext experiment. 
    assert np.allclose(gf._expt_std_scalar,[1.0,4/3.])

def test_GlobalModel__get_expt_normalization(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # Make sure we're properly getting mean & std for each experimental output
    for expt in this_expt_list:
        for k in expt.observables:
            values = np.array(expt.expt_data[k])
            m = np.nanmean(values)
            s = np.nanstd(values)

            assert np.allclose(gf._normalization_params[k],[m,s])
            
def test_GlobalModel__load_observables(fake_spec_and_itc_data):
    
    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)

    assert issubclass(type(gf._y_obs),np.ndarray)
    assert issubclass(type(gf._y_std),np.ndarray)
    assert issubclass(type(gf._y_norm_mean),np.ndarray)
    assert issubclass(type(gf._y_norm_std),np.ndarray)
    assert issubclass(type(gf._y_std_scalar),np.ndarray)
    assert issubclass(type(gf._y_obs_normalized),np.ndarray)
    assert issubclass(type(gf._y_std_normalized),np.ndarray)

    # Get the mean and standard deviation across each observable type across 
    # all experiments
    obs_values = {}
    for expt in this_expt_list:
        for k in expt.observables:
            
            values = np.array(expt._expt_data[k])
            if k not in obs_values:
                obs_values[k] = []
            obs_values[k].extend(list(values))
        
    obs_means = {}
    obs_stds = {}
    for k in obs_values:
        obs_means[k] = np.nanmean(obs_values[k])
        obs_stds[k] = np.nanstd(obs_values[k])
                      
    y_obs = []
    y_std = []
    y_norm_mean = []
    y_norm_std = []
    y_std_scalar = []

    expt_scalars = [1.0,4/3.]
    for counter, expt in enumerate(this_expt_list):
        for obs in expt.observables:

            this_obs = np.array(expt._expt_data[obs])
            this_std = np.array(expt._expt_data[f"{obs}_std"])

            mask = np.logical_not(np.isnan(this_obs))

            m = obs_means[obs]
            s = obs_stds[obs]

            y_obs.extend(list(this_obs[mask]))
            y_std.extend(list(this_std[mask]))
            y_norm_mean.extend(np.ones(np.sum(mask))*m)
            y_norm_std.extend(np.ones(np.sum(mask))*s)
            y_std_scalar.extend(np.ones(np.sum(mask))*expt_scalars[counter])

        
    y_obs = np.array(y_obs)
    y_std = np.array(y_std)
    y_norm_mean = np.array(y_norm_mean)
    y_norm_std = np.array(y_norm_std)
    y_std_scalar = np.array(y_std_scalar)

    y_obs_norm = (y_obs - y_norm_mean)/y_norm_std
    y_std_norm = y_std_scalar * y_std/y_norm_std

    assert np.allclose(gf._y_obs,y_obs)
    assert np.allclose(gf._y_std,y_std)
    assert np.allclose(gf._y_norm_mean,y_norm_mean)
    assert np.allclose(gf._y_norm_std,y_norm_std)
    assert np.allclose(gf._y_std_scalar,y_std_scalar)
    assert np.allclose(gf._y_obs_normalized,y_obs_norm)
    assert np.allclose(gf._y_std_normalized,y_std_norm)

    # This function also syncs up the macrospecies between experiments. Make
    # sure this is happening
    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)


    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)

    # Make sure array starts without AT
    assert np.array_equal(this_expt_list[1].expt_concs.columns,
                          ["injection","volume","CT","ET"])
    
    # make sure we added AT
    assert np.array_equal(gf._expt_list[1].expt_concs.columns,
                          ["injection","volume","CT","ET","AT"])
    
    num_points = len(gf._expt_list[1].expt_concs["AT"])
    assert np.array_equal(gf._expt_list[1].expt_concs["AT"],
                          np.zeros(num_points,dtype=float))


def test_GlobalModel__get_enthalpy_param(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert hasattr(gf,"_dh_param_start_idx")
    assert hasattr(gf,"_dh_param_end_idx")
    assert hasattr(gf,"_dh_sign")
    assert hasattr(gf,"_dh_product_mask")
    assert hasattr(gf,"_dh_dilution_mask")

    expected = ['dH_I','dH_E','dH_1','dH_2','dH_3','dH_4',
                "nuisance_dil_ET"]
    dh_param = gf._parameter_names[gf._dh_param_start_idx:gf._dh_param_end_idx + 1]

    assert np.array_equal(expected,dh_param)
    
    # "EC","I","AC1","AC2","AC3","AC4"
    order_in_class = np.array([8,0,4,5,6,7])

    # make sure it is correctly mapping reactions
    assert gf._dh_param_start_idx == 6
    assert gf._dh_param_end_idx == 12
    assert np.array_equal(gf._dh_sign,np.ones(6,dtype=float))
    for i in range(6):
        assert np.sum(gf._dh_product_mask[i]) == 1
        assert np.arange(9,dtype=int)[gf._dh_product_mask[i]] == order_in_class[i]

    assert np.array_equal(gf._dh_sign,[1,1,1,1,1,1])

    # Make sure it is figuring out the dilution correctly
    assert np.array_equal(gf._dh_dilution_mask,[False,False,True])

    # Remove itc experiment; should have no enthalpies
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list.pop(-1)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert not hasattr(gf,"_dh_param_start_idx")
    assert not hasattr(gf,"_dh_param_end_idx")
    assert not hasattr(gf,"_dh_sign")
    assert not hasattr(gf,"_dh_product_mask")
    assert not hasattr(gf,"_dh_dilution_mask")

def test_GlobalModel__get_expt_fudge(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf.parameter_names[-1] == "nuisance_expt_0_AT_fudge"
    assert gf.parameter_guesses[-1] == 1.0
    assert gf._fudge_list[0][0] == 0
    assert gf._fudge_list[0][1] == len(gf.parameter_names) - 1
    assert gf._fudge_list[1] is None

def test_GlobalModel__add_point():

    # This is a difficult method to test on its own because of how it integrates
    # with other attributes in the class. It's mostly tested by build_point_map...
    pass


def test_GlobalModel__build_point_map(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)  

    this_expt_list = copy.deepcopy(base_expt_list)  
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # Test micro_array
    assert len(gf._micro_arrays) == 2
    assert gf._micro_arrays[0].shape[1] == len(gf._bm.micro_species)
    assert gf._micro_arrays[1].shape[1] == len(gf._bm.micro_species)

    # Test ref_macro_array
    assert len(gf._ref_macro_arrays) == 2
    assert gf._ref_macro_arrays[0].shape[1] == len(gf._bm.macro_species)
    assert gf._ref_macro_arrays[1].shape[1] == len(gf._bm.macro_species)

    # Test macro_array
    assert len(gf._macro_arrays) == 2
    assert gf._macro_arrays[0].shape[1] == len(gf._bm.macro_species)
    assert gf._macro_arrays[1].shape[1] == len(gf._bm.macro_species)

    # Test del_macro_array
    assert len(gf._del_macro_arrays) == 2
    assert gf._del_macro_arrays[0].shape[1] == len(gf._bm.macro_species)
    assert gf._del_macro_arrays[1].shape[1] == len(gf._bm.macro_species)

    # Test expt_syring_concs
    assert len(gf._expt_syringe_concs) == 2
    assert len(gf._expt_syringe_concs[0]) == 3
    assert len(gf._expt_syringe_concs[1]) == 3

    # Test points
    num_points = 0
    for expt in this_expt_list:
        num_obs = len(expt.observables)
        num_not_ignore = np.sum(np.logical_not(expt._expt_data["ignore_point"]))
        num_points += num_obs*num_not_ignore

    assert num_points == len(gf._points)

    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list[0].observables["cd222"]["type"] = "not_really"
    with pytest.raises(ValueError):
        gf = GlobalModel(model_name="SixStateEDTA",
                         expt_list=this_expt_list)


def test_GlobalModel_model_normalized(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    # Run with zeros for all parameters -- nothing
    y_calc = gf.model_normalized(gf.parameter_guesses)
    assert np.allclose(np.zeros(len(gf.parameter_guesses)),gf.parameter_guesses)
    assert np.allclose(np.ones(len(y_calc))*0.92377416,y_calc)

    # Run with ln(K) = 10, dH = -10
    y_calc = gf.model_normalized(np.array([10,-10,0]))
    assert np.isclose(y_calc[0],0.9237741556674668)
    assert np.isclose(y_calc[-1],0.9231043096727171)


def test_GlobalModel_model(simulated_itc):
    
    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    # Run with zeros for all parameters -- nothing
    y_calc = gf.model(gf.parameter_guesses)
    assert np.allclose(np.zeros(len(gf.parameter_guesses)),gf.parameter_guesses)
    assert np.allclose(np.zeros(len(y_calc)),y_calc)

    # Run with ln(K) = 10, dH = -10
    y_calc = gf.model(np.array([10,-10,0]))
    assert np.isclose(y_calc[0],0)
    assert np.isclose(y_calc[-1],-0.00949876)


def test_GlobalModel_y_obs(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.y_obs is gf._y_obs

def test_GlobalModel_y_std(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.y_std is gf._y_std


def test_GlobalModel_y_obs_normalized(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.y_obs_normalized is gf._y_obs_normalized

def test_GlobalModel_y_std_normalized(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.y_std_normalized is gf._y_std_normalized


def test_GlobalModel_parameter_names(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.parameter_names is gf._parameter_names


def test_GlobalModel_parameter_guesses(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.parameter_guesses is gf._parameter_guesses

def test_GlobalModel_model_name(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert gf.model_name is gf._model_name


def test_GlobalModel_macro_species(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert np.array_equal(gf.macro_species,gf._bm.macro_species)


def test_GlobalModel_micro_species(simulated_itc):

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    assert np.array_equal(gf.micro_species,gf._bm.micro_species)

def test_GlobalModel_as_df(simulated_itc):

    # partial test of outputs, but should run almost every line of code in the 
    # function to generate outputs

    expt_list = simulated_itc["expt_list"]
    this_expt_list = copy.deepcopy(expt_list)

    gf = GlobalModel(model_name="CaEDTA",
                     expt_list=this_expt_list)
    
    df = gf.as_df.copy()
    assert issubclass(type(df),pd.DataFrame)

    assert np.array_equal(df.columns,["expt_id",
                                      "expt_type",
                                      "expt_obs",
                                      "volume",
                                      "injection",
                                      "CT","ET",
                                      "C","E","EC",
                                      "y_obs",
                                      "y_std",
                                      "y_obs_norm",
                                      "y_std_norm"])
    
    assert np.allclose(gf._y_obs,np.array(df["y_obs"]))
    assert np.allclose(gf._y_std,np.array(df["y_std"]))
    assert np.allclose(gf._y_obs_normalized,np.array(df["y_obs_norm"]))
    assert np.allclose(gf._y_std_normalized,np.array(df["y_std_norm"]))
