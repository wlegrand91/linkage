

import pytest

import linkage
from linkage.organizer.global_model import GlobalModel
from linkage.experiment.point.spec_point import SpecPoint
from linkage.experiment.point.itc_point import ITCPoint

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
                          gf.parameter_names[:len(param_names)])
    
def test_GlobalModel__sync_model_and_expt(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # Make sure array starts without AT
    assert np.array_equal(this_expt_list[1].expt_concs.columns,
                          ["injection","volume","meas_vol_dilution","CT","ET"])
    
    # make sure we added AT
    assert np.array_equal(gf._expt_list[1].expt_concs.columns,
                          ["injection","volume","meas_vol_dilution","CT","ET","AT"])
    
    num_points = len(gf._expt_list[1].expt_concs["AT"])
    assert np.array_equal(gf._expt_list[1].expt_concs["AT"],
                          np.zeros(num_points,dtype=float))

def test_GlobalModel__count_expt_points(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    counts = []
    for expt in this_expt_list:
        counts.append(np.sum(np.logical_not(np.array(expt.expt_data["ignore_point"]))))
        counts[-1] = counts[-1]*len(expt.observables)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert np.array_equal(gf._points_per_expt,counts)

    # Set whole experiment to ignore_point is True, turning experiment off
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list[0].expt_data["ignore_point"] = True
    counts[0] = 0

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert np.array_equal(gf._points_per_expt,counts)

def test_GlobalModel__get_enthalpy_param(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf._dh_param_start_idx is not None
    assert gf._dh_param_end_idx is not None

    expected = ['dH_I','dH_E','dH_1','dH_2','dH_3','dH_4',
                "nuisance_dil_AT","nuisance_dil_CT","nuisance_dil_ET"]
    dh_param = gf._all_parameter_names[gf._dh_param_start_idx:gf._dh_param_end_idx + 1]
    assert np.array_equal(expected,dh_param)
    
    # "EC","I","AC1","AC2","AC3","AC4"
    order_in_class = np.array([8,0,4,5,6,7])

    # make sure it is correctly mapping reactions
    assert gf._dh_param_start_idx == 6
    assert gf._dh_param_end_idx == 14
    assert np.array_equal(gf._dh_sign,np.ones(6,dtype=float))
    for i in range(6):
        assert np.sum(gf._dh_product_mask[i]) == 1
        assert np.arange(9,dtype=int)[gf._dh_product_mask[i]] == order_in_class[i]

    assert np.array_equal(gf._dh_sign,[1,1,1,1,1,1])

    # Remove itc experiment; should have no enthalpies
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list.pop(-1)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf._dh_param_start_idx is None
    assert gf._dh_param_end_idx is None
    assert gf._dh_sign is None
    assert gf._dh_product_mask is None

def test_GlobalModel__process_expt_fudge(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf.parameter_names[-1] == "nuisance_expt_0_AT_fudge"
    assert gf.parameter_guesses[-1] == 1.0
    assert gf._fudge_list[0][0] == 0
    assert gf._fudge_list[0][1] == len(gf.parameter_names) - 1
    assert gf._fudge_list[1] is None

def test_GlobalModel__calc_expt_normalization(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    # Basic check of output
    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    assert len(gf._normalization_params) == 3


    assert len(gf._normalization_params["cd222"]) == 2
    assert len(gf._normalization_params["cd240"]) == 2
    assert len(gf._normalization_params["heat"]) == 2

    # Make sure calculated values are correct when we dump in specific values 
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list[0].expt_data["cd222"] = 1
    this_expt_list[0].expt_data.loc[this_expt_list[0].expt_data.index[0],"cd222"] = 0
    this_expt_list[0].expt_data["cd240"] = np.random.choice([1,2],50)
    this_expt_list[1].expt_data["heat"] = np.nan

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # Should be 1 and 0. All values were 1. 
    assert np.isclose(gf._normalization_params["cd222"][0],0.98)
    assert np.isclose(gf._normalization_params["cd222"][1],0.13999999999)

    # There is a small chance this fails numerically if the random choice above
    # was super extreme.
    assert gf._normalization_params["cd240"][0] > 1
    assert gf._normalization_params["cd240"][0] < 2
    assert gf._normalization_params["cd240"][1] > 0.2
    assert gf._normalization_params["cd240"][1] < 0.8

    # These will be 0 and 1 if all values were nan
    assert gf._normalization_params["heat"][0] == 0
    assert gf._normalization_params["heat"][1] == 1
    
    # Make sure it is normalizing correctly between experiments with shared
    # observables. 
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list.append(copy.deepcopy(this_expt_list[0]))
    this_expt_list[0].expt_data["cd222"] = 1
    this_expt_list[2].expt_data["cd222"] = 2

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    # Should be 1 and 0. All values were 1. 
    assert np.isclose(gf._normalization_params["cd222"][0],1.5)
    assert np.isclose(gf._normalization_params["cd222"][1],0.5)

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

    num_spec = sum([issubclass(type(p),SpecPoint) for p in gf._points])
    num_itc = sum([issubclass(type(p),ITCPoint) for p in gf._points])

    assert np.array_equal(gf._points_per_expt,[num_spec,num_itc])

    assert num_points == len(gf._points)

    # Test y_obs, y_std, y_calc
    assert num_points == len(gf._y_obs)
    assert num_points == len(gf._y_std)

    # Test y_obs_normalized, y_std_normalized
    assert num_points == len(gf._y_obs_normalized)
    assert num_points == len(gf._y_std_normalized)
    
    # Test y_norm_mean and y_norm_stdev
    assert num_points == len(gf._y_norm_mean)
    assert num_points == len(gf._y_norm_std)
    
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list[0].observables["cd222"]["type"] = "not_really"
    with pytest.raises(ValueError):
        gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)