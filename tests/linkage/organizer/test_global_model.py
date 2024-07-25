

import pytest

import linkage
from linkage.organizer.global_model import GlobalModel
from linkage.experiment.experimental_point import SpecPoint
from linkage.experiment.experimental_point import ITCPoint

import numpy as np
import pandas as pd
import copy

def test_GlobalModel_integrated(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)
    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    gf.model(gf.parameter_guesses)

    df = gf.as_df
    assert issubclass(type(df),pd.DataFrame)


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

    expected = ['dH_I','dH_A','dH_C','dH_E',
                'dH_AC1','dH_AC2','dH_AC3','dH_AC4','dH_EC']
    dh_param = gf._all_parameter_names[gf._dh_param_start_idx:gf._dh_param_end_idx + 1]

    assert np.array_equal(expected,dh_param)

    # Remove itc experiment; should have no enthalpies
    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list.pop(-1)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf._dh_param_start_idx is None
    assert gf._dh_param_end_idx is None

def test_GlobalModel__process_expt_fudge(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)

    this_expt_list = copy.deepcopy(base_expt_list)
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    assert gf.parameter_names[-1] == "expt_0_AT_fudge"
    assert gf.parameter_guesses[-1] == 1.0
    assert gf._fudge_list[0][0] == 0
    assert gf._fudge_list[0][1] == len(gf.parameter_names) - 1
    assert gf._fudge_list[1] is None

def test_GlobalModel__build_point_map(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)  

    this_expt_list = copy.deepcopy(base_expt_list)  
    gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)
    
    assert len(gf._micro_arrays) == 2
    assert gf._micro_arrays[0].shape[1] == len(gf._bm.micro_species)
    assert gf._micro_arrays[1].shape[1] == len(gf._bm.micro_species)

    assert len(gf._macro_arrays) == 2
    assert gf._macro_arrays[0].shape[1] == len(gf._bm.macro_species)
    assert gf._macro_arrays[1].shape[1] == len(gf._bm.macro_species)

    num_points = 0
    for expt in this_expt_list:
        num_obs = len(expt.observables)
        num_not_ignore = np.sum(np.logical_not(expt._expt_data["ignore_point"]))
        num_points += num_obs*num_not_ignore

    
    num_spec = sum([issubclass(type(p),SpecPoint) for p in gf._points])
    num_itc = sum([issubclass(type(p),ITCPoint) for p in gf._points])

    assert np.array_equal(gf._points_per_expt,[num_spec,num_itc])

    assert num_points == len(gf._points)
    assert num_points == len(gf._y_obs)
    assert num_points == len(gf._y_stdev)
    assert num_points == len(gf._y_calc)


    this_expt_list = copy.deepcopy(base_expt_list)
    this_expt_list[0].observables["cd222"]["obs_type"] = "not_really"
    with pytest.raises(ValueError):
        gf = GlobalModel(model_name="SixStateEDTA",
                     expt_list=this_expt_list)