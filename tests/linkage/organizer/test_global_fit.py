

import pytest

import linkage
from linkage.organizer.global_fit import GlobalFit

import numpy as np
import copy

def test_GlobalFit_integrated(fake_spec_and_itc_data):

    base_expt_list = copy.deepcopy(fake_spec_and_itc_data)


    this_expt_list = copy.deepcopy(base_expt_list)

    gf = GlobalFit(model_name="SixStateEDTA",
                   expt_list=this_expt_list)
    
    gf.total_model(gf.parameter_guesses)
