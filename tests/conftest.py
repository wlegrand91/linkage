import pytest

import linkage

import pandas as pd
import numpy as np

import os
import glob

def get_files(base_dir):
    """
    Traverse base_dir and return a dictionary that keys all files and some
    rudimentary *.ext expressions to absolute paths to those files. They keys
    will be things like "some_dir/test0/rocket.txt" mapping to
    "c:/some_dir/life/base_dir/some_dir/test0/rocket.txt". The idea is to have
    easy-to-read cross-platform keys within unit tests.

    Classes of keys:

        + some_dir/test0/rocket.txt maps to a file (str)
        + some_dir/test0/ maps to the test0 directory itself (str)
        + some_dir/test0/*.txt maps to all .txt (list)
        + some_dir/test0/* maps to all files or directories in the directory
          (list)

    Note that base_dir is *not* included in the keys. All are relative to that
    directory by :code:`os.path.basename(__file__)/{base_dir}`.

    Parameters
    ----------
    base_dir : str
        base directory for search. should be relative to test file location.

    Returns
    -------
    output : dict
        dictionary keying string paths to absolute paths
    """

    containing_dir = os.path.dirname(os.path.realpath(__file__))
    starting_dir = os.path.abspath(os.path.join(containing_dir,base_dir))

    base_length = len(starting_dir.split(os.sep))

    # Traverse starting_dir
    output = {}
    for root, dirs, files in os.walk(starting_dir):

        # path relative to base_dir as a list
        this_path = root.split(os.sep)[base_length:]

        # Build paths to specific files
        local_files = []
        for file in files:
            local_files.append(os.path.join(root,file))
            new_key = this_path[:]
            new_key.append(file)
            output["/".join(new_key)] = local_files[-1]

        # Build paths to patterns of file types
        patterns = {}
        ext = list(set([f.split(".")[-1] for f in local_files]))
        for e in ext:
            new_key = this_path[:]
            new_key.append(f"*.{e}")
            output["/".join(new_key)] = glob.glob(os.path.join(root,f"*.{e}"))

        # Build path to all files in this directory
        new_key = this_path[:]
        new_key.append("*")
        output["/".join(new_key)] = glob.glob(os.path.join(root,f"*"))

        # Build paths to directories in this directory
        for this_dir in dirs:
            new_key = this_path[:]
            new_key.append(this_dir)
            # dir without terminating /
            output["/".join(new_key)] = os.path.join(root,this_dir)

            # dir with terminating /
            new_key.append("")
            output["/".join(new_key)] = os.path.join(root,this_dir)

    # make sure output is sorted stably
    for k in output:
        if issubclass(type(output[k]),str):
            continue

        new_output = list(output[k])
        new_output.sort()
        output[k] = new_output

    return output

@pytest.fixture(scope="module")
def simulated_itc():

    files = get_files(os.path.join("data","simulated_itc"))
        
    blank = linkage.Experiment(expt_data=files["blank_expt.csv"],
                               cell_contents={},
                               syringe_contents={"ET":5e-3},
                               cell_volume=280)
    blank.define_itc_observable(obs_column="obs_heat",
                                obs_std=0.003)
    
    expt = linkage.Experiment(expt_data=files["binding_expt.csv"],
                              cell_contents={"CT":0.5e-3},
                              syringe_contents={"ET":5e-3},
                              cell_volume=280)
    expt.define_itc_observable(obs_column="obs_heat",
                               obs_std=0.003)
    
    guesses = np.array([7,-11900,0,-50])

    return {"files":files,
            "expt_list":[blank,expt],
            "guesses":guesses}

@pytest.fixture(scope="module")
def fake_spec_and_itc_data():

    # Fake data
    expt_data = pd.DataFrame({"injection":25*np.ones(50),
                            "cd222":np.random.normal(0,1,50),
                            "cd240":np.random.normal(0,1,50)})
    expt_data.loc[expt_data.index[0],"injection"] = 0.0

    itc_data = pd.DataFrame({"injection":25*np.ones(50),
                            "obs_heat":np.random.normal(0,1,50)})


    # Load spec data
    e = linkage.experiment.Experiment(expt_data,
                                    cell_contents={"AT":50e-6,
                                                    "CT":0.5e-3},
                                    syringe_contents={"ET":1e-3},
                                    conc_to_float="AT",
                                    cell_volume=1800)

    e.define_spectroscopic_observable(obs_column="cd222",
                                      obs_std=0.1,
                                      obs_microspecies="I",
                                      obs_macrospecies="AT")
    e.define_spectroscopic_observable(obs_column="cd240",
                                      obs_std=0.1,
                                      obs_microspecies=["I","A"],
                                      obs_macrospecies="AT")

    # Load ITC data
    f = linkage.experiment.Experiment(itc_data,
                cell_contents={"CT":0.5e-3},
                syringe_contents={"ET":1e-3},
                conc_to_float=None,
                cell_volume=1800)
    f.define_itc_observable(obs_column="obs_heat",
                            obs_std=0.1)

    expt_list = [e,f]

    return expt_list