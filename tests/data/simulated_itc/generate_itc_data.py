"""
Script used to generate the simulated ITC test fixtures (blank_expt.csv and
binding_expt.csv).  Not run as part of the test suite — the CSV files it
produces are checked in directly.  Re-run this script if the test fixtures
ever need to be regenerated.
"""

import linkage

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


CA_EDTA_SPEC = """
equilibria:
    E + C -> EC; KE

species:
    ET = E + EC
    CT = C + EC
"""


def create_fake_itc_data():

    np.random.seed(20070401)

    err = 0.003
    names = ["blank", "binding"]

    itc_data = pd.DataFrame({"injection": 2 * np.ones(25),
                             "obs_heat": np.random.normal(0, 1, 25)})

    # Blank experiment: titrate ET into an empty cell
    a = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={},
                                      syringe_contents={"ET": 5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    a.define_itc_observable(obs_column="obs_heat", obs_std=0.1)

    # Binding experiment: titrate ET into a cell containing CT
    b = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={"CT": 0.5e-3},
                                      syringe_contents={"ET": 5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    b.define_itc_observable(obs_column="obs_heat", obs_std=0.1)

    gm = linkage.GlobalModel(expt_list=[a, b],
                             model_spec=CA_EDTA_SPEC)

    # [log(KE), dH_E, nuisance_dil_CT, nuisance_dil_ET]
    guesses = np.array([7, -11900, 0, -50])

    y_calc = gm.model(guesses)

    gm._y_obs = y_calc + np.random.normal(0, err, len(y_calc))
    gm._y_std = np.ones(len(gm._y_obs)) * err

    fig, ax = plt.subplots(1, figsize=(6, 6))
    df = gm.as_df
    for expt in np.unique(df["expt_id"]):
        this_df = df.loc[df["expt_id"] == expt, :]
        ax.plot(this_df["ET"], this_df["y_obs"], 'o', label=names[expt])

    ax.legend()
    ax.set_xlabel("injection number")
    ax.set_ylabel("measured heat")
    fig.savefig("simulated_itc.pdf")

    for expt in np.unique(gm.as_df["expt_id"]):
        this_df = gm.as_df.loc[gm.as_df["expt_id"] == expt, :]

        inj = list(gm._expt_list[expt].expt_data.loc[:, "injection"])

        heat = [np.nan]
        heat.extend(list(this_df.loc[:, "y_obs"]))

        out_df = pd.DataFrame({"injection": np.array(inj),
                                "obs_heat": np.array(heat)})
        out_df.to_csv(f'{names[expt]}_expt.csv', index=None)


create_fake_itc_data()
