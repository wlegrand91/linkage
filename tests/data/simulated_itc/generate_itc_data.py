
import linkage

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def create_fake_itc_data():

    np.random.seed(20070401)
    
    # Randomness we are going to inject into our simulated results
    err = 0.003
    
    # Names of series
    names = ["blank","binding"]

    # Create fake data that has the number of injections we want, but no sane
    # values. 
    itc_data = pd.DataFrame({"injection":2*np.ones(25),
                             "obs_heat":np.random.normal(0,1,25)})
    
    # Create an experiment from the fake data where we titrate ET into an 
    # empty cell
    a = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={},
                                      syringe_contents={"ET":5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    a.define_itc_observable(obs_column="obs_heat",
                            obs_std=0.1)
    
    
    # Create an experiment from the fake data where we titrate ET into a cell
    # with CT. 
    b = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={"CT":0.5e-3},
                                      syringe_contents={"ET":5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    b.define_itc_observable(obs_column="obs_heat",
                            obs_std=0.1)
    
    # Create a linkage model using the CaEDTA binding model and these two 
    # experiments. 
    gm = linkage.GlobalModel(model_name="CaEDTA",
                             expt_list=[a,b])
    
    # [log10('KE'), 'dH_E', 'nuisance_dil_CT', 'nuisance_dil_ET']
    guesses = np.array([7,-11900,0,-50])
    
    # Calculate the values for our model. This creates _y_calc 
    y_calc = gm.model(guesses)

    # Hack for the simulation. Set y_obs and y_std to y_calc
    gm._y_obs = y_calc + np.random.normal(0,err,len(y_calc))
    gm._y_std = np.ones(len(gm._y_obs))*err
    
    # Plot results
    fig, ax = plt.subplots(1,figsize=(6,6))
    df = gm.as_df
    for expt in np.unique(df["expt_id"]):
        this_df = df.loc[df["expt_id"] == expt,:]
        ax.plot(this_df["ET"],this_df["y_obs"],'o',label=names[expt])

    ax.legend()
    ax.set_xlabel("injection number")
    ax.set_ylabel("measured heat")
    fig.savefig("simulated_itc.pdf")
    
    # Now create output files
    for expt in np.unique(gm.as_df["expt_id"]):
        this_df = gm.as_df.loc[gm.as_df["expt_id"] == expt,:]
        
        inj = []
        inj.extend(list(gm._expt_list[expt].expt_data.loc[:,"injection"]))
    
        heat = [np.nan]
        heat.extend(list(this_df.loc[:,"y_obs"]))

        out = {}
        out["injection"] = np.array(inj)
        out["obs_heat"] = np.array(heat)
    
        out_df = pd.DataFrame(out)
    
        out_df.to_csv(f'{names[expt]}_expt.csv',index=None)

create_fake_itc_data()