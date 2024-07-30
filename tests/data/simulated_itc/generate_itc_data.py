
import linkage

import numpy as np
import pandas as pd

def create_fake_itc_data():

    np.random.seed(20070401)
    
    # Load ITC data
    itc_data = pd.DataFrame({"injection":2*np.ones(25),
                             "heat":np.random.normal(0,1,25)})
    
    a = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={},
                                      syringe_contents={"ET":5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    a.define_itc_observable(obs_column="heat",
                            obs_stdev=0.1)
    
    
    b = linkage.experiment.Experiment(expt_data=itc_data.copy(),
                                      cell_contents={"CT":0.5e-3},
                                      syringe_contents={"ET":5e-3},
                                      conc_to_float=None,
                                      cell_volume=280)
    b.define_itc_observable(obs_column="heat",
                            obs_stdev=0.1)
    
    expt_list = [a,b] 
    
    gm = linkage.GlobalModel(model_name="CaEDTA",
                             expt_list=expt_list)
    
    # log10(KE), dH_C, dH_E, dH_EC
    guesses = np.array([7,-11900,0,-50])
    
    err = 0.003
    initial_values = gm.model(guesses)
    gm._y_obs = gm._y_calc + np.random.normal(0,err,len(gm._y_calc))
    gm._y_stdev = np.ones(len(gm._y_obs))*err
    
    df = gm.as_df
    
    # for expt in np.unique(df["expt_id"]):
    #     this_df = df.loc[df["expt_id"] == expt,:]
    #     plt.plot(this_df["ET"],this_df["y_obs"],'o')
    
    
    names = ["blank","binding"]
    for expt in np.unique(gm.as_df["expt_id"]):
        this_df = gm.as_df.loc[gm.as_df["expt_id"] == expt,:]
        
        inj = []
        inj.extend(list(gm._expt_list[expt].expt_data.loc[:,"injection"]))
    
        heat = [np.nan]
        heat.extend(list(this_df.loc[:,"y_obs"]))

        out = {}
        out["injection"] = np.array(inj)
        out["heat"] = np.array(heat)
    
        out_df = pd.DataFrame(out)
    
        out_df.to_csv(f'{names[expt]}_expt.csv',index=None)

create_fake_itc_data()