
import numpy as np

class BindingModel:

    def __init__(self):
        pass
    
    def get_concs(self,K_array,C_array):
        return np.array([],dtype=float)

    @property
    def K_names(self):
        return np.array(["KI","KE","K1","K2","K3","K4"])

    @property
    def macro_species(self):
        return np.array(["AT","CT","ET"])
    
    @property
    def species(self):
        return np.array(["I", "A", "C", "E", "AC1", "AC2", "AC3", "AC4", "EC"])