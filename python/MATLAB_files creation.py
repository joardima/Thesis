# -*- coding: utf-8 -*-

import pickle
import numpy as np
import scipy.io as sio

sp = [
      264,528,792,1848,7392,
      ]
folder  = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories/Msc-Thesis/Pickle/'

for s in sp:
    
#%% LOAD WNTR DATA FROM A PICKLE FILE    
    
    file    = 'wntr_sp_' + str(s) + '_dt_5_Nu_2_cv'
    
    filename        = folder + file + '.pickle'
    
    with open(filename, 'rb') as handle:
        pickle_obj = pickle.load(handle)                                                              
    
    WNTR_DPR    = pickle_obj[0]
    WNTR_IV     = pickle_obj[1]
    WNTR_SER    = pickle_obj[2]
    WNTR_LSER   = pickle_obj[3]
    
    del filename,handle,file,pickle_obj
    
#%% CREATE A .mat OBJECT FOR PATTERN WITHOUT LEAK
    
    sim = WNTR_SER[0]['RESULTS']
    T = np.array(sim.time)
    H = sim.node['head'].to_numpy()
    D = sim.node['demand'].to_numpy()
    Q = sim.link['flowrate'].to_numpy()
    
    mdic = {
            'T': T,
            'H': H,
            'D': D,
            'Q': Q,
            }
    
    file        = 'wntr_sp_' + str(s) + '_dt_5_Nu_2'
    mat_obj     = 'matfiles/' + file + '.mat'
    
    sio.savemat(mat_obj, mdic)
    
    del file,sim,T,H,D,Q,mdic,mat_obj
    
#%% CREATE A .mat OBJECT FOR PATTERN FOR EACH LEAK
    
    for wntr_lser in WNTR_LSER:
        sim     = wntr_lser['RESULTS']
        leak    = str(wntr_lser['LEAK']['Leak'] + 6)
        T       = np.array(sim.time)
        H       = sim.node['head'].to_numpy()
        D1      = sim.node['demand'].to_numpy()
        D2      = sim.node['leak_demand'].to_numpy()
        D       = D1 + D2
        Q       = sim.link['flowrate'].to_numpy()
            
        mdic = {
                'T': T,
                'H': H,
                'D': D,
                'Q': Q,
                }
        
        file        = 'wntr_sp_' + str(s) + '_dt_5_Nu_2_leak_' + leak
        mat_obj     = 'matfiles/' + file + '.mat'
        
        sio.savemat(mat_obj, mdic)
        
    del file,sim,T,H,D,D1,D2,leak,Q,mdic,mat_obj,s

del sp,folder