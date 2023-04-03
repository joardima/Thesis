# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 09:56:42 2023

@author: jdi004
"""

import pickle
import numpy as np
import scipy.io as sio
import math

sp = [
      24,48,72,168,672
#      264,528,792,1848,7392,
      ]
folder  = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories/Msc-Thesis/Pickle/'

df_list = [ ]
for s in sp:
    
#% LOAD WNTR DATA FROM A PICKLE FILE    
    
    file_1      = 'wntr_sp_' + str(s) + '_dt_5_Nu_2_with_leaks'
    file_2      = 'wntr_sp_' + str(s) + '_dt_5_Nu_2_with_leaks_6to8'

    filename_1  = folder + file_1 + '.pickle'
    filename_2  = folder + file_2 + '.pickle'
    
    with open(filename_1, 'rb') as handle:
        pickle_obj = pickle.load(handle)                                                              
    
    WNTR_DPR_1      = pickle_obj[0]
    WNTR_IV_1      = pickle_obj[1]
    WNTR_SER_1      = pickle_obj[2]
    WNTR_LSER_1     = pickle_obj[3]
    
    with open(filename_2, 'rb') as handle:
        pickle_obj = pickle.load(handle)                                                              
    
    WNTR_DPR_2      = pickle_obj[0]
    WNTR_IV_2      = pickle_obj[1]
    WNTR_SER_2      = pickle_obj[2]
    WNTR_LSER_2     = pickle_obj[3]
    
    del filename_1,handle,file_1,pickle_obj,filename_2,file_2,
    
    a       = WNTR_IV_1['LEAKS']; b = WNTR_IV_2['LEAKS']
    leaks   = a.append(b, ignore_index=True).sort_values(by=['Aleak'])
    leaks   = leaks.reset_index(drop=True)
    
    del a,b,
    
    leaks['diameter']   = (4*leaks['Aleak']/math.pi)**0.5/2.54
    leaks['sp']         = s
    leaks['dt']         = 5
    leaks['T']          = (leaks['T2'] - leaks['T1'])*leaks['sp']
    
    c = np.append(WNTR_LSER_1,WNTR_LSER_2)
    res_list = [ ]
    for arr_row in c[:]:
        area        = arr_row['LEAK']['Area_leak']
        end_time    = arr_row['LEAK']['End_time']
        for index, row in leaks.iterrows():            
            if area - row['Aleak'] == 0 and abs(end_time - row['T2']*row['sp']*3600) < row['dt']*60:
                res_list.append(arr_row['RESULTS'])
    del c,arr_row,area,end_time
    
    min_list =[ ]; max_list = [ ]
    for t in res_list:
        a = t.node['leak_demand']
        qmin = a['6'][a['6']!=0].min()
        qmax = a['6'][a['6']!=0].max()
        min_list.append(qmin*1000)
        max_list.append(qmax*1000)

    leaks['q_min'] = min_list
    leaks['q_max'] = max_list
    
    df_list.append(leaks)

leaks = df_list[0]
for temp in df_list[1:]:
    leaks = leaks.append(temp)

leaks               = leaks.drop(['Cd','Junctions'], axis=1)
leaks               = leaks[[leaks.columns[i] for i in l]]
l                   = [5,0,1,2,3,4,6,7,8,]
leaks               = leaks[[leaks.columns[i] for i in l]]
        
                             
#%%

WNTR_IV_2['LEAKS']
A = WNTR_LSER[0]['RESULTS'].node['leak_demand']['6']
minValue = A.col1[A.6!=0].min()
maxValue = A.col1[A.6!=0].max()

temp = [ ]
for arr_row in c:
    for index, row in leaks.iterrows():
        if arr_row['LEAK']['Area_leak'] == row['A_leak'] and arr_row['LEAK']['End_time'] - row['T2']*row['sp'] <= row['dt']*60:
            temp.append(arr_row['RESULTS'])
