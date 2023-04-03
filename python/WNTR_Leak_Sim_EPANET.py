# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:33:30 2023

@author: jdi004

"""

def WNTR_Leak_Sim_EPANET(
        ALTERNATIVES,
        DEMAND_PATTERN_RESULTS,
        NETWORK_FILE,
        SCRIPT_FILE,
        TEST_FOLDER,
        LEAKS,
        LEAKS_EPANET
        ):
    
# IMPORT LIBRARIES

    import numpy as np
    import wntr
    import pandas as pd
    import math
    #import xlsxwriter
    #import time
    #import os

    ALTERNATIVES            = ALTERNATIVES
    WNTR_DPR                = DEMAND_PATTERN_RESULTS
    NETWORK_FILE            = NETWORK_FILE
    SCRIPT_FILE             = SCRIPT_FILE
    TEST_FOLDER             = TEST_FOLDER
    LEAKS                   = LEAKS
    LEAKS_EPANET            = LEAKS_EPANET

    RETURN_RESULTS  = np.array([ ])
    
    for index, row in ALTERNATIVES.iterrows():
        
# DEFINE NETWORK OPTIONS
        
        PN                  = str(index)                            #Pattern number
        PATTERN             = WNTR_DPR[index]['AVEDAY_PAT_MULT']
        SIMULATION_PERIOD   = row['SP']
        HYDRAULIC_TIMESTEP  = row['HTS']
        PATTERN_TIMESTEP    = row['HTS']                      
        REPORT_TIMESTEP     = int(row['HTS'])
        NOISE               = row['NOISE']
        
# INPUT FILES AND FOLDERS NAMES
        
        NETWORK_FILE    = str(NETWORK_FILE)
        TEST_NAME       = 'SP_' + str(int(SIMULATION_PERIOD)) + '_HTS_' + str(int(HYDRAULIC_TIMESTEP)) + '_Noise_' + str(int(NOISE)) #+ '_patt_ts_' + str(PATTERN_TIMESTEP) + '_rep_ts_' + str(REPORT_TIMESTEP)
        INP_FILE        = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories\Msc-Thesis/INP_FILES/' + NETWORK_FILE
        NETWORK         = NETWORK_FILE.split('_')
        NETWORK         = NETWORK[0]
        
# CREATE A NETWORK MODEL
        
        WN = wntr.network.WaterNetworkModel(INP_FILE) 
        
# DEFINE THE PATTERNS
        
        WN.add_pattern(PN, pattern = PATTERN)
        
# CHANGE NETWORK OPTIONS
        
        WN.options.time.duration            = SIMULATION_PERIOD*3600
        WN.options.time.hydraulic_timestep  = HYDRAULIC_TIMESTEP*60
        WN.options.time.pattern_timestep    = PATTERN_TIMESTEP*60
        WN.options.time.report_timestep     = REPORT_TIMESTEP*60
        #WN.options.time.pattern             = PN
        
# OBTAIN A DATAFRAME WITH THE NAME OF THE NODES
        
        JUNCTION_NAME = pd.DataFrame(
            WN.junction_name_list,
            columns =['name']
            )
        
# CHANGE PATTERN IN NODES       
        
        JUNC          = JUNCTION_NAME.shape[0]
        junc           = 0
        while junc < JUNC:
            junction = WN.get_node(JUNCTION_NAME.iloc[junc]['name'])
            junction.demand_timeseries_list[0].pattern_name = PN
            junc += 1
        
# INPUT A LEAKAGE IN NODES (WNTRSimulator)

        for lind, ro in LEAKS.iterrows(): 
            
            LEAK        = lind
            Cd          = ro['Cd']
            Aleak       = ro['Aleak']
            JUNCTIONS   = ro['Junctions'] 
            T1          = ro['T1']
            T2          = ro['T2']

            T1          = int(
                T1*(SIMULATION_PERIOD*3600/(HYDRAULIC_TIMESTEP*60))
                )*HYDRAULIC_TIMESTEP*60
            T2          = int(
                T2*(SIMULATION_PERIOD*3600/(HYDRAULIC_TIMESTEP*60))
                )*HYDRAULIC_TIMESTEP*60
                       
            if JUNCTIONS == 0:
                JUNCTIONS = 1
            else:
                JUNCTIONS = math.ceil(JUNCTION_NAME.shape[0]*JUNCTIONS)
            
            np.random.seed(0)
            JUNCTION    = np.random.choice(
                JUNCTION_NAME.to_numpy().flatten(),
                size=JUNCTIONS,
                replace=0,
                )
            
            for j in JUNCTION:
                node = WN.get_node(j)
                node.add_leak(
                    wn                  = WN,
                    area                = Aleak/10**4,
                    discharge_coeff     = Cd,
                    start_time          = T1,
                    end_time            = T2
                    )
                leak_temp   = { }
                leak_temp   = {
                'Leak':         LEAK,
                'Cd':           Cd,
                'Area_leak':    Aleak,
                'Start_time':   T1,
                'End_time':     T2,
                'Junc_leak':    JUNCTION
                }

# INPUT A LEAKAGE IN NODES (EpanetSimulator)

#        for lind, row in LEAKS_EPANET.iterrows(): 
#            leak_temp   = np.array([ ])
#            LEAK        = lind
#            Qleak       = row['Qleak']
#            T1          = row['T1']*round(SIMULATION_PERIOD*3600/HYDRAULIC_TIMESTEP*60)*HYDRAULIC_TIMESTEP*60
#            T2          = row['T2']*round(SIMULATION_PERIOD*3600/HYDRAULIC_TIMESTEP*60)*HYDRAULIC_TIMESTEP*60
#            NODES       = row['Nodes']
#            leak_temp   = np.append(
#                leak_temp,
#                (LEAK,Qleak,T1,T2,NODES)
#                )  
            
#            ROWS          = JUNCTION_NAME.shape[0]
#            ROW           = 0
#            while ROW < ROWS:
#                node = WN.get_node(JUNCTION_NAME.iloc[ROW]['name'])
#                node.add_leak(
#                    wn                  = WN,
#                    area                = Aleak,
#                    discharge_coeff     = Cd,
#                    start_time          = T1,
#                    end_time            = T2
#                    )
#                ROW += 1
      
# SAVE THE WATER NETOWORK MODEL
        
            EPS_RESULTS_INPFILE = str(TEST_FOLDER) + '/' + NETWORK + '_' + str(TEST_NAME) + '_Leak_' + str(lind) + '.inp'
            WN.write_inpfile(EPS_RESULTS_INPFILE, WN)
            
# CREATE SIMULATION OBJECTS
            
            SIM = wntr.sim.WNTRSimulator(WN)
            
# CREATE RESULT OBJECTS
            
            RESULTS = SIM.run_sim()
        
            RESULT_DICT     = {
                'Sim_Period':       SIMULATION_PERIOD,
                'Hydraulic_TS':     HYDRAULIC_TIMESTEP,
                'Pattern_TS':       PATTERN_TIMESTEP,
                'Report_TS':        REPORT_TIMESTEP,
                'WN':               WN,
                'RESULTS':          RESULTS,
                'LEAK':             leak_temp, 
                #'SIM':              SIM
                }
                
            RETURN_RESULTS      = np.append(
                RETURN_RESULTS,
                RESULT_DICT,
                )
            
            for j in JUNCTION:              #Loop for removing the added leaks
                node = WN.get_node(j)
                node.remove_leak(WN)
            
            WN.reset_initial_values()
            
    return RETURN_RESULTS