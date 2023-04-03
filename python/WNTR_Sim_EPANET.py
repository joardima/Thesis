# -*- coding: utf-8 -*-

'''
Before runing the script, the INPUT VARIABLES should be defined in the script 'Input_variables'

TEST_No should be a list of str with name/number of each test

'''

def WNTR_Sim_EPANET(
        ALTERNATIVES,
        DEMAND_PATTERN_RESULTS,
        NETWORK_FILE,
        SCRIPT_FILE,
        TEST_FOLDER
        ):
    
    # IMPORT LIBRARIES
    import numpy as np
    import wntr
    import pandas as pd
    import time

    start_time = time.time()

    ALTERNATIVES            = ALTERNATIVES
    WNTR_DPR                = DEMAND_PATTERN_RESULTS
    NETWORK_FILE            = NETWORK_FILE
    SCRIPT_FILE             = SCRIPT_FILE
    TEST_FOLDER             = TEST_FOLDER

    RETURN_RESULTS  = np.array([ ])
    
    
    sim_epanet_number = 0
    for index, row in ALTERNATIVES.iterrows():
        start_sim_epanet_time = time.time()

# NETWORK OPTIONS
        
        PN                  = str(index) #Pattern number
        PATTERN             = WNTR_DPR[index]['AVEDAY_PAT_MULT']
        SIMULATION_PERIOD   = row['SP']
        HYDRAULIC_TIMESTEP  = row['HTS']
        PATTERN_TIMESTEP    = row['HTS']                      
        REPORT_TIMESTEP     = int(row['HTS'])
        NOISE               = row['NOISE']
        
# INPUT FILES AND FOLDERS NAMES
        
        NETWORK_FILE    = str(NETWORK_FILE)
        SCRIPT_FILE     = str(SCRIPT_FILE)                                       
        TEST_NAME       = 'SP_' + str(int(SIMULATION_PERIOD)) + '_HTS_' + str(int(HYDRAULIC_TIMESTEP)) + '_Noise_' + str(int(NOISE))#+ '_patt_ts_' + str(PATTERN_TIMESTEP) + '_rep_ts_' + str(REPORT_TIMESTEP)
        INP_FILE        = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories\Msc-Thesis/INP_FILES/' + NETWORK_FILE
        NETWORK         = NETWORK_FILE.split('_')
        NETWORK         = NETWORK[0]
        
# CREATE A NETWORK MODEL
        
        WN = wntr.network.WaterNetworkModel(INP_FILE) 
        
# OBTAIN MODEL NETWORK ELEMENTS

        junctions   = WN.num_junctions
        reservoirs  = WN.num_reservoirs
        pipes       = WN.num_pipes         
        
# PATTERNS
        
        WN.add_pattern(PN, pattern = PATTERN)
        
# CHANGING NETWORK OPTIONS
        
        WN.options.time.duration            = SIMULATION_PERIOD*3600
        WN.options.time.hydraulic_timestep  = HYDRAULIC_TIMESTEP*60
        WN.options.time.pattern_timestep    = PATTERN_TIMESTEP*60
        WN.options.time.report_timestep     = REPORT_TIMESTEP*60
        #WN.options.time.pattern             = PN
        
# CHANGING PATTERN
        
        JUNCTION_NAME = pd.DataFrame(
            WN.junction_name_list,
            columns =['name']
            )
        
        ROWS          = JUNCTION_NAME.shape[0]
        ROW           = 0
        while ROW < ROWS:
            junction = WN.get_node(JUNCTION_NAME.iloc[ROW]['name'])
            junction.demand_timeseries_list[0].pattern_name = PN
            ROW += 1

# SAVE THE WATER NETOWORK MODEL
        
        EPS_RESULTS_INPFILE = str(TEST_FOLDER) + '/' + NETWORK + '_' + str(TEST_NAME) + '.inp'
        WN.write_inpfile(EPS_RESULTS_INPFILE, WN)
        
# CREATE SIMULATION OBJECTS
        
        SIM = wntr.sim.WNTRSimulator(WN)
        
# CREATE RESULT OBJECTS
        
        RESULTS = SIM.run_sim()

        RESULT_DICT     = {
            'Sim_Period':           SIMULATION_PERIOD,
            'Hydraulic_TS':         HYDRAULIC_TIMESTEP,
            'Pattern_TS':           PATTERN_TIMESTEP,
            'Report_TS':            REPORT_TIMESTEP,
            'WN':                   WN,
            'RESULTS':              RESULTS,
            'ELEMENTS':             np.array([
                junctions + reservoirs,
                junctions,
                pipes
                ]),
            'INDEXES':              np.array([
                junctions + reservoirs,
                junctions + reservoirs + junctions,
                junctions + reservoirs + junctions + pipes
                ]),
            'JUNCTION_NAME':        JUNCTION_NAME,
            'EPS_RESULTS_INPFILE':  EPS_RESULTS_INPFILE
            }
        
        RETURN_RESULTS      = np.append(
            RETURN_RESULTS,
            RESULT_DICT,
            )
        
        WN.reset_initial_values()
        
        end_sim_epanet_time = time.time()
        sim_epanet_time = (end_sim_epanet_time - start_sim_epanet_time)
    
        print('performed simulation {0:.0f}'.format(sim_epanet_number))
        print('model simulation time {0:.3f}'.format(sim_epanet_time))        
        
        sim_epanet_number += 1
        
    end_time = time.time()
    run_time = (end_time - start_time)/60
    
    print('total simulated models {0:.2f}'.format(sim_epanet_number))
    print('total simulation time {0:.2f}'.format(run_time))
    
    del start_time,end_time,run_time,sim_epanet_number,end_sim_epanet_time,start_sim_epanet_time
    
    return RETURN_RESULTS