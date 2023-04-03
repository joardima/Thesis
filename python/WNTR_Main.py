# -*- coding: utf-8 -*-

"""
Before runing the script, the INPUT VARIABLES should be defined in the script 'Input_variables'

INP_VAR:                Dictionary imported from the file 'Input_variables'
INP_VAR_IM_PySINDy:     Dictionary imported from the file 'Input_variables'

This script run the following functions:
    WNTR_Demand_Pattern:         results stored in the variable 'RESULTS_DEMAND_PATTERN'
    WNTR_Sim_EPANET:             results stored in the variable 'RESULTS_Sim_EPANET'
    from_EPANET_to_PySINDy:      results stored in the variable 'RESULTS_EPANET_to_PySINDy'
    
    
FUNCTIONS OUTPUTS
    WNTR_Input_Variables:   A dictionary with the input variables
    WNTR_Demand_Pattern:    A list of results per alternative
                            One alternative per DT
                            Each list per alternative contains:
                                A dictionary with input parameters
                                A list with timesteps based on the input DT
                                A list with the average day pattern multiplier
                                A list with the friday day pattern multiplier
                                A list with the weekdays pattern multiplier
    
    WNTR_Sim_EPANET:        An array with size 3 x number of DT (rows x columns)
                            Each column of the aforementioned array contains:
                                DT value
                                WNTR water network model
                                WNTR simulation results

    from_WNTR_to_PySINDy:   An list with a number of elements equal to the number of DT
                            Each element contains:
                                DT value
                                A list with the name of the features/elements:
                                    P: pressure in m3/s,
                                    D: Demand in m3/s,
                                    Q: Flow in in m3/s
                                    Q^1.852: Flow power
                                An array with the results per each timestep per features/elements
                            
"""

#IMPORT LIBRARIES

import numpy as np
import pandas as pd
import pickle
import time

START_TIME = time.time()

# IMPORT INPUT VARIABLES

from WNTR_Input_Variables import WNTR_Input_Variables as WNTR_IV

# RUN THE FUNCTION 'WNTR_Demand_Pattern'

from WNTR_Demand_Pattern import WNTR_Demand_Pattern

WNTR_DPR = WNTR_Demand_Pattern(
    WNTR_IV['BASE_PAT_MULT'],
    WNTR_IV['BASE_PAT_FRIDAY_TIME_CHANGE'],
    WNTR_IV['BASE_PAT_FRIDAY_INCREMENT'],
    WNTR_IV['WEEKEND_DELAY_TIME'],
    WNTR_IV['ALTERNATIVES']
    )

# RUN THE FUNCTION 'WNTR_Sim_EPANET'

from WNTR_Sim_EPANET import WNTR_Sim_EPANET

WNTR_SER = WNTR_Sim_EPANET(
    WNTR_IV['ALTERNATIVES'],
    WNTR_DPR,
    WNTR_IV['NETWORK_FILE'],
    WNTR_IV['SCRIPT_FILE'],
    WNTR_IV['TEST_FOLDER']
    )

END_TIME = time.time()

print("Overall execution time: {:.2f} seconds".format(END_TIME - START_TIME))

del START_TIME,END_TIME

#%%

# RUN THE FUNCTION 'WNTR_Leak_Sim_EPANET'

from WNTR_Leak_Sim_EPANET import WNTR_Leak_Sim_EPANET

WNTR_LSER = WNTR_Leak_Sim_EPANET(
    WNTR_IV['ALTERNATIVES'],
    WNTR_DPR,
    WNTR_IV['NETWORK_FILE'],
    WNTR_IV['SCRIPT_FILE'],
    WNTR_IV['TEST_FOLDER'],
    WNTR_IV['LEAKS'],
    WNTR_IV['LEAKS_EPANET']
    )

#END_TIME = time.time()

#print("Overall execution time: {:.2f} seconds".format(END_TIME - START_TIME))

#del START_TIME,END_TIME

#%% CREATE A PICKLE FILE WITH WNTR RESULTS

sp = int(WNTR_IV['ALTERNATIVES']['SP'].to_numpy()[0])
dt = int(WNTR_IV['ALTERNATIVES']['HTS'].to_numpy()[0])
nu = int(WNTR_IV['ALTERNATIVES']['NOISE'].to_numpy()[0])

results = [
    WNTR_DPR,
    WNTR_IV,
    WNTR_SER,
    WNTR_LSER,
    ]

folder      = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories/Msc-Thesis/Pickle'
name        = '/wntr_sp_' + str(sp) + '_dt_' + str(dt) + '_Nu_' + str(nu) + '_cv'
 
filename    = folder + name + '.pickle'

with open(filename, 'wb') as handle:
    pickle.dump(
        obj         = results,
        file        = handle,
        protocol    = pickle.HIGHEST_PROTOCOL            # Pickle the 'data' dictionary using the highest protocol available.
        ) 

del name,folder,handle,filename,sp,dt,nu,results


#%% 

# RUN THE FUNCTION 'from_WNTR_to_PySINDy'

from from_WNTR_to_PySINDy import from_WNTR_to_PySINDy

FWTPR = from_WNTR_to_PySINDy(
    WNTR_IV['FLOW_EXPONENT'],
    WNTR_IV['FLOW_MULTIPLIER'],
    WNTR_SER,
    )

END_TIME = time.time()

print("Overall execution time: {:.2f} seconds".format(END_TIME - START_TIME))

del START_TIME,END_TIME

#%% PLOT ONE PATTERN (BASE, WEEKDAY, FRIDAY, WEEKENDS)

from Plotting_Interpolate_Patterns import Plot_Pattern

PN  = 54               # Pattern number
Plot_Pattern(
        WNTR_IV,
        PN,
        WNTR_DPR
        )
