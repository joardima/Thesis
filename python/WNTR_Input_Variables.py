# -*- coding: utf-8 -*-

"""
Created on Tue Nov 29 16:00:30 2022
@author: jdi004

FLOW_EXPONENT:              based on reference 10
                            0.852 when using Hazen-Williams equation
                            1.0 when using Darcy-Weisbach equation
FLOW_MULTIPLIER:            coefficient for converting node demand from m3/s

"""
# IMPORTING LIBRARIES

import numpy as np
import pandas as pd
import math

# INPUT VARIABLES FOR 'WNTR_Demand_Pattern' FUNCTION

BASE_PAT_MULT                   = np.array([
    0.9, 0.6, 0.5, 0.4, 0.3, 0.4,
    0.5, 1.2, 1.6, 1.6, 1.5, 1.3,
    1.1, 1, 1, 0.9, 0.9, 1, 1.1,
    1.2, 1.3, 1.3, 1.3, 1.2, 0.9
    ])

BASE_PAT_FRIDAY_TIME_CHANGE     = 18                                
BASE_PAT_FRIDAY_INCREMENT       = 10

PAT_DT_LIST                     = ([
    5,
    ])

WEEKEND_DELAY_TIME              = 2

NOISE                           = [
    2,
    ] 

# INPUT VARIABLES FOR 'WNTR_Sim_EPANET' FUNCTION

SIMULATION_PERIODS  = [
    7392,
    #24,48,72,168,672
    #264,528,792,1848,7392,
    ]
NETWORK_FILE        = 'Twoloops_Base_WNTR.inp'       #Root after 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories\Msc-Thesis/INP_FILES/' and name of the .inp file
SCRIPT_FILE         = 'WNTR_Sim_EPANET'                                       
TEST_FOLDER         = 'D:/OneDrive/IHE_HI/015. Thesis/Github_repositories/Msc-Thesis/INP_FILES_WNTR'     #Root of the folder on where to save new EPANET input file(s), .inp, for each simulation

# INPUT VARIABLES FOR 'WNTR_Leak_Sim_EPANET' FUNCTION

Dleak   = np.array([
    1
    ])                              # Asuming circular hole, inches 
Cd      = np.array([
    0.75                                # Assuming turbulent flow, unitless
    ])
Aleak   = (Dleak*2.54)**2*math.pi/4   # Area of the leakage hole, cm2
Qleak   = np.array([                    # Flow of the leak in m3/s, for EPANET Simulator
    1,2,3,4
    ])/1000
T1      = np.array([                    # Initial timestep to simulate the leakage, percentage of the total simulation period
    30
    ])/100   
                     
T2      = np.array([                    # Final timestep to simulate the leakage, percentage of the total simulation period
   60 
#    (30 + (6/SIMULATION_PERIODS[0])*100),
#    (30 + (81/SIMULATION_PERIODS[0])*100),
#    (30 + (168/SIMULATION_PERIODS[0])*100),
    ])/100                         
JUNCTIONS = np.array([                      # Percentage of nodes to input the leakage. Zero (0) means one node
    0,
    ])/100

# INPUT VARIABLES FOR 'WNTR_to_PySINDy' FUNCTION

FLOW_EXPONENT       = 0.852           
FLOW_MULTIPLIER     = 1
DT_MULTIPLIER       = 1

#MATCH 'WNTR_Demand_Pattern' FUNCTION OUTPUT VARIABLES WITH 'WNTR_Sim_EPANET' FUNCTION INPUT VARIABLES

HYDRAULIC_TIMESTEPS = [ ]
for DT in PAT_DT_LIST:
    HYDRAULIC_TIMESTEPS.append(DT)

PATTERN_TIMESTEPS   = HYDRAULIC_TIMESTEPS
REPORT_TIMESTEPS    = HYDRAULIC_TIMESTEPS

# DEFINE ALTERNATIVES FOR DEMAND PATTERN

ALTERNATIVES    = np.array([ ])
for SP in SIMULATION_PERIODS:
    VALUES = [ ]
    for HTS in HYDRAULIC_TIMESTEPS:
        for N in NOISE:
            VALUES = [SP,HTS,N]
            ALTERNATIVES = np.append(ALTERNATIVES, VALUES)

ALTERNATIVES = ALTERNATIVES.reshape(
    int(len(ALTERNATIVES)/len(VALUES)),
    int(len(VALUES))
    )

ALTERNATIVES   = pd.DataFrame(
    data    = ALTERNATIVES,
    columns = ['SP','HTS','NOISE']
    )

# DEFINE ALTERNATIVES FOR LEAKS (WNTR Simulator)

LEAKS = np.array([ ])
for cd in Cd:
    for a in Aleak:
        for t1 in T1:
            for t2 in T2:
                for junctions in JUNCTIONS:
                    LEAKS = np.append(
                        LEAKS,
                        (cd,a,t1,t2,junctions)
                        )

LEAKS = LEAKS.reshape(
    int(
        len(LEAKS)/len((cd,a,t1,t2,junctions))
        ),
    int(
        len((cd,a,t1,t2,junctions))
        )
    )

LEAKS   = pd.DataFrame(
    data    = LEAKS,
    columns = [
        'Cd',
        'Aleak',
        'T1',
        'T2',
        'Junctions'
        ]
    )

# DEFINE ALTERNATIVES FOR LEAKS (Epanet Simulator)

LEAKS_EPANET = np.array([ ])
for q in Qleak:
    for t1 in T1:
        for t2 in T2:
            for junctions in JUNCTIONS:
                LEAKS_EPANET = np.append(
                    LEAKS_EPANET,
                    (q,t1,t2,junctions)
                    )

LEAKS_EPANET = LEAKS_EPANET.reshape(
    int(
        len(LEAKS_EPANET)/len((q,t1,t2,junctions))
        ),
    int(
        len((q,t1,t2,junctions))
        )
    )

LEAKS_EPANET   = pd.DataFrame(
    data    = LEAKS_EPANET,
    columns = ['Qleak',
               'T1',
               'T2',
               'Junctions'
               ]
    )

# CREATE THE DATAFRAME 'ALT_OVERALL' WITH ALL THE TESTS

df = pd.DataFrame( )
ALT_OVERALL = np.array([ ])

for alt, alt_var in ALTERNATIVES.iterrows():
    for leak, leak_var in LEAKS.iterrows():
        ALT_OVERALL = np.append(
        ALT_OVERALL,
        (alt,leak)
        )
ALT_OVERALL = ALT_OVERALL.reshape(
int(len(ALT_OVERALL)/2),
2
)

i = 0
for alt in ALT_OVERALL:
    one = alt[0]
    two = alt[1]
    df1 = pd.DataFrame(
        ALTERNATIVES.loc[[alt[0]]]
        )
    df1.index = [i]
    df2 = pd.DataFrame(
        LEAKS.loc[[alt[1]]]
        )
    df2.index = [i]
    df3 = pd.concat([df1, df2], axis=1, join='inner')
    df = pd.concat(
        [df, df3],
        axis=0,
        #join='inner'
        )
    i += 1

WNTR_Input_Variables = {
    'ALTERNATIVES':                 ALTERNATIVES,
    'BASE_PAT_FRIDAY_TIME_CHANGE':  BASE_PAT_FRIDAY_TIME_CHANGE,
    'BASE_PAT_FRIDAY_INCREMENT':    BASE_PAT_FRIDAY_INCREMENT,
    'WEEKEND_DELAY_TIME':           WEEKEND_DELAY_TIME,
    'BASE_PAT_MULT':                BASE_PAT_MULT,
    'PAT_DT_LIST':                  PAT_DT_LIST,
    'TEST_FOLDER':                  TEST_FOLDER,
    'FLOW_EXPONENT':                FLOW_EXPONENT,
    'FLOW_MULTIPLIER':              FLOW_MULTIPLIER,
    'DT_MULTIPLIER':                DT_MULTIPLIER,
    'NETWORK_FILE':                 NETWORK_FILE,
    'SCRIPT_FILE':                  SCRIPT_FILE,
    'LEAKS':                        LEAKS,
    'LEAKS_EPANET':                 LEAKS_EPANET,
    'ALT_OVERALL':                  df,
    }

del alt,ALT_OVERALL,alt_var,df,df1,df2,df3,i,leak,leak_var,one,two
del HTS,HYDRAULIC_TIMESTEPS,PATTERN_TIMESTEPS,DT,REPORT_TIMESTEPS,SP,VALUES,N
del BASE_PAT_FRIDAY_TIME_CHANGE, BASE_PAT_FRIDAY_INCREMENT, WEEKEND_DELAY_TIME, BASE_PAT_MULT, NETWORK_FILE, SCRIPT_FILE
del PAT_DT_LIST,NOISE,SIMULATION_PERIODS,TEST_FOLDER,FLOW_EXPONENT,FLOW_MULTIPLIER,DT_MULTIPLIER, ALTERNATIVES
del cd,a,t1,t2,Cd,Aleak,T1,T2,LEAKS,Dleak,LEAKS_EPANET,JUNCTIONS,junctions,q,Qleak

