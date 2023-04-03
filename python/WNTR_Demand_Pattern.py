# -*- coding: utf-8 -*-

"""
Function for daily demand pattern interpolation based on BASE_PAT_MULT (average day pattern demand)

---- ASSUMPTIONS ----
    Pattern demand on friday increases BASE_PAT_FRIDAY_INCREMENT after BASE_PAT_FRIDAY_TIME_CHANGE
    Pattern demand on saturday and sunday delays 1 hour in comparisson with BASE_PAT_MULT
    Initial and final multipliers in each demand pattern  should be the same
    If a EPS for and entire week is required, check initial and final demand pattern multiplier
    noise.......

---- INPUT VARIABLES ----
BASE_PAT_MULT:                  Multipliers of the base/input demand pattern for a day (1D numpy array, unitless).
                                If the base/input demand pattern is every hour then len(BASE_PAT_MULT) should be 25
                                If the base/input demand pattern is every half hour then len(BASE_PAT_MULT) should be 49
BASE_PAT_FRIDAY_TIME_CHANGE:    Hour at which the friday demand pattern changes in comparison with average day demand pattern (integer, in min)
BASE_PAT_FRIDAY_INCREMENT:      Increment on the base pattern demand on friday nights (float, in percentage)
PAT_DT_LIST:                    DTs of the average day demand patterns (list, in min)
WEEKEND_DELAY_TIME:             Delay time for weekend demand pattern (integer, in hour)
NOISE:                          Noise or standard deviation to be included in the multiplier of the base pattern demand (Integer, Percentage)

---- OUTPUT VARIABLES ----
INITIAL_INPUTS:                 List with the input variables used               
AVEDAY_PAT_TIME:                List with timesteps for demand patterns computed 
AVEDAY_PAT_MULT:                List with the multipliers for the average day demand pattern
FRIDAY_PAT_MULT:                List with the multipliers for the friday demand pattern
WEEKEND_PAT_MULT:               List with the multipliers for the weekend demand pattern
RETURN_RESULTS:                 Python list with INITIAL_INPUTS, AVEDAY_PAT_TIME, AVEDAY_PAT_MULT, FRIDAY_PAT_MULT, WEEKEND_PAT_MULT
"""

def WNTR_Demand_Pattern(
        BASE_PAT_MULT,
        BASE_PAT_FRIDAY_TIME_CHANGE,
        BASE_PAT_FRIDAY_INCREMENT,
        WEEKEND_DELAY_TIME,
        ALTERNATIVES
        ):
    
# IMPORT LIBRARIES
    
    import numpy as np
    import copy
    import time
    from scipy.interpolate import interp1d

    start_time = time.time()
    
# DEFINE FUNCTION VARIABLES

    BASE_PAT_MULT                   = BASE_PAT_MULT
    BASE_PAT_FRIDAY_TIME_CHANGE     = BASE_PAT_FRIDAY_TIME_CHANGE
    BASE_PAT_FRIDAY_INCREMENT       = BASE_PAT_FRIDAY_INCREMENT
    WEEKEND_DELAY_TIME              = WEEKEND_DELAY_TIME
    ALT                             = ALTERNATIVES

# DEFINE VARIABLES
    
    DICT_RESULTS        = np.array([ ])
    BASE_PAT_DT         = 60/((len(BASE_PAT_MULT) - 1)/24)
    BASE_PAT_TS         = int(24*60/BASE_PAT_DT + 1)                         #Timesteps per day of the base/input demand pattern (number per day)
    BASE_PAT_TIME       = np.linspace(0,24*60,BASE_PAT_TS)                 #1440 = 24 hours/day*60 min/hour

    ALT = ALT.reset_index()             #make sure indexes pair with number of rows
    pattern_number = 0
    for index, row in ALT.iterrows():
        start_pattern_time = time.time()
#        PARTIAL_RESULTS = np.array([ ])
        SP              = row['SP']
        AVEDAY_PAT_DT   = row['HTS']            #DT of the pattern to compute
        NOISE           = row['NOISE']
        RATIO           = int(SP/24)                #Number of days in the simulation period
        
# INTERPOLATE BASE PATTERN FOR EACH DT
        
        BP_MULT_INT             = np.array([ ])
        BP_TS_INT               = int(24*60/AVEDAY_PAT_DT + 1)
        BP_TIME_INT             = np.linspace(0,24*60,BP_TS_INT)
        BPM_INT                 = interp1d(BASE_PAT_TIME, BASE_PAT_MULT, 'cubic')
        for TIME in BP_TIME_INT:
            BP_MULT_INT = np.append(
                BP_MULT_INT,
                BPM_INT(TIME)
                )
        i = 1
        A = BP_MULT_INT[1:]
        while i < RATIO:
            BP_MULT_INT     = np.append(
                BP_MULT_INT,
                A
                )
            i += 1
        
# COMPUTE AVERAGE DAY PATTERN (NEW)
        
        AVEDAY_PAT_TS           = int(SP*60/AVEDAY_PAT_DT + 1)                              #Timesteps per day of the average day demand pattern (number per day)
        AVEDAY_PAT_TIME         = np.linspace(0,SP*60,AVEDAY_PAT_TS)                        #1440 = 24 hours/day*60 min/hour

        ADPT_DAY                = int(24*60/AVEDAY_PAT_DT + 1)                              #AVEDAY_PAT_TS per one day
        ADPT_TIME               = np.linspace(0,24*60,ADPT_DAY)                        #AVEDAY_PAT_TS per one day

#ADD NOISE TO AVERAGE DAY BASE PATTERN
    
        np.random.seed(0)
        AVEDAY_PAT_MULT_NOISE   = (1 + (NOISE/100)*np.random.randn(1, len(AVEDAY_PAT_TIME)))
        AVEDAY_PAT_MULT_NOISE   = AVEDAY_PAT_MULT_NOISE.flatten()
        AVEDAY_PAT_MULT         = BP_MULT_INT*AVEDAY_PAT_MULT_NOISE
        AVEDAY_PAT_MULT[-1]     = AVEDAY_PAT_MULT[0]

# CREATE A PATTERN FOR LEAKAGE

        AVEDAY_LEAK_PAT     = copy.deepcopy(AVEDAY_PAT_TIME)
        AVEDAY_LEAK_PAT[AVEDAY_LEAK_PAT != 1] = 1
        ONEDAY_LEAK_PAT     = copy.deepcopy(BP_TIME_INT)
        ONEDAY_LEAK_PAT[ONEDAY_LEAK_PAT != 1] = 1

# COMPUTE FRIDAY PATTERN

        TIME                = int(BASE_PAT_FRIDAY_TIME_CHANGE*60/AVEDAY_PAT_DT)
        FRIDAY_PAT_MULT     = copy.deepcopy(AVEDAY_PAT_MULT[:TIME])
        while TIME <= 24*60/AVEDAY_PAT_DT:
            FRIDAY_PAT_MULT     = np.append(
                FRIDAY_PAT_MULT,
                AVEDAY_PAT_MULT[TIME]*(1 + BASE_PAT_FRIDAY_INCREMENT/100)
                )
            TIME                += 1
        FRIDAY_PAT_MULT[-1]     = FRIDAY_PAT_MULT[0]        

# COMPUTE WEEKEND PATTERN
       
        WEEKEND_PAT_MULT        = FRIDAY_PAT_MULT
        WEEKEND_PAT_MULT_TEMP   = FRIDAY_PAT_MULT[-int(WEEKEND_DELAY_TIME*60/AVEDAY_PAT_DT):]
        WEEKEND_PAT_MULT        = np.insert(WEEKEND_PAT_MULT,0,WEEKEND_PAT_MULT_TEMP)
        WDT = 0
        while WDT < WEEKEND_DELAY_TIME*60/AVEDAY_PAT_DT:
            WEEKEND_PAT_MULT    = np.delete(WEEKEND_PAT_MULT, -1, 0)
            WDT                 += 1
        WEEKEND_PAT_MULT_AVE    = (WEEKEND_PAT_MULT[0] + WEEKEND_PAT_MULT[-1])/2
        WEEKEND_PAT_MULT[0]     = WEEKEND_PAT_MULT_AVE
        WEEKEND_PAT_MULT[-1]    = WEEKEND_PAT_MULT_AVE
        
#FINAL RESULTS
               
        DICT_PARTIAL = {
            'BASE_PAT_TIME':                BP_TIME_INT,
            'BASE_PAT_MULT_INT':            BP_MULT_INT,
            'BASE_PAT_FRIDAY_INCREMENT':    BASE_PAT_FRIDAY_INCREMENT,
            'BASE_PAT_FRIDAY_TIME_CHANGE':  BASE_PAT_FRIDAY_TIME_CHANGE,
            'WEEKEND_DELAY_TIME':           WEEKEND_DELAY_TIME,
            'NOISE':                        NOISE,
            'AVEDAY_PAT_DT':                AVEDAY_PAT_DT,
            'AVEDAY_PAT_TIME':              AVEDAY_PAT_TIME,
            'AVEDAY_PAT_TIME_DAY':          ADPT_TIME,
            'AVEDAY_PAT_MULT':              AVEDAY_PAT_MULT,
            'FRIDAY_PAT_MULT':              FRIDAY_PAT_MULT,
            'WEEKEND_PAT_MULT':             WEEKEND_PAT_MULT,
            'AVEDAY_LEAK_PAT':              AVEDAY_LEAK_PAT,
            'ONEDAY_LEAK_PAT':              ONEDAY_LEAK_PAT
            }
        
        DICT_RESULTS = np.append(
            DICT_RESULTS,
            DICT_PARTIAL
            )
        
        end_pattern_time = time.time()
        
        pattern_time = (end_pattern_time - start_pattern_time)

        print('performed simulation {0:.0f}'.format(pattern_number))
        print('model simulation time {0:.3f}'.format(pattern_time))
        
        pattern_number += 1
    
    end_time = time.time()
    
    run_time = (end_time - start_time)/60
    
    print('total simulated models {0:.2f}'.format(pattern_number))
    print('total simulation time {0:.2f}'.format(run_time))
    
    #del start_time,pattern_number,start_pattern_time,end_pattern_time,pattern_time,end_time,run_time  
    
    return DICT_RESULTS
