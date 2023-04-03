# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.family"] = "Times New Roman"


filename        = 'xlsx_files/twoloops_results_MATLAB_HDQ_testing.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='testing',
    header      = 0,
    )

del filename,#column_names

#unique values of the dataframe
sp          = df['sp'].unique()
dt          = df['dt'].unique()
noise       = df['noise'].unique()
fw          = df['FW'].unique()
lambd       = df['lambd'].unique()

df_scale    = ['l/s']       # Scale in the dataframe
dq_scale    = ['ls-1']     # scale of the demand and flow features
mult        = [1]              # 1 for input data in l/s / 1000 for input data in m3/s

elements    = [6,6,8]
df_indexes  = [
        5,
        5 + elements[0],
        5 + elements[0] + elements[1],
        5 + elements[0] + elements[1] + elements[2]
        ]

y = [
    ['Pressure Head (wmc)','p_rmse_min','p_rmse_max','p_rmse_mean','p_rmse_sum',],
    ['Demand (l/s)','d_rmse_min','d_rmse_max','d_rmse_mean','d_rmse_sum',],
    ['Flow (l/s)','q_rmse_min','q_rmse_max','q_rmse_mean','q_rmse_sum',],
    ]

df['1/λ'] = 1/df['lambd']

# create lists for each type of element, P,D,Q
p = [ ]; d = [ ]; q = [ ];
index = 0
for col in df.columns:                      
    if index >= df_indexes[0] and index < df_indexes[1]:
        p.append(col)
    if index >= df_indexes[1] and index < df_indexes[2]:
        d.append(col)
    if index >= df_indexes[2] and index < df_indexes[3]:
        q.append(col)    
    index += 1

for index, row in df.iterrows():    
    if row['d_units'] == 'm3/s':
        df.iloc[index,df_indexes[1]:df_indexes[2]] = df.iloc[index,df_indexes[1]:df_indexes[2]]*1000
        df.iloc[index,df_indexes[2]:df_indexes[3]] = df.iloc[index,df_indexes[2]:df_indexes[3]]*1000


df['total_rmse'] = df.iloc[
    :,
    df_indexes[0]:df_indexes[-1]
    ].sum(axis = 1)                                                                                  
                                                                                     
#%% 

width = 0.25
mult = [-0.5,0.5,]

fig, axs = plt.subplots(
        len(sp),
        4,
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'col',
#        sharey      = 'row'
        )

title = 'Results of testing SINDY metamodels\n(10 testing datasets)'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Testing dataset (No.)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'RMSE',
    fontsize    = 16,
    x = 0.1, y = 0.5,
    )

for i in range(0,len(sp)):

    p = [ ]; d = [ ]; q = [ ];
    df_temp     = df[
        (df['sp'] == sp[i])].sort_values(by=['CV_dataset'])

    sp_name         = 'sp_all'            
    dt_name         = str(df_temp['dt'].to_numpy()[0])
    nu_name         = '_nu_' + str(noise[0])
    fw_name         = '_fw=sp'
    lambda_name     = str(lambd)                                                            

    index = 0
    for col in df_temp.columns:                      # create lists for each type of element, P,D,Q
        if index >= df_indexes[0] and index < df_indexes[1]:
            p.append(col)
        if index >= df_indexes[1] and index < df_indexes[2]:
            d.append(col)    
        if index >= df_indexes[2] and index < df_indexes[3]:
            q.append(col)    
        index += 1
        
    df_temp[y[0][1]]    = df_temp[p].min(axis=1)
    df_temp[y[0][2]]    = df_temp[p].max(axis=1)
    df_temp[y[0][3]]    = df_temp[p].mean(axis=1)
    df_temp[y[0][4]]    = df_temp[p].sum(axis=1)
    
    df_temp[y[1][1]]     = df_temp[d].min(axis=1)#*mult[sc]
    df_temp[y[1][2]]     = df_temp[d].max(axis=1)#*mult[sc]
    df_temp[y[1][3]]     = df_temp[d].mean(axis=1)#*mult[sc]
    df_temp[y[1][4]]     = df_temp[d].sum(axis=1)#*mult[sc]
    
    df_temp[y[2][1]]     = df_temp[q].min(axis=1)#*mult[sc]
    df_temp[y[2][2]]     = df_temp[q].max(axis=1)#*mult[sc]
    df_temp[y[2][3]]     = df_temp[q].mean(axis=1)#*mult[sc]
    df_temp[y[2][4]]     = df_temp[q].sum(axis=1)#*mult[sc]
    
    x_labels = pd.unique(df_temp['CV_dataset'])[1:]
    
    axs[i,0].bar(
        df_temp['CV_dataset'][1:] + mult[0]*width,
        df_temp['p_rmse_min'][1:],
        width,
        label = 'Testing Min',
        )
    axs[i,0].bar(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        df_temp['p_rmse_max'][1:],
        width,
        label = 'Testing Max',
        )
    axs[i,0].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['p_rmse_min'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel min',
        )
    axs[i,0].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['p_rmse_max'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel max',
        )
    
    axs[i,1].bar(
        df_temp['CV_dataset'][1:] + mult[0]*width,
        df_temp['d_rmse_min'][1:],
        width,
        label = 'Testing Min',
        )
    axs[i,1].bar(
        df_temp['CV_dataset'][1:] + mult[1]*width,
        df_temp['d_rmse_max'][1:],
        width,
        label = 'Testing Max',
        )
    axs[i,1].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['d_rmse_min'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel min',
        )
    axs[i,1].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['d_rmse_max'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel max',
        )
      
    axs[i,2].bar(
        df_temp['CV_dataset'][1:] + mult[0]*width,
        df_temp['q_rmse_min'][1:],
        width,
        label = 'Testing Min',
        )
    axs[i,2].bar(
        df_temp['CV_dataset'][1:] + mult[1]*width,
        df_temp['q_rmse_max'][1:],
        width,
        label = 'Testing Max',
        )
    axs[i,2].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['q_rmse_min'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel min',
        )
    axs[i,2].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['q_rmse_max'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'Trained SINDY metamodel max',
        )
       
    axs[i,3].bar(
        df_temp['CV_dataset'][1:],
        df_temp['total_rmse'][1:],
        width,
        label = 'Testing',
        )
    axs[i,3].plot(
        df_temp['CV_dataset'][1:]  + mult[1]*width,
        [df_temp['total_rmse'].to_numpy()[0]]*10,
        linestyle = '--',
        label = 'trained SINDY metamodel',
        )
    axs[i,3].set_ylim(
        0.9*df_temp['total_rmse'][1:].min(),
        1.1*df_temp['total_rmse'][1:].max(),
        )
    
    mm = 'SP=' + str(pd.unique(df_temp['sp'])[0]) + ' hr; \n'
    mm = mm + 'dt=' + str(pd.unique(df_temp['dt'])[0]) + ' min;\n'
    mm = mm + 'Nu=' + str(pd.unique(df_temp['noise'])[0]) + ' %; '
    mm = mm + 'λ=' + str(pd.unique(df_temp['lambd'])[0]) 
    axs[i,3].text(
                1.3, 0.5,
                mm,
                fontsize    = 16,
                ha          = 'center',
                va          = 'center',
                rotation    = 0,
                transform   = axs[i,3].transAxes,
                )

axs[0,0].set_title(
'Pressure head (wmc)',
fontdict = {
    'fontsize': 16,
    }
)

axs[0,1].set_title(
'Demand (l/s)',
fontdict = {
    'fontsize': 16,
    }
)
axs[0,2].set_title(
'Flow (l/s)',
fontdict = {
    'fontsize': 16,
    }
)
axs[0,3].set_title(
'Model (dimensionless)',
fontdict = {
    'fontsize': 16,
    }
)

axs[4,1].legend(
    loc             = 'center',
    bbox_to_anchor=(0.5, 0, 0, -0.7),
    ncol = 4,
    fontsize = 12,
    )
axs[4,3].legend(
    loc             = 'center',
    bbox_to_anchor=(0.5, 0, 0, -0.7),
    ncol = 3,
    fontsize = 12,
    )

for ax in axs.flatten():
    
    ax.tick_params(
    axis        = 'both',
    which       ='major',
    length      = 0,
    labelsize   = 14,
    )
   
    ax.grid(
    True,
    axis    = 'y',
    which   ='both',
    ls      = "-",
    color   = '0.65'
    )
    

for ax in axs[1,:].flatten():
    ax.set_xticks(
        x_labels, x_labels,
    )
    
#    ax.xaxis.get_major_ticks()[-1].label.set_visible(False)
      
plt.savefig('Plots/bar_sp_all_dt_all_nu_2_sc_ls-1_testing.png')
plt.close()

#%% VARIATION OF RMSE WITH TESTING DATASET

metadata = [ ]

for i in range(0,len(sp)):

    p = [ ]; d = [ ]; q = [ ];
    df_temp     = df[
        (df['sp'] == sp[i])].sort_values(by=['CV_dataset'])

    sp_name         = 'sp_all'            
    dt_name         = str(df_temp['dt'].to_numpy()[0])
    nu_name         = '_nu_' + str(noise[0])
    fw_name         = '_fw=sp'
    lambda_name     = str(lambd)                                                            

    index = 0
    for col in df_temp.columns:                      # create lists for each type of element, P,D,Q
        if index >= df_indexes[0] and index < df_indexes[1]:
            p.append(col)
        if index >= df_indexes[1] and index < df_indexes[2]:
            d.append(col)    
        if index >= df_indexes[2] and index < df_indexes[3]:
            q.append(col)    
        index += 1
        
    df_temp[y[0][1]]    = df_temp[p].min(axis=1)
    df_temp[y[0][2]]    = df_temp[p].max(axis=1)
    df_temp[y[0][3]]    = df_temp[p].mean(axis=1)
    df_temp[y[0][4]]    = df_temp[p].sum(axis=1)
    
    df_temp[y[1][1]]     = df_temp[d].min(axis=1)#*mult[sc]
    df_temp[y[1][2]]     = df_temp[d].max(axis=1)#*mult[sc]
    df_temp[y[1][3]]     = df_temp[d].mean(axis=1)#*mult[sc]
    df_temp[y[1][4]]     = df_temp[d].sum(axis=1)#*mult[sc]
    
    df_temp[y[2][1]]     = df_temp[q].min(axis=1)#*mult[sc]
    df_temp[y[2][2]]     = df_temp[q].max(axis=1)#*mult[sc]
    df_temp[y[2][3]]     = df_temp[q].mean(axis=1)#*mult[sc]
    df_temp[y[2][4]]     = df_temp[q].sum(axis=1)#*mult[sc]
    
        
    p_rmse_max          = df_temp['p_rmse_sum'][1:].max(axis=0)
    p_rmse_mm           = df_temp['p_rmse_sum'].iloc[0]
    diff_p_rmse         = p_rmse_max - p_rmse_mm
    
    d_rmse_max          = df_temp['d_rmse_sum'][1:].max(axis=0)
    d_rmse_mm           = df_temp['d_rmse_sum'].iloc[0]
    diff_d_rmse         = d_rmse_max - d_rmse_mm
    
    q_rmse_max          = df_temp['q_rmse_sum'][1:].max(axis=0)
    q_rmse_mm           = df_temp['q_rmse_sum'].iloc[0]
    diff_q_rmse         = q_rmse_max - q_rmse_mm
    
    model_rmse_max      = df_temp['total_rmse'][1:].max(axis=0)
    model_rmse_mm       = df_temp['total_rmse'].iloc[0]
    diff_model_rmse     = model_rmse_max - model_rmse_mm
    
    temp = [diff_p_rmse, diff_q_rmse, diff_d_rmse, diff_model_rmse]
    
    metadata.append(temp)
    
    del temp,diff_p_rmse,diff_q_rmse,diff_d_rmse,diff_model_rmse
    del p_rmse_max,p_rmse_mm,d_rmse_max,d_rmse_mm,q_rmse_max,q_rmse_mm

x_temp = range(0,len(sp))
    
p = [ ]; d = [ ]; q = [ ]; model = [ ];
for ele in metadata:
    p.append(ele[0])
    d.append(ele[1])
    q.append(ele[2])
    model.append(ele[3])
    
title = 'Variation of RMSE per SINDY metamodels\n(10 testing datasets)'

width = 0.2
mult = [-1.5,-0.5,0.5,1.5]

fig, axs = plt.subplots(
        1,
        1,
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'col',
#        sharey      = 'row'
        )

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Simulation period (hr)',
    fontsize    = 16,
    x = 0.5, y = 0.01,
    )
fig.supylabel(
    'RMSE variation\n(Max RMSE testing - RMSE training)',
    fontsize    = 16,
    x   = 0.07, y = 0.5,
    ha  = 'center',
    va  = 'center',
    )
axs.bar(
    [x + mult[0]*width for x in x_temp],
    p ,
    width,
    label = 'Pressure head Total (wmc)',
    )
axs.bar(
    [x + mult[1]*width for x in x_temp],
    d,
    width,
    label = 'Demand Total (l/s)',
    )
axs.bar(
    [x + mult[2]*width for x in x_temp],
    q,
    width,
    label = 'Demand Total (l/s)',
    )
axs.bar(
    [x + mult[3]*width for x in x_temp],
    model,
    width,
    label = 'Model (dimensionless)',
    )

fig.legend(
    loc             ='upper center',
    bbox_to_anchor  = (0.5, 0.08, 0, 0),
    ncol            = 4,
    fontsize        = 14,
    )

axs.tick_params(
    axis        = 'both',
    which       ='major',
    length      = 0,
    labelsize   = 14,
    )
 
axs.grid(
    True,
    axis    = 'y',
    which   ='both',
    ls      = "-",
    color   = '0.65'
    )
    
axs.set_xticks(
        x_temp, sp,
    )

plt.savefig('Plots/bar_sp_all_dt_all_nu_2_sc_ls-1_totalvariation.png')
plt.close()