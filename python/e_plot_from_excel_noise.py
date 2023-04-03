# -*- coding: utf-8 -*-

#%% LOAD INPUT DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

filename        = 'xlsx_files/twoloops_results_MATLAB_HDQ_noise_testing.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='rmse',
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

index = 0
for col in df.columns:                      # create lists for each type of element, P,D,Q
    if index >= df_indexes[0] and index < df_indexes[1]:
        p.append(col)
    if index >= df_indexes[1] and index < df_indexes[2]:
        d.append(col)    
    if index >= df_indexes[2] and index < df_indexes[3]:
        q.append(col)    
    index += 1
    
df[y[0][1]]    = df[p].min(axis=1)
df[y[0][2]]    = df[p].max(axis=1)
df[y[0][3]]    = df[p].mean(axis=1)
df[y[0][4]]    = df[p].sum(axis=1)

df[y[1][1]]     = df[d].min(axis=1)#*mult[sc]
df[y[1][2]]     = df[d].max(axis=1)#*mult[sc]
df[y[1][3]]     = df[d].mean(axis=1)#*mult[sc]
df[y[1][4]]     = df[d].sum(axis=1)#*mult[sc]

df[y[2][1]]     = df[q].min(axis=1)#*mult[sc]
df[y[2][2]]     = df[q].max(axis=1)#*mult[sc]
df[y[2][3]]     = df[q].mean(axis=1)#*mult[sc]
df[y[2][4]]     = df[q].sum(axis=1)#*mult[sc]       

#%% CREATE A LINE PLOT FOR MODEL RMSE vs FW PER EACH SINDY METAMODEL (1 PLOT)

fig, axs = plt.subplots(
        1,
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'col',
        sharey      = 'row',
        )

title   = 'SINDY metamodel RMSE per different percentage of noise in demand pattern\ndt = 5 min; FW = SP and Nu from '
title   = title + str(noise[0]) + ' to ' + str(noise[-1]) + ' %'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Noise in pattern demand (Nu)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.075, y = 0.5,
    )  

axs_title = [ ]
for i in range(0,len(sp)):
    df_temp     = df[
                    (df['sp'] == sp[i])  
                    ].sort_values(by=['total_rmse'])
    axs_title.append(
        'SP=' + str(sp[i]) + ' hr λ=' + str(df_temp['lambd'].unique()[0])
        )
    x = np.arange(0,len(df_temp['noise']))
    axs.plot(
        x,
        df_temp['total_rmse'],
        label = axs_title[i],
        zorder      = 2,
        linewidth   = 2,
        markersize  = 10,
        marker  = 'o',
        )
    
axs.tick_params(
    axis    = 'both',
    which   ='both',
    length  = 0,
    )
axs.grid(
    True,
    axis    = 'y',
    which   ='both',
    ls      = "-",
    color   = '0.65'
    )
axs.set_xticks(
    x, df_temp['noise'],
    fontsize    = 16,
    rotation    = 0,
    )

axs.tick_params(
     axis        = 'both',
     which       = 'major',
     labelsize   = 16,
     )    

axs.legend(
    loc             = 'best',
#    bbox_to_anchor  = (0.5, -0.35, 0, 0),
    ncol            = 5,
    prop={'size': 14},
    )

plt.savefig('Plots/line_sp_all_dt_5min_nu_all_sc_ls-1.png')
plt.close()

#%% CREATE A BAR PLOT FOR MODEL RMSE vs FW PER EACH SINDY METAMODEL (1 PLOT)

fig, axs = plt.subplots(
        1,
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'col',
        sharey      = 'row',
        )

title   = 'SINDY metamodel RMSE per different percentage of noise in pattern demand \ndt = 5 min; FW = SP and Nu from '
title   = title + str(noise[0]) + ' to ' + str(noise[-1]) + ' %'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Noise in pattern demand (Nu)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.075, y = 0.5,
    )  

width = 0.15
mult = [-2.0,-1.0,0,1.0,2,]

axs_title = [ ]
for i in range(0,len(sp)):
    df_temp     = df[
                    (df['sp'] == sp[i])  
                    ].sort_values(by=['total_rmse'])
    axs_title.append(
        'SP=' + str(sp[i]) + ' hr λ=' + str(df_temp['lambd'].unique()[0])
        )
    x = np.arange(0,len(df_temp['noise']))
    axs.bar(
        x + mult[i]*width,
        df_temp['total_rmse'],
        width,
        label = axs_title[i],
        zorder      = 2,
        linewidth   = 2,
        )
    
axs.tick_params(
    axis    = 'both',
    which   ='both',
    length  = 0,
    )
axs.grid(
    True,
    axis    = 'y',
    which   ='both',
    ls      = "-",
    color   = '0.65'
    )
axs.set_xticks(
    x, df_temp['noise'],
    fontsize    = 16,
    rotation    = 0,
    )

axs.tick_params(
     axis        = 'both',
     which       = 'major',
     labelsize   = 16,
     )    

axs.legend(
    loc             = 'best',
#    bbox_to_anchor  = (0.5, -0.35, 0, 0),
    ncol            = 5,
    prop={'size': 14},
    )

plt.savefig('Plots/bar_sp_all_dt_5min_nu_all_sc_ls-1.png')
plt.close()

#%% CREATE A BAR PLOT WITH MIN AND MAX VALUES RMSE PER FEATURE PER EACH COMBINATION SP(sim), dt(res), Nu(noi) FOR ALL LEAK DIAMETERS AND DURATIONS
#   (6 sUBPLOTS)

fig, axs = plt.subplots(
        2,
        3,
        figsize     = (19, 10),
        dpi         = 100,
#        sharex      = 'col',
        sharey      = 'row',
        )

title   = 'SINDY metamodel RMSE minimum and maximum values per different noise in pattern demand\ndt = 5 min; FW = SP and Nu from '
title   = title + str(noise[0]) + ' to ' + str(noise[-1]) + ' %'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Noise in pattern demand (Nu)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.075, y = 0.5,
    )  

width = 0.15
mult = [-2.0,-1.0,0,1.0,2,]

axs_subtitle = [
    'Pressure head (wmc)',
    'Demand (l/s)',
    'Flow (l/s)',
    ]

df_col_index = [
    [
     'p_rmse_min',
     'd_rmse_min',
     'q_rmse_min',
     ],
    [
     'p_rmse_max',
     'd_rmse_max',
     'q_rmse_max',
     ],
    ]
axs_title =[ ]

for col in range (0,3):
    for row in range (0,2):
    
        series_label = [ ]
        for i in range(0,len(sp)):
            df_temp     = df[
                            (df['sp'] == sp[i])  
                            ].sort_values(by=['total_rmse'])
            axs_title.append(
                'SP=' + str(sp[i]) + ' hr λ=' + str(df_temp['lambd'].unique()[0])
                )
            x = np.arange(0,len(df_temp['noise']))
            axs[row,col].bar(
                x + mult[i]*width,
                df_temp[df_col_index[row][col]],
                width,
                label = axs_title[i],
                zorder      = 2,
                linewidth   = 2,
                )
            if row == 0:
                axs[row,col].set_title(
                axs_subtitle[col],
                fontsize = 16,
                )

axs[0,0].set_ylabel(
    ylabel  = "(Minimum)",
    fontsize    = 16,
    )
axs[1,0].set_ylabel(
    ylabel  = "(Maximum)",
    fontsize    = 16,
    )

for ax in axs.flatten():
    ax.tick_params(
        axis    = 'both',
        which   ='both',
        length  = 0,
        )
    ax.grid(
        True,
        axis    = 'y',
        which   ='both',
        ls      = "-",
        color   = '0.65'
        )
    ax.set_xticks(
        x, df_temp['noise'],
        fontsize    = 16,
        rotation    = 0,
        )
    
    ax.tick_params(
         axis        = 'both',
         which       = 'major',
         labelsize   = 16,
         )    
    
axs[1,2].legend(
    loc             = 'best',
    bbox_to_anchor  = (0.5, -0.05, 0, 0),
    ncol            = 5,
    prop={'size': 14},
    )

plt.savefig('Plots/bar_pervariable_minmax_sp_all_dt_5min_nu_all_sc_ls-1.png')
plt.close()