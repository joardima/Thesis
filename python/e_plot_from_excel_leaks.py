# -*- coding: utf-8 -*-

#%% LOAD THE DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
plt.rcParams["font.family"] = "Times New Roman"

filename        = 'xlsx_files/twoloops_results_MATLAB_HDQ_leaks_testing.xlsx'

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
duration    = df['leak_duration'].unique()
diameter    = df['leak_diameter'].unique()
diameter    = np.sort(diameter)

df_scale    = ['l/s']       # Scale in the dataframe
dq_scale    = ['ls-1']     # scale of the demand and flow features
mult        = [1]              # 1 for input data in l/s / 1000 for input data in m3/s

elements    = [6,6,8]
df_indexes  = [
        10,
        10 + elements[0],
        10 + elements[0] + elements[1],
        10 + elements[0] + elements[1] + elements[2]
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
                                                                                     
#%% CREATE A BAR PLOT WITH THE ERROR MODEL PER LEAK DIAMETER (FIRST APPROACH)

fig, axs = plt.subplots(
        len(sp),
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'row',
        sharey      = 'col',
        )
title   = 'Model RMSE per simulation period (hr) and leak duration (hr)'
title   = title + '\ntime resolution ' + str(dt[0]) + 'min and Noise ' + str(noise[0]) + '%'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Leak duration (hours)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.05, y = 0.5,
    )

legend = [ ]
for s in sp:
    temp = 'sp=' + str(s) + 'hr'
    legend.append(temp)

width = 0.15
mult = [-1.0,0,1.0,]

y_rmse = [ ] 
for s in sp:
    df_temp     = df[
        (df['sp'] == s) &
        (df['leak_duration'] == 0)
        ].sort_values(by=['leak_diameter'])
    y   = np.ones(3)
    y   = y*df_temp['total_rmse'].to_numpy()[0]
    y_rmse.append(y)


i = 0
x_labels = [ ] 
for s in sp:
    df_temp     = df[
        (df['sp'] == s)
        ].sort_values(by=['leak_diameter'])
    x           = np.arange(0,len(df_temp['leak_duration'].unique()))[:-1]
    x_labels_temp    = df_temp['leak_duration'].unique()[1:]
    x_labels.append(x_labels_temp)
    
    j = 0
    for du in x_labels[i]:
        df_temp     = df[
            (df['sp'] == s) & 
            (df['leak_duration'] == du)
            ].sort_values(by=['leak_diameter'])    
        
        sp_name     = 'SP=' + str(s) + ' hr - '
        lamb_name   = 'λ=' + str(df_temp['lambd'].unique()[0])
        text        = sp_name + lamb_name
        
        axs[i].bar(
            x + mult[j]*width,
            df_temp['total_rmse'],
            width,
            label = 'leak diameter ' + str(diameter[1:][j]) + ' in',
            )    
        axs[i].plot(
            [-5,0,5],
            y_rmse[i],
            color       = 'red',
            linewidth   = 1,
            ls='--',
            )
        
        if i == 0:           
            axs[i].text(
            0, 1.1,
            '(The red line correspond to the baseline or metamodel without leak)',
            fontsize    = 14,
            ha          = 'left',
            va          = 'center',
            rotation    = 0,
            transform   = axs[i].transAxes,
            )

        if j == 2:           
            axs[i].text(
            1.01, 0.5,
            text,
            fontsize    = 14,
            ha          = 'left',
            va          = 'center',
            rotation    = 90,
            transform   = axs[i].transAxes,
            )
 
        j += 1                
    i += 1

i = 0
for ax in axs.flatten():
    ax.tick_params(
        axis    = 'both',
        which   ='both',
        length  = 0,
        )
    ax.grid(
        True,
        axis    = 'both',
        which   ='both',
        ls      = "-",
        color   = '0.65'
        )
    ax.set_xticks(
        x, x_labels[i],
        fontsize    = 16,
        )
    ax.legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.35, 0, 0),
        ncol            = 4,
        prop={'size': 12},
        )
    ax.tick_params(
        axis        = 'both',
        which       = 'major',
        labelsize   = 16,
        )    
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(200)
        )
    ax.set_xlim([
        -0.5,
        2.5
        ])
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(25)
        )
#    ax.set_ylim([
#        4,
#        10
#        ])
        
    i += 1

plt.savefig('Plots/bar_sp_all_dt_5min_nu_2_leak_all_sc_ls-1.png')
plt.close()       

#%% CREATE A BAR PLOT WITH THE ERROR MODEL PER LEAK DIAMETER (SECOND APPROACH)

fig, axs = plt.subplots(
        len(sp),
        figsize     = (19, 10),
        dpi         = 100,
#        sharex      = 'col',
#        sharey      = 'col',
        )
title   = 'Model RMSE per simulation period (hr) and leak duration (hr)'
title   = title + '\ntime resolution ' + str(dt[0]) + 'min and Noise ' + str(noise[0]) + '%'

fig.suptitle(
    title,
    fontsize = 20,
    )
fig.supxlabel(
    'Leak duration (hours)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.07, y = 0.5,
    )

x_base = np.arange(-1,len(diameter))
y_base = []
for s in sp:
    df_temp     = df[
        (df['sp'] == s) &
        (df['leak_diameter'] == diameter[0])
        ].sort_values(by=['leak_duration'])
    y_temp  = df_temp['total_rmse'].to_numpy()[0]
    y_temp  = np.repeat(y_temp,len(diameter) + 1)
    y_base.append(y_temp)

width = 0.15
mult = [-1.0,0,1.0,]

i = 0
for s in sp:
    m = 0
    for dia in diameter[1:]:
        label = 'leak diameter ' + str(dia) + ' in'
        df_temp     = df[
            (df['sp'] == s) &
            (df['leak_diameter'] == dia)
            ].sort_values(by=['leak_duration'])
        x = np.arange(0,len(df_temp['leak_duration']))

        sp_name     = 'SP=' + str(s) + ' hr;  '
        lamb_name   = 'λ=' + str(df_temp['lambd'].unique()[0])
        base_rmse   = '\nmodel RMSE\nwithout leak = ' + str(round(y_base[i][0],2))
        text        = sp_name + lamb_name + base_rmse
        
        axs[i].plot(
            x_base,
            y_base[i],
            ls = '--',
            lw = 2,
            color = 'r',
            )
        axs[i].bar(
            x + mult[m]*width,
            df_temp['total_rmse'],
            width,
            label = label,
            )
        axs[i].set_xticks(
            x, df_temp['leak_duration'],
            fontsize    = 16,
            )         
        axs[i].text(
            1.01, 0.5,
            text,
            fontsize    = 14,
            ha          = 'left',
            va          = 'center',
#            rotation    = 90,
            transform   = axs[i].transAxes,
            )
        
        m += 1
    i += 1
    m = 0

axs[0].text(
0, 1.1,
'(The red line corresponds to SINDY metamodel without leak)',
fontsize    = 14,
ha          = 'left',
va          = 'center',
rotation    = 0,
transform   = axs[0].transAxes,
)

i = 0
for ax in axs.flatten():    
    ax.tick_params(
        axis    = 'both',
        which   ='both',
        length  = 0,
        )
    ax.grid(
        True,
        axis    = 'both',
        which   ='both',
        ls      = "-",
        color   = '0.65',
        )
    ax.legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.35, 0, 0),
        ncol            = 4,
        prop={'size': 16},
        )
    ax.tick_params(
        axis        = 'both',
        which       = 'major',
        labelsize   = 16,
        )    
    ax.yaxis.set_major_locator(
        ticker.MultipleLocator(100)
        )
    ax.set_xlim([
        -0.5,
        2.5
        ])
    if df_temp['total_rmse'].max() > 10:
        ax.set_yscale('log')
        
    i += 1

plt.savefig('Plots/bar_sp_all_dt_5min_nu_2_leak_all_sc_ls-1.png')
plt.close()       

#%% CREATE A PLOT WITH MIN AND MAX VALUES PER FEATURE PER EACH COMBINATION SP(sim), dt(res), Nu(noi) FOR ALL LEAK DIAMETERS AND DURATIONS

for sim in sp:

    res = 5; noi = 2;
    leak_dur = np.arange(3) 
    
    x = np.arange(-1,len(diameter))
    yp_min_base = [ ]; yp_max_base = [ ];  yp_min = [ ]; yp_max = [ ]
    yd_min_base = [ ]; yd_max_base = [ ];  yd_min = [ ]; yd_max = [ ]
    yq_min_base = [ ]; yq_max_base = [ ];  yq_min = [ ]; yq_max = [ ]
    
    for s in sp:
        yp_min_sp = [ ]; yp_max_sp = [ ];
        yd_min_sp = [ ]; yd_max_sp = [ ];
        yq_min_sp = [ ]; yq_max_sp = [ ];
        
        p = [ ]; d = [ ]; q = [ ];
        df_temp     = df[
            (df['sp'] == s)
            ].sort_values(by=['total_rmse'])                                                       
    
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
        
        temp = df_temp[
            (df_temp['leak_diameter'] == diameter[0])
            ].sort_values(by=['total_rmse'])
        
        ytemp  = temp['p_rmse_min'].to_numpy()[0]
        yp_min_base.append(ytemp)
        ytemp  = temp['p_rmse_max'].to_numpy()[0]
        yp_max_base.append(ytemp)
        
        ytemp  = temp['d_rmse_min'].to_numpy()[0]
        yd_min_base.append(ytemp)
        ytemp  = temp['d_rmse_max'].to_numpy()[0]
        yd_max_base.append(ytemp)
        
        ytemp  = temp['q_rmse_min'].to_numpy()[0]
        yq_min_base.append(ytemp)
        ytemp  = temp['q_rmse_max'].to_numpy()[0]
        yq_max_base.append(ytemp)
        
        for dur in df_temp['leak_duration'].unique()[1:]:
            yp_min_dur = [ ]; yp_max_dur = [ ]
            yd_min_dur = [ ]; yd_max_dur = [ ]
            yq_min_dur = [ ]; yq_max_dur = [ ]
            temp = df_temp[
                (df_temp['leak_duration'] == dur)
                ].sort_values(by=['total_rmse'])        
            for ldia in temp['leak_diameter'].unique(): 
                temp2 = temp[
                    (temp['leak_diameter'] == ldia)
                    ].sort_values(by=['total_rmse'])
        
                ytemp  = temp2['p_rmse_min'].to_numpy()[0]; yp_min_dur.append(ytemp)
                ytemp  = temp2['p_rmse_max'].to_numpy()[0]; yp_max_dur.append(ytemp)
                ytemp  = temp2['d_rmse_min'].to_numpy()[0]; yd_min_dur.append(ytemp)
                ytemp  = temp2['d_rmse_max'].to_numpy()[0]; yd_max_dur.append(ytemp)
                ytemp  = temp2['q_rmse_min'].to_numpy()[0]; yq_min_dur.append(ytemp)
                ytemp  = temp2['q_rmse_max'].to_numpy()[0]; yq_max_dur.append(ytemp)
                            
            yp_min_sp.append(yp_min_dur); yp_max_sp.append(yp_max_dur)
            yd_min_sp.append(yd_min_dur); yd_max_sp.append(yd_max_dur)
            yq_min_sp.append(yq_min_dur); yq_max_sp.append(yq_max_dur)
        
        yp_min.append(yp_min_sp); yp_max.append(yp_max_sp)
        yd_min.append(yd_min_sp); yd_max.append(yd_max_sp)
        yq_min.append(yq_min_sp); yq_max.append(yq_max_sp)
    
    p = [yp_min_base, yp_max_base, yp_min, yp_max]
    d = [yd_min_base, yd_max_base, yd_min, yd_max]
    q = [yq_min_base, yq_max_base, yq_min, yq_max]
    pdq = [p,d,q]
    
    sim = np.where(sp == sim)[0][0]; res = np.where(dt == res)[0][0]; noi = np.where(noise == noi)[0][0]
    
    sp_name         = 'SP=' + str(sp[sim]) + ' hr'         
    dt_name         = 'dt=' + str(dt[res]) + ' min'
    nu_name         = 'Nu=' + str(noise[noi]) + ' %'
    fw_name         = 'FW=SP'
    temp = df[
        (df['sp'] == sp[sim])
        ]
    lambda_name     = 'λ=' + str(temp['lambd'].unique()[0])
    
    fig, axs = plt.subplots(
            2,
            3,
            figsize     = (19, 10),
            dpi         = 100,
            sharex      = 'all',
    #        sharey      = 'col',
            )
    title   = 'Minimum and maximum RMSE per feature'
    title   = title + '\n' + sp_name + '; ' + dt_name + '; ' + nu_name + '; ' + lambda_name + '; ' + fw_name
    
    fig.suptitle(
        title,
        fontsize = 20,
        )
    fig.supxlabel(
        'Leak duration (hours)',
        fontsize    = 16,
        x = 0.5, y = 0.02,
        )
    fig.supylabel(
        'RMSE',
        fontsize    = 16,
        x = 0.07, y = 0.5,
        )
    
    width           = 0.15
    mult            = [-1.0,0,1.0,]
    label           = ['leak diameter ' + str(x) + ' in' for x in diameter[1:]]
    xtick_labels    = temp['leak_duration'].unique()
    xtick_labels    = np.append(xtick_labels,'0')
    
    for i in range(0,2):    
        for j in range(0,3):
            for dur in leak_dur:
                if i == 0:
                    axs[i,j].bar(
                            leak_dur + mult[dur]*width,
                            pdq[j][2][sim][dur],       
                            width,
                            label = label[dur],
                            )
                    axs[i,j].plot(
                            x,
                            np.repeat(pdq[j][0][sim],len(x)),
                            ls = '--',
                            lw = 2,
                            color = 'r',
                            )
                if i == 1:
                    axs[i,j].bar(
                            leak_dur + mult[dur]*width,
                            pdq[j][3][sim][dur],       
                            width,
                            label = label[dur],
                            )
                    axs[i,j].plot(
                            x,
                            np.repeat(pdq[j][1][sim],len(x)),
                            ls = '--',
                            lw = 2,
                            color = 'r',
    #                        legend = '',
                            )
    ax_title = [
        'Minimum Pressure Head (wmc)','Minimum Demand (l/s)','Minimum Flow (l/s)',
        'Maximum Pressure Head (wmc)','Maximum Demand (l/s)','Maximum Flow (l/s)',
        ]
    i = 0
    for ax in axs.flatten():    
        ax.tick_params(
            axis    = 'both',
            which   ='both',
            labelsize   = 14,
            length  = 0,
            )
        ax.set_xticks(
            x, xtick_labels,
            fontsize    = 16,
            ) 
        ax.set_title(
            ax_title[i],
            fontdict={
                'fontsize': 16,
                }
            )
        ax.grid(
            True,
            axis    = 'y',
            which   ='both',
            ls      = "-",
            color   = '0.65',
            )
        ax.set_xlim([
            -0.5,
            2.5
            ])
        i += 1   
        
#    axs[0,0].set_yscale('log')
#    axs[0,1].set_yscale('log')
#    axs[0,2].set_yscale('log')
#    axs[1,0].set_yscale('log')
#    axs[1,1].set_yscale('log')
#    axs[1,2].set_yscale('log')
    
    
    axs[1,1].legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.35, 0, 0.45),
        ncol            = 4,
        prop = {'size': 14},
        )
    axs[1,2].text(
    0.7, -0.15,
    '(The red line corresponds to\n SINDY metamodel without leak)',
    fontsize    = 14,
    ha          = 'center',
    va          = 'center',
    rotation    = 0,
    transform   = axs[1,2].transAxes,
    )
    
    plt.savefig('Plots/bar_minmax_sp_' + str(sp[sim]) + '_dt_5min_nu_2_leak_all_sc_ls-1.png')
    plt.close()  

#%% RMSE ONLY FOR PRESSURE ELEMENTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import copy
#import re
#from openpyxl import load_workbook
plt.rcParams["font.family"] = "Times New Roman"

filename        = r'xlsx_files/twoloops_results_MATLAB_HDQ_leaks_testing.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='rmse',
    header      = 0,
    )

del filename,#column_names

#unique values of the dataframe
sp              = df['sp'].unique()
dt              = df['dt'].unique()
noise           = df['noise'].unique()
fw              = df['FW'].unique()
lambd           = df['lambd'].unique()
leak            = df['leak'].unique()
leak_duration   = df['leak_duration'].unique()
leak_diameter   = df['leak_diameter'].unique()

df_scale    = ['l/s']       # Scale in the dataframe
dq_scale    = ['ls-1']     # scale of the demand and flow features
mult        = [1]              # 1 for input data in l/s / 1000 for input data in m3/s

elements    = [6,6,8]
df_indexes  = [
        10,
        10 + elements[0],
        10 + elements[0] + elements[1],
        10 + elements[0] + elements[1] + elements[2]
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

leak_zero_rmse = df[
        (df['leak_diameter'] == 0)
        ].sort_values(by=['sp'])
leak_zero_rmse = leak_zero_rmse.iloc[
        :,df_indexes[0]:df_indexes[1]
        ].to_numpy()

for k in range(0,len(leak_diameter[1:])):
    fig, axs = plt.subplots(
            len(sp),
    #        3,
            figsize     = (19, 10),
            dpi         = 100,
    #        sharex      = 'all',
    #        sharey      = 'row',
            )
    
    fig.supxlabel(
        'leak duration (hr)',
        fontsize    = 16,
        x = 0.5, y = 0.02,
        )
    fig.supylabel(
        'Presure RMSE (wmc)',
        fontsize    = 16,
        x = 0.09, y = 0.5,
        )  
    
    title   = 'Pressure features RMSE per all SP, no leak and leak diameter '
    
    width   = 0.1
    mult    = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
    legend  = df.columns[
        df_indexes[0]:df_indexes[1]
        ].to_numpy()

    title = title + str(leak_diameter[k + 1]) + ' in\n'
    title = title + 'dt=5 min; Nu=2%; FW=SP'
    
    fig.suptitle(
        title,
        fontsize = 20,
        )
    
    for i in range(0,len(sp)): 
        
        df_leak_dia_sp     = df[
                (df['leak_diameter'] == leak_diameter[k + 1]) &
                (df['sp'] == sp[i])  
                ].sort_values(by=['leak_duration'])
        
        sp_name         = 'SP=' + str(sp[i]) + ' hr\n'                 
        lambda_name     = 'λ=' + str(df_leak_dia_sp['lambd'].unique()[0])
        subtitle        = sp_name + lambda_name
        
        leak_dur        = df_leak_dia_sp['leak_duration'].unique()
        leak_dur        = np.append(0,leak_dur) 
        x               = np.arange(1,len(leak_dur) + 1)
            
        for e in range(0,len(legend)):
            rmse_zero   = pd.DataFrame([leak_zero_rmse[i][e]])
            y_value     = df_leak_dia_sp[legend[e]]
            y_value     = pd.concat([rmse_zero, y_value]).to_numpy()
            y_value     = y_value.flatten()
                        
            axs[i].bar(
            x + mult[e]*width,
            y_value,
            width,
            label       = legend[e].split('_')[1],
            zorder      = 2,
            )
    
            axs[i].grid(
                True,
                axis    = 'y',
                which   ='both',
                ls      = "-",
                color   = '0.65',
                )
                                       
            axs[i].tick_params(
                axis    = 'both',
                which   ='both',
                labelsize   = 14,
                length  = 0,
                )
            axs[i].set_xticks(
                x, leak_dur,
                fontsize    = 16,
                )
            axs[i].text(
            1.05, 0.5,
            subtitle,
            fontsize    = 14,
            ha          = 'center',
            va          = 'center',
            rotation    = 0,
            transform   = axs[i].transAxes,
            )
            
            if df_leak_dia_sp[legend].max().max() > 10:
                axs[i].set_yscale('log') 
            
    plt.legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.55, 0, 0.45),
        ncol            = len(legend),
        prop = {'size': 14},
        )

    plt.savefig('Plots/bar_P_features_sp_dt_λ_nu_fw=sp_leak_dia=' + str(leak_diameter[k + 1]) + '.png')
    plt.close() 

#%% RMSE ONLY FOR DEMAND ELEMENTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import copy
#import re
#from openpyxl import load_workbook
plt.rcParams["font.family"] = "Times New Roman"

filename        = r'xlsx_files/twoloops_results_MATLAB_HDQ_leaks_testing.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='rmse',
    header      = 0,
    )

del filename,#column_names

#unique values of the dataframe
sp              = df['sp'].unique()
dt              = df['dt'].unique()
noise           = df['noise'].unique()
fw              = df['FW'].unique()
lambd           = df['lambd'].unique()
leak            = df['leak'].unique()
leak_duration   = df['leak_duration'].unique()
leak_diameter   = df['leak_diameter'].unique()

df_scale    = ['l/s']       # Scale in the dataframe
dq_scale    = ['ls-1']     # scale of the demand and flow features
mult        = [1]              # 1 for input data in l/s / 1000 for input data in m3/s

elements    = [6,6,8]
df_indexes  = [
        10,
        10 + elements[0],
        10 + elements[0] + elements[1],
        10 + elements[0] + elements[1] + elements[2]
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

leak_zero_rmse = df[
        (df['leak_diameter'] == 0)
        ].sort_values(by=['sp'])
leak_zero_rmse = leak_zero_rmse.iloc[
        :,df_indexes[1]:df_indexes[2]
        ].to_numpy()

for k in range(0,len(leak_diameter[1:])):
    fig, axs = plt.subplots(
            len(sp),
    #        3,
            figsize     = (19, 10),
            dpi         = 100,
    #        sharex      = 'all',
    #        sharey      = 'row',
            )
    
    fig.supxlabel(
        'leak duration (hr)',
        fontsize    = 16,
        x = 0.5, y = 0.02,
        )
    fig.supylabel(
        'Demand RMSE (l/s)',
        fontsize    = 16,
        x = 0.09, y = 0.5,
        )  
    
    title   = 'Demand features RMSE per all SP, no leak and leak diameter '
    
    width   = 0.1
    mult    = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
    legend  = df.columns[
        df_indexes[1]:df_indexes[2]
        ].to_numpy()

    title = title + str(leak_diameter[k + 1]) + ' in\n'
    title = title + 'dt=5 min; Nu=2%; FW=SP'
    
    fig.suptitle(
        title,
        fontsize = 20,
        )
    
    for i in range(0,len(sp)): 
        
        df_leak_dia_sp     = df[
                (df['leak_diameter'] == leak_diameter[k + 1]) &
                (df['sp'] == sp[i])  
                ].sort_values(by=['leak_duration'])
        
        sp_name         = 'SP=' + str(sp[i]) + ' hr\n'                 
        lambda_name     = 'λ=' + str(df_leak_dia_sp['lambd'].unique()[0])
        subtitle        = sp_name + lambda_name
        
        leak_dur        = df_leak_dia_sp['leak_duration'].unique()
        leak_dur        = np.append(0,leak_dur) 
        x               = np.arange(1,len(leak_dur) + 1)
            
        for e in range(0,len(legend)):
            rmse_zero   = pd.DataFrame([leak_zero_rmse[i][e]])
            y_value     = df_leak_dia_sp[legend[e]]
            y_value     = pd.concat([rmse_zero, y_value]).to_numpy()
            y_value     = y_value.flatten()
                        
            axs[i].bar(
            x + mult[e]*width,
            y_value,
            width,
            label       = legend[e].split('_')[1],
            zorder      = 2,
            )
    
            axs[i].grid(
                True,
                axis    = 'y',
                which   ='both',
                ls      = "-",
                color   = '0.65',
                )
                                       
            axs[i].tick_params(
                axis    = 'both',
                which   ='both',
                labelsize   = 14,
                length  = 0,
                )
            axs[i].set_xticks(
                x, leak_dur,
                fontsize    = 16,
                )
            axs[i].text(
            1.05, 0.5,
            subtitle,
            fontsize    = 14,
            ha          = 'center',
            va          = 'center',
            rotation    = 0,
            transform   = axs[i].transAxes,
            )
            
            if df_leak_dia_sp[legend].max().max() > 10:
                axs[i].set_yscale('log') 
            
    plt.legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.55, 0, 0.45),
        ncol            = len(legend),
        prop = {'size': 14},
        )

    plt.savefig('Plots/bar_D_features_sp_dt_λ_nu_fw=sp_leak_dia=' + str(leak_diameter[k + 1]) + '.png')
    plt.close() 
        

#%% RMSE ONLY FOR FLOW ELEMENTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import copy
#import re
#from openpyxl import load_workbook
plt.rcParams["font.family"] = "Times New Roman"

filename        = r'xlsx_files/twoloops_results_MATLAB_HDQ_leaks_testing.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='rmse',
    header      = 0,
    )

del filename,#column_names

#unique values of the dataframe
sp              = df['sp'].unique()
dt              = df['dt'].unique()
noise           = df['noise'].unique()
fw              = df['FW'].unique()
lambd           = df['lambd'].unique()
leak            = df['leak'].unique()
leak_duration   = df['leak_duration'].unique()
leak_diameter   = df['leak_diameter'].unique()

df_scale    = ['l/s']       # Scale in the dataframe
dq_scale    = ['ls-1']     # scale of the demand and flow features
mult        = [1]              # 1 for input data in l/s / 1000 for input data in m3/s

elements    = [6,6,8]
df_indexes  = [
        10,
        10 + elements[0],
        10 + elements[0] + elements[1],
        10 + elements[0] + elements[1] + elements[2]
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

leak_zero_rmse = df[
        (df['leak_diameter'] == 0)
        ].sort_values(by=['sp'])
leak_zero_rmse = leak_zero_rmse.iloc[
        :,df_indexes[2]:df_indexes[3]
        ].to_numpy()

for k in range(0,len(leak_diameter[1:])):
    fig, axs = plt.subplots(
            len(sp),
    #        3,
            figsize     = (19, 10),
            dpi         = 100,
    #        sharex      = 'all',
    #        sharey      = 'row',
            )
    
    fig.supxlabel(
        'leak duration (hr)',
        fontsize    = 16,
        x = 0.5, y = 0.02,
        )
    fig.supylabel(
        'Flow RMSE (l/s)',
        fontsize    = 16,
        x = 0.09, y = 0.5,
        )  
    
    title   = 'Flow features RMSE per all SP, no leak and leak diameter '
    
    width   = 0.1
    mult    = [-3.5,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.5]
    legend  = df.columns[
        df_indexes[2]:df_indexes[3]
        ].to_numpy()

    title = title + str(leak_diameter[k + 1]) + ' in\n'
    title = title + 'dt=5 min; Nu=2%; FW=SP'
    
    fig.suptitle(
        title,
        fontsize = 20,
        )
    
    for i in range(0,len(sp)): 
        
        df_leak_dia_sp     = df[
                (df['leak_diameter'] == leak_diameter[k + 1]) &
                (df['sp'] == sp[i])  
                ].sort_values(by=['leak_duration'])
        
        sp_name         = 'SP=' + str(sp[i]) + ' hr\n'                 
        lambda_name     = 'λ=' + str(df_leak_dia_sp['lambd'].unique()[0])
        subtitle        = sp_name + lambda_name
        
        leak_dur        = df_leak_dia_sp['leak_duration'].unique()
        leak_dur        = np.append(0,leak_dur) 
        x               = np.arange(1,len(leak_dur) + 1)
            
        for e in range(0,len(legend)):
            rmse_zero   = pd.DataFrame([leak_zero_rmse[i][e]])
            y_value     = df_leak_dia_sp[legend[e]]
            y_value     = pd.concat([rmse_zero, y_value]).to_numpy()
            y_value     = y_value.flatten()
                        
            axs[i].bar(
            x + mult[e]*width,
            y_value,
            width,
            label       = legend[e].split('_')[1],
            zorder      = 2,
            )
    
            axs[i].grid(
                True,
                axis    = 'y',
                which   ='both',
                ls      = "-",
                color   = '0.65',
                )
                                       
            axs[i].tick_params(
                axis    = 'both',
                which   ='both',
                labelsize   = 14,
                length  = 0,
                )
            axs[i].set_xticks(
                x, leak_dur,
                fontsize    = 16,
                )
            axs[i].text(
            1.05, 0.5,
            subtitle,
            fontsize    = 14,
            ha          = 'center',
            va          = 'center',
            rotation    = 0,
            transform   = axs[i].transAxes,
            )
            
            if df_leak_dia_sp[legend].max().max() > 10:
                axs[i].set_yscale('log') 
            
    plt.legend(
        loc             = 'center',
        bbox_to_anchor  = (0.5, -0.55, 0, 0.45),
        ncol            = len(legend),
        prop = {'size': 14},
        )

    plt.savefig('Plots/bar_Q_features_sp_dt_λ_nu_fw=sp_leak_dia=' + str(leak_diameter[k + 1]) + '.png')
    plt.close() 
        
#%% RESULTS TABLES

range_all = []
for s in sp:
    df_temp     = df[
        (df['sp'] == s) &
        (df['leak_diameter'] == diameter[0])
        ].sort_values(by=['leak_duration'])
    base_rmse = df_temp['total_rmse'].to_numpy()[0]
    
    df_temp     = df[
        (df['sp'] == 24) &
        (df['leak_diameter'] != diameter[0])
        ].sort_values(by=['leak_duration'])
    min_base = df_temp['total_rmse'].min(axis=0)
    max_base = df_temp['total_rmse'].max(axis=0)
    
    range_rmse = [
        min_base/base_rmse,
        max_base/base_rmse,
        ]
    
    range_rmse = [ '%.2f' % elem for elem in range_rmse ]
    range_all.append(range_rmse)
    
range_all = np.array(range_all)     