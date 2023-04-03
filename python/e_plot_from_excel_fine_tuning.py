# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams["font.family"] = "Times New Roman"

filename        = 'xlsx_files/twoloops_results_MATLAB_HDQ_finetuning.xlsx'

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name  ='fine_tunning',
    header      = 0,
    )

del filename,#column_names

#unique values of the dataframe
sp          = df['sp'].unique()
dt          = df['dt'].unique()
noise       = df['noise'].unique()
fw          = df['FW'].unique()
lambd       = df['lambd'].unique()

df_scale    = ['l/s', 'm3/s']       # Scale in the dataframe
dq_scale    = ['ls-1', 'm3s-1']     # scale of the demand and flow features
mult        = [1,1000]              # 1 for input data in l/s / 1000 for input data in m3/s

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

lr = [          #Lambda range
    [0.1,1.0],
    [1,10],
    [10,100],
    [100,1000],
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

#%% CREATE A BAR PLOT WITH THE MIN TOTAL RMSE MODEL PER SP AND dt (WITH l/s and m3/s)

fig, axs = plt.subplots(
        1,
        len(df_scale),
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'all',
        sharey      = 'row'
        )
fig.suptitle(
    'Lowest model RMSE per simulation period and dt',
    fontsize = 20,
    )
fig.supxlabel(
    'Simulation period (hours)',
    fontsize    = 16,
    x = 0.5, y = 0.025,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.05, y = 0.5,
    )

legend = [ ]
for delta in dt:
    temp = 'dt=' + str(delta) + 'min'
    legend.append(temp)

i = 0
for dfs in df_scale:
    rmse_min = [ ]
    rmse_max = [ ] 
    for delta in dt:
        rmse_min_dt = [ ]
        rmse_max_dt = [ ]
        for s in sp:       
            df_temp     = df[
                (df['sp'] == s) & 
                (df['dt'] == delta) & 
                (df['d_units'] == dfs)
                ].sort_values(by=['total_rmse'])    
            temp_min    = df_temp['total_rmse'].min(axis=0)
            temp_max    = df_temp['total_rmse'].max(axis=0)
            rmse_min_dt.append(temp_min)
            rmse_max_dt.append(temp_max)
        rmse_min.append(rmse_min_dt)
        rmse_max.append(rmse_max_dt)
    
    width = 0.15
    mult = [-1.5,-0.5,0.5,1.5]
    color = ['r','g','b','y'] 
    
    x = np.arange(len(sp))

    for l in range(0,len(legend)):
        
        axs[i].bar(
        x + mult[l]*width,
        rmse_min[l],
        width,
        label = legend[l],
        color = color[l]
        )
        
        axs[i].set_title(
        str(dfs),
        fontsize = 16,
        )
        
        for an in range(0,len(x)):
            value = round(rmse_min[l][an],2)
            axs[i].annotate(
                    value,
                    (
                        x[an] + mult[l]*width - 0.2,
                        value + 0.4
                        ),
                    )    
    i += 1

for ax in axs.flatten():
            ax.tick_params(
                axis    ='both',
                which   ='major',
                length  = 0
                )
            ax.grid(axis='y')
            ax.set_xticks(x, sp)
            

plt.legend(
    loc             = 'center',
    bbox_to_anchor  = (-0.1, -0.05, 0, 0),
    
#    bbox_to_anchor=(0.75, 1.15),
    ncol = len(legend),
    )
plt.savefig('Plots/bar_sp_all_dt_all_nu_2_sc_all.png')
plt.close()

plt.show()

#%% CREATE A BAR PLOT WITH THE MIN TOTAL RMSE MODEL PER SP AND dt (WITH l/s)

fig, axs = plt.subplots(
#        1,
#        len(df_scale),
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'all',
        sharey      = 'row'
        )
fig.suptitle(
    'Lowest model RMSE per simulation period and dt\nDemand and Flow feature units in l/s',
    fontsize = 20,
    )
fig.supxlabel(
    'Simulation period (hours)',
    fontsize    = 16,
    x = 0.5, y = 0.02,
    )
fig.supylabel(
    'Model RMSE (dimensionless)',
    fontsize    = 16,
    x = 0.075, y = 0.5,
    )

legend = [ ]
for delta in dt:
    temp = 'dt=' + str(delta) + 'min'
    legend.append(temp)

i = 0
for dfs in df_scale[:1]:
    rmse_min = [ ]
    rmse_max = [ ] 
    for delta in dt:
        rmse_min_dt = [ ]
        rmse_max_dt = [ ]
        for s in sp:       
            df_temp     = df[
                (df['sp'] == s) & 
                (df['dt'] == delta) & 
                (df['d_units'] == dfs)
                ].sort_values(by=['total_rmse'])    
            temp_min    = df_temp['total_rmse'].min(axis=0)
            temp_max    = df_temp['total_rmse'].max(axis=0)
            rmse_min_dt.append(temp_min)
            rmse_max_dt.append(temp_max)
        rmse_min.append(rmse_min_dt)
        rmse_max.append(rmse_max_dt)
    
    width = 0.15
    mult = [-1.5,-0.5,0.5,1.5]
    color = ['r','g','b','y'] 
    
    x = np.arange(len(sp))

    for l in range(0,len(legend)):
        
        axs.bar(
        x + mult[l]*width,
        rmse_min[l],
        width,
        label = legend[l],
        color = color[l]
        )
        
#        axs.set_title(
#        str(dfs),
#        fontsize = 16,
#        )
        
    i += 1

    axs.tick_params(
        axis    ='both',
        which   ='major',
        length  = 0,
        labelsize = 16,
        )
    axs.grid(axis='y')
    axs.set_xticks(x, sp)           

#plt.yscale('log')
plt.legend(
    loc             = 'center',
    bbox_to_anchor  = (0.5, -0.06, 0, 0),
    ncol = len(legend),
    fontsize = 14,
    )
plt.savefig('Plots/bar_sp_all_dt_all_nu_2_sc_ls-1.png')
plt.close()

plt.show()   

#%% CREATE A PLOT TYPE dt,SP,LAMBDA PER LAMBDA RANGES - (TOTAL RMSE PER ELEMENT TYPE AND P+Q+D vs 1/λ)

for ds in df_scale:         # 0 for input data in l/s / 1 for input data in m3/s
    if ds == 'm3/s':
        sc = 1
    if ds == 'l/s':
        sc = 0
    
    for k in lr:                                        #Lambda range
        
        for i in range(0,len(dt)):
            dt_name         = '_dt_' + str(dt[i])
            fig, axs = plt.subplots(
                len(elements) + 1,                  # one aditional ax for total rmse
                len(sp),
                figsize     = (19, 10),
                dpi         = 100,
                sharex      = 'all',
                sharey      = 'row'
                )
            
            title = 'Fine tuning of lambda (λ) coefficient for different simulation periods\ndt='
            title =  title + str(dt[i]) + 'min, 1/λ = 1/' + str(k[1]) + ' to 1/' + str(k[0])
            title = title + ', units ' + str(df_scale[sc])
            fig.suptitle(
                title,
                fontsize    = 20
                )
            fig.supxlabel(
                'Lambda (1/λ)',
                fontsize    = 16,
                x = 0.5, y = 0.025,
                )
            fig.supylabel(
                'Total RMSE',
                fontsize    = 16,
                x = 0.05, y = 0.5,
                )
                    
            j = 0
            for s in sp:
                p = [ ]; d = [ ]; q = [ ];
                df_temp     = df[
                    (df['sp'] == s) & 
                    (df['dt'] == dt[i]) & 
                    (df['lambd'] >= k[0]) & 
                    (df['lambd'] <= k[1]) &
                    (df['d_units'] == ds)].sort_values(by=['1/λ'])
                sp_name         = 'sp_all'            
                nu_name         = '_nu_' + str(noise[0])
                fw_name         = '_fw=sp'
                lambda_name     = '_λ_' + str(k[0]) + 'to' + str(k[1])                                                              
                plot_name       =  sp_name + dt_name +  nu_name + fw_name + lambda_name
                del sp_name,nu_name,fw_name,lambda_name
                                                       
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
                
                index       = df_temp.columns.get_loc('lambd')                        
                labels  = [ ]
                for i in df_temp.iloc[:,index]:
                    i = str(i)
                    if i.split('.')[1] != '0':
                        labels.append('1/' + i)
                    else:
                        i = i.split('.')[0]
                        labels.append('1/' + i)
                
                i = 0
            
                for elem in range(0,len(elements)):
            
                    axs[i,j].plot(
                        df_temp['1/λ'],
                        df_temp[y[elem][4]],
                        label = y[elem][4],
                        )
    
                    if i == 2:
                        axs[i + 1,j].plot(
                            df_temp['1/λ'],
                            df_temp['total_rmse'],
                            label = 'Total RMSE',
                            )
    
                    if i == 2 and j == 0:
                        axs[i + 1,j].text(
                            -0.2, 0.5,
                            'Model (Dimensionless)',
                            horizontalalignment     = 'center',
                            verticalalignment       = 'center',
                            rotation                = 90,
                            transform               = axs[i + 1,j].transAxes,
                            fontsize = 14,
                            )
            
                    if j == 0:
                        axs[i,j].text(
                            -0.2, 0.5,
                            y[elem][0],
                            horizontalalignment     = 'center',
                            verticalalignment       = 'center',
                            rotation                = 90,
                            transform               = axs[i,j].transAxes,
                            fontsize = 14,
                            )                                
            
                    if i == 0:
                        axs[i,j].set_title(
                            'sp=' + str(sp[j]) + 'hours',
                            fontsize = 14,
                            )
                                         
                    i += 1                           
                j+=1
                
                for ax in axs.flatten():
                            ax.tick_params(
                                axis    ='both',
                                which   ='major',
                                length  = 0,
                                labelsize = 14,                                
                                )
                            ax.grid(axis='both')
                            ax.set_xticklabels(labels)
                            #ax.xaxis.set_major_formatter(lambda x, pos: '1/' + str(1/x))
                            plt.xticks(np.arange(1/k[1], (1/k[0]) + (1/k[1]), 1/k[1]))                           
                            plt.setp(
                                ax.xaxis.get_majorticklabels(),
                                rotation    = 90,
                                fontsize = 14,
                                )
                
                plt.savefig('Plots/' + plot_name + '_sc_' + dq_scale[sc] + '.png')
            plt.close()
    
#%% CREATE A PLOT TYPE dt,SP,LAMBDA ONE LAMBDA RANGE - (TOTAL RMSE PER ELEMENT TYPE AND ALL vs 1/λ)

from matplotlib.ticker import FuncFormatter

for ds in df_scale:
    if ds == 'm3/s':
        sc = 1000
    if ds == 'l/s':
        sc = 1    

    for k in lr:                                        #Lambda range
        for delta in dt:
            dt_name         = '_dt_' + str(delta)
            fig, axs = plt.subplots(
                len(sp),
                len(elements) + 1,                  #one additional to include total RMSE
                figsize     = (19, 10),
                dpi         = 100,
                sharex      = 'all',
                sharey      = 'col'
                )
            
            title = 'Fine tuning of lambda (λ) coefficient for different simulation periods\n'
            title = title + 'dt=' + str(delta) + 'min, '
            title = title + '1/λ = 1/' + str(max(lr)[1]) + ' to 1/' + str(min(lr)[0])
            title = title + ', units ' + str(ds)
            fig.suptitle(
                title,
                fontsize = 20,
                )
            fig.supxlabel(
                'Lambda (1/λ) - Logarithmic scale',
                fontsize = 16,
                x = 0.5, y = 0.01,
                )
            fig.supylabel(
                'RMSE',
                fontsize    = 16,
                x = 0.05, y = 0.5,
                )
           
            i = 0
            #j = 0
            for s in sp:
                
                sp_name         = 'sp_all'            
                nu_name         = '_nu_' + str(noise[0])
                fw_name         = '_fw=sp'                                                                         
                                  
                df_temp     = df[
                    (df['sp'] == s) & 
                    (df['dt'] == delta) & 
                    (df['lambd'] >= min(lambd)) & 
                    (df['lambd'] <= max(lambd)) &
                    (df['d_units'] == ds)].sort_values(by=['1/λ'])
                
                lambda_name     = '_λ_' + str(min(lambd)) + 'to' + str(max(lambd))
                plot_name       =  sp_name + dt_name +  nu_name + fw_name + lambda_name
                
                p = [ ]
                d = [ ]
                q = [ ]
                
                index = 0
                for col in df_temp.columns:
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
                
                df_temp[y[1][1]]     = df_temp[d].min(axis=1)#*sc
                df_temp[y[1][2]]     = df_temp[d].max(axis=1)#*sc
                df_temp[y[1][3]]     = df_temp[d].mean(axis=1)#*sc
                
                df_temp[y[2][1]]     = df_temp[q].min(axis=1)#*sc
                df_temp[y[2][2]]     = df_temp[q].max(axis=1)#*sc
                df_temp[y[2][3]]     = df_temp[q].mean(axis=1)#*sc
                
                index       = df_temp.columns.get_loc('lambd')
                labels  = [ ]
                for ind in df_temp.iloc[:,index]:
                    ind = str(ind)
                    if ind.split('.')[1] != '0':
                        labels.append('1/' + ind)
                    else:
                        ind = ind.split('.')[0]
                        labels.append('1/' + ind)
                
                for j in range(0,len(y[0][1:]) - 1):             #-1 for not include 'rmse_total'
                    
                    if i == 0:
                        axs[i,j].text(0.5, 1.1,
                        y[j][0] + ' - average',
                        horizontalalignment     = 'center',
                        verticalalignment       = 'center',
                        rotation                = 0,
                        transform               = axs[i,j].transAxes,
                        fontsize    = 16,
                        )
                    if i == 0 and j == 2:
                        axs[i,j + 1].text(0.5, 1.1,
                        'Model (dimensionless)',
                        horizontalalignment     = 'center',
                        verticalalignment       = 'center',
                        rotation                = 0,
                        transform               = axs[i,j + 1].transAxes,
                        fontsize    = 16,
                        )                      
                    if j == 2:
                        axs[i,j + 1].text(
                        1.12, 0.5,
                        'simulation period\n' + str(s) + ' hours',
                        horizontalalignment     = 'center',
                        verticalalignment       = 'center',
                        rotation                = 90,
                        transform               = axs[i,j + 1].transAxes,
                        fontsize    = 14,
                        )
        
                    axs[i,j].plot(
                        df_temp['1/λ'],
                        df_temp[y[j][3]],
                        label = y[j][3],
                        )
    
                    if j == 2:
                        axs[i,j + 1].plot(
                            df_temp['1/λ'],
                            df_temp['total_rmse'],
                            )                    
    
#                    if i == 4 and j == 0:
#                        axs[i,j].legend(
#                            loc             = 'center',
#                            bbox_to_anchor  = (0.5, -0.45, 0, 0),
#                            ncol            = 3,
#                            )
                        
#                    if i == 4 and j == 1:
#                        axs[i,j].legend(
#                            loc             = 'center',
#                            bbox_to_anchor  = (0.5, -0.45, 0, 0),
#                            ncol            = 3,
#                            )
                    
#                    if i == 4 and j == 2:
#                        axs[i,j].legend(
#                            loc             = 'center',
#                            bbox_to_anchor  = (0.5, -0.45, 0, 0),
#                            ncol            = 3
#                            )
#                        axs[i,j + 1].legend(
#                            loc             = 'center',
#                            bbox_to_anchor  = (0.5, -0.45, 0, 0),
#                            ncol            = 3,
#                            )
                            
                    for ax in axs.flatten():
                        ax.tick_params(
                            axis    = 'both',
                            which   ='both',
                            length  = 0,
                            )
                        ax.set_xscale('log')
                        ax.grid(
                            True,
                            axis    = 'both',
                            which   ='both',
                            ls      = "-",
                            color   = '0.65'
                            )
                        for axis in [ax.xaxis]:
                            formatter = FuncFormatter(lambda y, _: '1/{:.16g}'.format(1/y))
                            axis.set_major_formatter(formatter)
                        plt.setp(
                            ax.xaxis.get_majorticklabels(),
                            rotation    = 90,
                            fontsize = 14,
                            )
                        plt.setp(
                            ax.yaxis.get_majorticklabels(),
                            fontsize = 14,
                            )

                    if j < 2:
                        j += 1
                    else:
                        i += 1
                        j = 0
                    
            plt.savefig('Plots/' + plot_name + '_sc_' + str(ds.replace("/s", "s-1")) + '.png')
            plt.close()
                
#%% CREATE A BOX OR WHISKER PLOT                
    
data    = [ ]
texts   = [
    'Pressure head (dimensionless)',
    'Demand (dimensionless)',
    'Flow (dimensionless)',
    ]
               
for ds in df_scale:
    if ds == 'm3/s':
        sc = 1
    if ds == 'l/s':           
        sc = 1

    for delta in dt:                              
        i = 0        

        for s in sp:            
            title       = 'Box and whiskers plot per model\n'
            title = title + 'sp=' + str(s) + ' dt=' + str(delta) + ' nu=' + str(noise[0]) + '%' + ' fw=sp units=' + str(ds)
            fig, axs = plt.subplots(
                    len(elements),
                    1,                  
                    figsize     = (19, 10),
                    dpi         = 100,
                    sharex      = 'all',
                    sharey      = 'col'
                    )     
            fig.supxlabel(
                    'Lambda (1/λ)',
                    fontsize    = 16,
                    x = 0.5, y = 0.025,
                    )
            fig.supylabel(
                    'Normalize RMSE\n(RMSE/Maximum RMSE)',
                    fontsize    = 16,
                    ha         = 'center',
                    x = 0.045, y = 0.5,
                    )
            fig.suptitle(
                title,
                fontsize = 20,
                )
            
            df_temp     = df[
                (df['sp'] == s) & 
                (df['dt'] == delta) & 
                (df['d_units'] == ds)].sort_values(by=['1/λ'])
            
            index       = df_temp.columns.get_loc('lambd')                        
            labels  = [ ]
            for i in df_temp.iloc[:,index]:
                i = str(i)
                if i.split('.')[1] != '0':
                    labels.append('1/' + i)
                else:
                    i = i.split('.')[0]
                    labels.append('1/' + i)
            
            lambda_name     = '_λ_' + str(min(lambd)) + 'to' + str(max(lambd))
            plot_name       =  'sp_' + str(s) + '_dt_' + str(delta) + '_nu_' + str(noise[0]) + '%' + '_fw=sp_units_' + str(ds) + lambda_name
            
            p = [ ]; d = [ ]; q = [ ]
            
            index = 0
            for col in df_temp.columns:
                if index >= df_indexes[0] and index < df_indexes[1]:
                    p.append(col)
                if index >= df_indexes[1] and index < df_indexes[2]:
                    d.append(col)    
                if index >= df_indexes[2] and index < df_indexes[3]:
                    q.append(col)    
                index += 1
            
            df_temp[y[0][2]]    = df_temp[p].max(axis=1)            
            df_temp[y[1][2]]    = df_temp[d].max(axis=1)#*sc
            df_temp[y[2][2]]    = df_temp[q].max(axis=1)#*sc
            
            p = [ ]; d = [ ]; q = [ ];
            for e in range(0,len(elements)):
                for row in range(0,len(df_temp)):   
                    temp = df_temp.iloc[
                        row,
                        df_indexes[e]:df_indexes[e + 1]
                        ]
                    if e == 0:
                        temp = temp/df_temp.iloc[
                            row,
                            df_temp.columns.get_loc('p_rmse_max')
                            ]
                        p.append(temp)
                    if e == 1:
                        temp = temp*sc/df_temp.iloc[
                            row,
                            df_temp.columns.get_loc('d_rmse_max')
                            ]
                        d.append(temp)
                    if e == 2:
                        temp = temp*sc/df_temp.iloc[
                            row,
                            df_temp.columns.get_loc('q_rmse_max')
                            ]
                        q.append(temp)                                           
            
            features    = [p,d,q]                                   
            for ax in range(0,len(texts)):
                axs[ax].boxplot(features[ax])
                axs[ax].text(
                -0.035, 0.5,
                texts[ax],
                ha          = 'center',
                va          = 'center',
                rotation    = 90,
                transform   = axs[ax].transAxes,
                fontsize   = 12,
                )        
            
                if ax == 2:
                    axs[ax].set_xticklabels(
                        np.tile(labels,3),
                        rotation    = 90,
                        fontsize   = 12, 
                        )
            maximum = [ ] 
            for fe in features:
                ma =[ ]
                for f in fe:
                     temp = f.min()
                     ma.append(temp)
                maximum.append(ma)                     
            
            x_max = np.arange(1,len(labels) + 1)            
            y_max = np.ones(len(labels)) + 0.3
            units = ['wmc','l/s','l/s',]
            i = 0
            for ax in axs.flatten():
                y_labels = maximum[i]
                ax.grid(
                    True,
                    axis    = 'y',
                    which   ='both',
                    ls      = "-",
                    color   = '0.65'
                    )
                ax.plot(
                    x_max,
                    y_max,
                    linestyle='',
                    marker = '',
                    )                
                ax.tick_params(
                    axis    = 'both',
                    which   ='both',
                    length  = 0,
                    )
                ax.text(
                    0.99,1.05,
                    '(maximum values in ' + units[i] + ')',
                    ha          = 'right',
                    va          = 'center',
                    transform   = ax.transAxes,
                    fontsize    = 14,
                    )     
                for j in range(len(x_max)):
                    ax.annotate(
                        '(' + str(round(y_labels[j],2)) + ')',
                        (
                            x_max[j] -0.2,
                            y_max[j] - 0.2
                            ),
                        fontsize    = 12,
                        rotation    = 90,
                        )
                i += 1
            plt.savefig('Plots/' + str(plot_name.replace("/s", "s-1")) + '.png')
            plt.close()
            
#%% METADATA MODEL RMSE

best = [ ]

for s in sp:
    for d in dt:

        df_temp     = df[
            (df['sp'] == s) & 
            (df['dt'] == d) & 
            (df['d_units'] == 'l/s')
            ].sort_values(by=['total_rmse'])
        
        l = df_temp[df_temp.total_rmse == df_temp.total_rmse.min()]
        
        df_temp     = df[
            (df['sp'] == s) & 
            (df['dt'] == d) & 
            (df['d_units'] == 'm3/s')
            ].sort_values(by=['total_rmse'])
        
        m = df_temp[df_temp.total_rmse == df_temp.total_rmse.min()]   

        best.append([s,d,l,m])                 
                
for b in best:
    s           = b[0]
    d           = b[1]
    l           = b[2]['total_rmse'].unique()[0]
    l_lambd     = b[2]['lambd'].unique()
    l_lambd     = np.sort(l_lambd)
    m           = b[3]['total_rmse'].unique()[0]
    m_lambd     = b[3]['lambd'].unique()
    obj = type(l_lambd)

    if type(l_lambd) == np.ndarray:
        lla = ''
        for ll in l_lambd:
            lla = lla + ',' + str(ll)
    else:
        lla = l_lambd
    if type(m_lambd) == np.ndarray:
        mla = ''
        for ml in m_lambd:
            mla = mla + ',' + str(ml)
    else:
        lla = l_lambd
            
    print(
        'SP={0:.0f} hr and dt={1:.0f} min'.format(s,d)
        )
    print(
        'l/s, model rmse={0:.3f} and λ={1}'.format(l,lla)
        )
    print(
        'm3/s, model rmse={0:.3f} and λ={1}\n'.format(m,mla)
        )

#%% METADATA MODEL RMSE PER FEATURE (P,D,Q)

best = [ ]
pmin = [ ]; dmin = [ ]; qmin = [ ];
for delta in dt:
    for s in sp:
        p = [ ]; d = [ ]; q = [ ];
        df_temp     = df[
            (df['sp'] == s) & 
            (df['dt'] == delta)].sort_values(by=['total_rmse'])
                                           
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
    
        for units in df_temp['d_units'].unique():
            
            df_temp_units     = df_temp[
                (df_temp['d_units'] == units)
                ].sort_values(by=['total_rmse'])
            
            pmin_temp = df_temp_units[
                df_temp_units.p_rmse_sum == df_temp_units.p_rmse_sum.min()
                ]
            pmin_rmse   = pmin_temp['p_rmse_sum'].unique()[0]
            pmin_l      = pmin_temp['lambd'].unique()
            pmin_l      = np.sort(pmin_l)
            if type(pmin_l) == np.ndarray:
                pml = ''
                for pm in pmin_l:
                    pml = pml + ',' + str(pm)
            else:
                pml = pmin_l
            
            pmin.append(
                [
                    delta,s,pmin_rmse,pml
                    ]
                )
            
            dmin_temp = df_temp_units[
                df_temp_units.d_rmse_sum == df_temp_units.d_rmse_sum.min()
                ]
            dmin_rmse   = dmin_temp['d_rmse_sum'].unique()[0]
            dmin_l      = dmin_temp['lambd'].unique()
            dmin_l      = np.sort(dmin_l)
            if type(dmin_l) == np.ndarray:
                dml = ''
                for dm in dmin_l:
                    dml = dml + ',' + str(dm)
            else:
                dml = dmin_l
            
            dmin.append(
                [
                    delta,s,dmin_rmse,dml
                    ]
                )
            
            qmin_temp = df_temp_units[
                df_temp_units.q_rmse_sum == df_temp_units.q_rmse_sum.min()
                ]
            qmin_rmse   = qmin_temp['q_rmse_sum'].unique()[0]
            qmin_l      = qmin_temp['lambd'].unique()
            qmin_l      = np.sort(qmin_l)
            if type(qmin_l) == np.ndarray:
                qml = ''
                for qm in qmin_l:
                    qml = qml + ',' + str(qm)
            else:
                qml = dmin_l
            
            qmin.append(
                [
                    delta,s,qmin_rmse,qml
                    ]
                )

del pmin_rmse,pmin_l,pmin_temp
del dmin_rmse,dmin_l,dmin_temp
del qmin_rmse,qmin_l,qmin_temp


p_rmse_l      = np.array([ ])
p_lambda_l    = np.array([ ],dtype=object)
p_rmse_m      = np.array([ ])
p_lambda_m    = np.array([ ],dtype=object)

for i in range(0,len(pmin),2):
    p_rmse_l      = np.append(
        p_rmse_l,
        round(pmin[i][2],
              2)
        ,
        )
    p_lambda_l    = np.append(p_lambda_l,
                              pmin[i][3],
                              )
    p_rmse_m      = np.append(
        p_rmse_m,
        round(pmin[i+1][2],2),
        )
    p_lambda_m    = np.append(
        p_lambda_m,
        pmin[i+1][3],
        )

p_table     = np.hstack(
    (np.resize(p_lambda_l, len(p_lambda_l)).reshape(len(p_lambda_l),1),
    np.resize(p_rmse_l, len(p_rmse_l)).reshape(len(p_rmse_l),1))
    )

d_rmse_l      = np.array([ ])
d_lambda_l    = np.array([ ],dtype=object)
d_rmse_m      = np.array([ ])
d_lambda_m    = np.array([ ],dtype=object)

for i in range(0,len(dmin),2):
    d_rmse_l      = np.append(
        d_rmse_l,
        round(dmin[i][2],
              2)
        ,
        )
    d_lambda_l    = np.append(d_lambda_l,
                              dmin[i][3],
                              )
    d_rmse_m      = np.append(
        d_rmse_m,
        round(dmin[i+1][2],2),
        )
    d_lambda_m    = np.append(
        d_lambda_m,
        dmin[i+1][3],
        )

d_table     = np.hstack(
    (np.resize(d_lambda_l, len(d_lambda_l)).reshape(len(d_lambda_l),1),
    np.resize(d_rmse_l, len(d_rmse_l)).reshape(len(d_rmse_l),1))
    )

q_rmse_l      = np.array([ ])
q_lambda_l    = np.array([ ],dtype=object)
q_rmse_m      = np.array([ ])
q_lambda_m    = np.array([ ],dtype=object)

for i in range(0,len(qmin),2):
    q_rmse_l      = np.append(
        q_rmse_l,
        round(qmin[i][2],
              2)
        ,
        )
    q_lambda_l    = np.append(q_lambda_l,
                              qmin[i][3],
                              )
    q_rmse_m      = np.append(
        q_rmse_m,
        round(dmin[i+1][2],2),
        )
    q_lambda_m    = np.append(
        q_lambda_m,
        dmin[i+1][3],
        )

q_table     = np.hstack(
    (np.resize(q_lambda_l, len(q_lambda_l)).reshape(len(q_lambda_l),1),
    np.resize(q_rmse_l, len(q_rmse_l)).reshape(len(q_rmse_l),1))
    )

table = np.hstack(
    (p_table,d_table)
    )

table = np.hstack(
    (table,q_table)
    )

#%% QUERY FOR AN SPECIFIC METAMODEL (SP,dt,λ)

du = 'l/s'

metamodels = [
    [168,5,20],
    [24,15,1],
    [24,30,100],
    [24,60,300],
#    [672,5,10],
    ]

table = np.empty([1,4])
for mm in metamodels:    
    df_temp     = df[
        (df['sp'] == mm[0]) & 
        (df['dt'] == mm[1]) &
        (df['lambd'] == mm[2]) &
        (df['d_units'] == du)
        ].sort_values(by=['total_rmse'])
    
    p = [ ]; d = [ ]; q = [ ]
    index = 0
    for col in df_temp.columns:
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
    
    df_temp[y[1][1]]     = df_temp[d].min(axis=1)#*sc
    df_temp[y[1][2]]     = df_temp[d].max(axis=1)#*sc
    df_temp[y[1][3]]     = df_temp[d].mean(axis=1)#*sc
    
    df_temp[y[2][1]]     = df_temp[q].min(axis=1)#*sc
    df_temp[y[2][2]]     = df_temp[q].max(axis=1)#*sc
    df_temp[y[2][3]]     = df_temp[q].mean(axis=1)#*sc
    
    ranges = [ ]
    ranges.append(str(round(df_temp['p_rmse_min'].to_numpy()[0],2)) + ' - ' + str(round(df_temp['p_rmse_max'].to_numpy()[0],2)))
    ranges.append(str(round(df_temp['d_rmse_min'].to_numpy()[0],2)) + ' - ' + str(round(df_temp['d_rmse_max'].to_numpy()[0],2)))
    ranges.append(str(round(df_temp['q_rmse_min'].to_numpy()[0],2)) + ' - ' + str(round(df_temp['q_rmse_max'].to_numpy()[0],2)))
    ranges.append(str(round(df_temp['total_rmse'].to_numpy()[0],2)))
    ranges = np.array(ranges)
    ranges = ranges.reshape(1, len(ranges))
    table = np.append(
        table,
        ranges,
        )
table = np.reshape(
    table,
    (len(metamodels) + 1,4)
    )
table = table[1:,:]

#%% NUMERICAL AND GRAPHICAL COMPRAISON (SP,dt,λ)

du = 'l/s'

mm_num_sp = [
    [24,5,1],
    [48,5,1],
    [72,5,0.3],
    [168,5,0.2],
    [672,5,1.0],
    ]
mm_num_dt = [
    [168,5,1],
    [24,15,1],
    [48,30,0.3],
    [24,60,1.0],
    ]
mm_gra_sp = [
    [24,5,90],
    [48,5,10],
    [72,5,10],
    [168,5,20],
    [672,5,10],
    ]
mm_gra_dt = [
    [168,5,20],
    [72,15,30],
    [48,30,60],
    [24,60,300],
    ]
models = [
    mm_num_sp,mm_num_dt,
    mm_gra_sp,mm_gra_dt,
    ]

delta   = np.empty([1,7])
s       = np.empty([1,7])
for mod in models:
    for mm in mod:    
        ranges = np.array([ ])
        df_temp     = df[
            (df['sp'] == mm[0]) & 
            (df['dt'] == mm[1]) &
            (df['lambd'] == mm[2]) &
            (df['d_units'] == du)
            ].sort_values(by=['total_rmse'])
        
        p = [ ]; d = [ ]; q = [ ]
        index = 0
        for col in df_temp.columns:
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
        
        df_temp[y[1][1]]     = df_temp[d].min(axis=1)#*sc
        df_temp[y[1][2]]     = df_temp[d].max(axis=1)#*sc
        df_temp[y[1][3]]     = df_temp[d].mean(axis=1)#*sc
        
        df_temp[y[2][1]]     = df_temp[q].min(axis=1)#*sc
        df_temp[y[2][2]]     = df_temp[q].max(axis=1)#*sc
        df_temp[y[2][3]]     = df_temp[q].mean(axis=1)#*sc
        
        ranges = [ ]
        ranges.append(round(df_temp['p_rmse_min'].to_numpy()[0],2))
        ranges.append(round(df_temp['p_rmse_max'].to_numpy()[0],2))
        ranges.append(round(df_temp['d_rmse_min'].to_numpy()[0],2))
        ranges.append(round(df_temp['d_rmse_max'].to_numpy()[0],2))
        ranges.append(round(df_temp['q_rmse_min'].to_numpy()[0],2))
        ranges.append(round(df_temp['q_rmse_max'].to_numpy()[0],2))
        ranges.append(round(df_temp['total_rmse'].to_numpy()[0],2))
    
        ranges = np.array(ranges)
        ranges = ranges.reshape(1, len(ranges))
        
        if len(mod) == len(sp):
            s = np.append(
                s,
                ranges,
                )
        if len(mod) == len(dt):
            delta = np.append(
                delta,
                ranges,
                )    

i = 0; j = 0;            
for mod in models:
    if len(mod) == len(sp):
        i += 1
    if len(mod) == len(dt):
        j += 1
   
s = np.reshape(
    s,
    (len(sp)*i + 1, np.shape(ranges)[1])
    )
s = s[1:,:]

delta = np.reshape(
    delta,
    (len(dt)*j + 1, np.shape(ranges)[1])
    )
delta = delta[1:,:]

fig, axs = plt.subplots(
        2,
        4,
        figsize     = (19, 10),
        dpi         = 100,
        sharex      = 'row',
#        sharey      = 'col'
        )
fig.suptitle(
    'Comparison among selected numerical and graphical models',
    fontsize = 20,
    )

fig.supxlabel(
    'Simulation period (hr)',
    fontsize    = 16,
    x = 0.5, y = 0.06,
    )
fig.supylabel(
    'RMSE',
    fontsize    = 16,
    x = 0.07, y = 0.5,
    )

label_x = [ ]
for mod in models:
    temp = [ ]
    for mm in mod:        
        temp.append(mm[0])
    label_x.append(temp)

width = 0.15

mult = [-1.5,-0.5,0.5,1.5]
legend = [
    'min numerical', 'min graphical',
    'max numerical', 'max graphical',
    ]

x_dt           = np.arange(0,len(dt))
j = 0
m = 0
for col in range(0,np.shape(delta)[1] - 1):
    y_dt_num     = delta[
        :len(dt),
        col
        ]
    y_dt_gra     = delta[
        len(dt):,
        col
        ]
    axs[0,j].bar(
    x_dt + mult[2*m]*width,
    y_dt_num,
    width,
    label = legend[2*m],
    )
    axs[0,j].bar(
    x_dt + mult[2*m + 1]*width,
    y_dt_gra,
    width,
    label = legend[2*m + 1],
    )
    
    if col == 1 or col == 3:
        j += 1
    if m == 0:
        m += 1
    else:
        m = 0
    

y_dt_num     = delta[
    :len(dt),
    6
    ]
y_dt_gra    = delta[
    len(dt):,
    6
    ]
axs[0,3].bar(
x_dt - 0.5*width,
y_dt_num,
width,
label = 'numerical',
)
axs[0,3].bar(
x_dt + 0.5*width,
y_dt_gra,
width,
label = 'graphical',
)


x_sp           = np.arange(0,len(sp))
j = 0
m = 0
for col in range(0,np.shape(s)[1] - 1):
    y_sp_num     = s[
        :len(sp),
        col
        ]
    y_sp_gra     = s[
        len(sp):,
        col
        ]
    axs[1,j].bar(
    x_sp + mult[2*m]*width,
    y_sp_num,
    width,
    label = legend[2*m],
    )
    axs[1,j].bar(
    x_sp + mult[2*m + 1]*width,
    y_sp_gra,
    width,
    label = legend[2*m + 1],
    )
    
    if col == 1 or col == 3:
        j += 1
    if m == 0:
        m += 1
    else:
        m = 0
    

y_sp_num     = s[
    :len(sp),
    6
    ]
y_sp_gra    = s[
    len(sp):,
    6
    ]
axs[1,3].bar(
x_sp - 0.5*width,
y_sp_num,
width,
label = 'numerical',
)
axs[1,3].bar(
x_sp + 0.5*width,
y_sp_gra,
width,
label = 'graphical',
)


#axs[0,0].set_ylim(0, 6)
#axs[0,1].set_ylim(0, 10)
#axs[0,2].set_ylim(0, 30)
#axs[0,3].set_ylim(0, 150)

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

axs[0,1].text(
            1.0, -0.12,
            'Time resolution (min)',
            fontsize    = 16,
            ha          = 'center',
            va          = 'center',
            rotation    = 0,
            transform   = axs[0,1].transAxes,
            )
axs[1,1].legend(
#        loc             = 'best',
    bbox_to_anchor  = (1.8, -0.14, 0, 0),
    ncol = 4,
    fontsize = 14,
    )
axs[1,3].legend(
#        loc             = 'best',
    bbox_to_anchor  = (1.0, -0.14, 0, 0),
    ncol = 4,
    fontsize = 14,
    )
      

for ax in axs.flatten():
    
    ax.tick_params(
    axis        = 'both',
    which       ='both',
    length      = 0,
    labelsize   = 16,
    )
   
    ax.grid(
    True,
    axis    = 'y',
    which   ='both',
    ls      = "-",
    color   = '0.65'
    )


for ax in axs[0,:].flatten():
    ax.set_xticks(
        x_dt, dt,
    )

for ax in axs[1,:].flatten():
    ax.set_xticks(
        x_sp, sp,
    )

plt.savefig('Plots/bar_sp_all_dt_all_nu_2_comparison_sc_ls-1.png')
plt.close() 