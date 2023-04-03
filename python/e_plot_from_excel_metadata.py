# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

filename        = 'xlsx_files/twoloops_results_MATLAB_HDQ.xlsx'

column_names = [
    'sp','dt','noise','FW','Lambda',
    'n_coeff_min','n_coeff_max','abs_val_coeff_min','abs_val_coeff_max',
    'rmse_h_min','rmse_h_max','rmse_d_min','rmse_d_max','rmse_q_min','rmse_q_max',
    'n_coeff_h_min','n_coeff_h_max','n_coeff_d_min','n_coeff_d_max','n_coeff_q_min','n_coeff_q_max',
    ]

# read Excel file into DataFrame
df = pd.read_excel(
    filename,
    sheet_name='HDQ',
    header=None,
    names=column_names,
    skiprows=[0,1,2,],
    )

del filename,column_names

#unique values of the dataframe
sp      = df['sp'].unique()
dt      = df['dt'].unique()
noise   = df['noise'].unique()

#%% Create subplots for (sp,dt,noise)

dq_sc       = 1000
dq_add      = 5
coeff_add   = 2

i = 0
for s in sp:
    sp_temp         = df[(df['sp'] == s)]
    h_lim           = round(max(sp_temp['rmse_h_max'])) + 2
    d_lim           = round((max(sp_temp['rmse_d_max']) + (dq_add/dq_sc))*dq_sc)
    q_lim           = round((max(sp_temp['rmse_q_max']) + (dq_add/dq_sc))*dq_sc)
    coeff_lim_min_h   = round(
       min(
           sp_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           sp_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           sp_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           sp_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           sp_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           sp_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    
    j = 0
    fig, axs = plt.subplots(3,len(dt),figsize=(10, 5),sharex="all",sharey='row')
    title = 'Minimum and maximum RMSE and number of coefficients\n(SP=' + str(s) + 'hr)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Noise in the pattern (%)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)
    
    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[0,0].transAxes,
                  )
    
    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[1,0].transAxes,
                  )
    
    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[2,0].transAxes,
                  )
    
    for d in dt:        
        sp_dt = df[(df['sp'] == s) & (df['dt'] == d)].sort_values(by=['noise'])              
        
        axs[0,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(noise), right = max(noise))
        axs[0,j].set_xticks(np.arange(0, max(noise) + 1, max(noise)/len(noise)))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('dt=' + str(d) + 'min')
        axs[0,j].grid(axis='y')
        
        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_h_max'],
            marker="2",
            )
        axs_t0.tick_params(axis='y', which='both', length=0)
        
        axs[1,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_d_min']*dq_sc,
            label='minimun D',
            )
        axs[1,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_d_max']*dq_sc,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')
        
        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_d_max'],
            marker="2",
            )
        axs_t1.tick_params(axis='y', which='both', length=0)
        
        axs[2,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_q_min']*dq_sc,
            label='minimun P',
            )
        axs[2,j].plot(
            sp_dt['noise'],
            sp_dt['rmse_q_max']*dq_sc,
            label='maximun P',
            )
        
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')
        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            sp_dt['noise'],
            sp_dt['n_coeff_q_max'],
            marker="2",
            ) 
        axs_t2.tick_params(axis='y', which='both', length=0)              
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1
        

del ax,axs,d,d_lim,dq_add,dq_sc,fig,h_lim,i,j,q_lim,sp_dt,sp_temp,title,s,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q,

#%% Create subplots for (sp,noise,dt)

dq_sca      = 1000
dq_max_sc   = 5
coeff_add   = 2

i = 0
for s in sp:
    sp_temp = df[(df['sp'] == s)]
    h_lim = round(max(sp_temp['rmse_h_max'])) + 2
    d_lim = round((max(sp_temp['rmse_d_max']) + (dq_max_sc/dq_sca))*dq_sca)
    q_lim = round((max(sp_temp['rmse_q_max']) + (dq_max_sc/dq_sca))*dq_sca)
    
    coeff_lim_min_h   = round(
       min(
           sp_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           sp_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           sp_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           sp_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           sp_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           sp_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )
       
    j = 0
    fig, axs = plt.subplots(3,len(noise),figsize=(10, 5),sharex="all",sharey='row')
    title = 'Minimum and maximum RMSE and number of coefficients\n(SP=' + str(s) + 'hr)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Pattern time resolution (min)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)
    
    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[0,0].transAxes,
                  )
    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[1,0].transAxes,
                  )
    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation=90,
                  transform=axs[2,0].transAxes,
                  )
    
    for n in noise:        
        sp_noise = df[(df['sp'] == s) & (df['noise'] == n)].sort_values(by=['dt'])              
        
        axs[0,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(dt), right = max(dt))
        axs[0,j].set_xticks(np.arange(0, max(dt) + 1, max(dt)/len(dt)))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('noise=' + str(n) + '%')
        axs[0,j].grid(axis='y')

        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_h_max'],
            marker="2",
            )
        axs_t0.tick_params(axis='y', which='both', length=0)
        
        axs[1,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_d_min']*dq_sca,
            label='minimun D',
            )
        axs[1,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_d_max']*dq_sca,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')

        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_d_max'],
            marker="2",
            )
        axs_t1.tick_params(axis='y', which='both', length=0)
        
        axs[2,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_q_min']*dq_sca,
            label='minimun P',
            )
        axs[2,j].plot(
            sp_noise['dt'],
            sp_noise['rmse_q_max']*dq_sca,
            label='maximun P',
            )
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')                

        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            sp_noise['dt'],
            sp_noise['n_coeff_q_max'],
            marker="2",
            )
        axs_t2.tick_params(axis='y', which='both', length=0)
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1

del ax,axs,d_lim,dq_max_sc,dq_sca,fig,h_lim,i,j,q_lim,sp_temp,title,s,n,sp_noise,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q,
        
#%% Create subplots for (noise,dt,sp)

dq_sca      = 1000
dq_max_sc   = 20
coeff_add   = 2

i = 0
for n in noise:
    noise_temp = df[(df['noise'] == n)]
    h_lim = round(max(noise_temp['rmse_h_max'])) + 2
    d_lim = round((max(noise_temp['rmse_d_max']) + (dq_max_sc/dq_sca))*dq_sca)
    q_lim = round((max(noise_temp['rmse_q_max']) + (dq_max_sc/dq_sca))*dq_sca)
    coeff_lim_min_h   = round(
       min(
           noise_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           noise_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           noise_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           noise_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           noise_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           noise_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )

    j = 0
    fig, axs = plt.subplots(3,len(dt),figsize=(10, 5),sharex="all",sharey='row')
    title = 'MMinimum and maximum RMSE and number of coefficients\n(Nu=' + str(n) + '%)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Simulation period (hours)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)

    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[0,0].transAxes,
                  )

    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[1,0].transAxes,
                  )

    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[2,0].transAxes,
                  )
    
    for d in dt:        
        noise_dt = df[(df['noise'] == n) & (df['dt'] == d)].sort_values(by=['sp'])              
        
        axs[0,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(sp), right = max(sp))
        axs[0,j].set_xticks(np.arange(0, max(sp), 100))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('dt=' + str(d) + 'min')
        axs[0,j].grid(axis='y')

        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_h_max'],
            marker="2",
            )
        axs_t0.tick_params(axis='y', which='both', length=0)
        
        axs[1,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_d_min']*dq_sca,
            label='minimun D',
            )
        axs[1,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_d_max']*dq_sca,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')

        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_d_max'],
            marker="2",
            )
        axs_t1.tick_params(axis='y', which='both', length=0)
        
        axs[2,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_q_min']*dq_sca,
            label='minimun P',
            )
        axs[2,j].plot(
            noise_dt['sp'],
            noise_dt['rmse_q_max']*dq_sca,
            label='maximun P',
            )
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')                

        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            noise_dt['sp'],
            noise_dt['n_coeff_q_max'],
            marker="2",
            )
        axs_t2.tick_params(axis='y', which='both', length=0)
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1

del ax,axs,d_lim,dq_max_sc,dq_sca,fig,h_lim,i,j,q_lim,title,n,d,noise_dt,noise_temp,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q,

#%% Create subplots for (noise,sp,dt)

dq_sca      = 1000
dq_max_sc   = 20
coeff_add   = 2

i = 0
for n in noise:
    noise_temp = df[(df['noise'] == n)]
    h_lim = round(max(noise_temp['rmse_h_max'])) + 2
    d_lim = round((max(noise_temp['rmse_d_max']) + (dq_max_sc/dq_sca))*dq_sca)
    q_lim = round((max(noise_temp['rmse_q_max']) + (dq_max_sc/dq_sca))*dq_sca)
    coeff_lim_min_h   = round(
       min(
           noise_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           noise_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           noise_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           noise_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           noise_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           noise_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )

    j = 0
    fig, axs = plt.subplots(3,len(sp),figsize=(10, 5),sharex="all",sharey='row')
    title = 'Minimum and maximum RMSE and number of coefficients\n(Nu=' + str(n) + '%)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Pattern time resolution (min)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)

    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[0,0].transAxes,
                  )
    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[1,0].transAxes,
                  )
    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[2,0].transAxes,
                  )
    
    for s in sp:        
        noise_sp = df[(df['noise'] == n) & (df['sp'] == s)].sort_values(by=['dt'])              
        
        axs[0,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(dt), right = max(dt))
        axs[0,j].set_xticks(np.arange(0, max(dt) + 1, max(dt)/len(dt)))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('sp=' + str(s) + 'hours')
        axs[0,j].grid(axis='y')

        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_h_max'],
            marker="2",
            )
        axs_t0.tick_params(axis='y', which='both', length=0)
        
        axs[1,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_d_min']*dq_sca,
            label='minimun D',
            )
        axs[1,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_d_max']*dq_sca,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')

        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_d_max'],
            marker="2",
            )
        axs_t1.tick_params(axis='y', which='both', length=0)
        
        axs[2,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_q_min']*dq_sca,
            label='minimun P',
            )
        axs[2,j].plot(
            noise_sp['dt'],
            noise_sp['rmse_q_max']*dq_sca,
            label='maximun P',
            )
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')                

        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            noise_sp['dt'],
            noise_sp['n_coeff_q_max'],
            marker="2",
            )
        axs_t2.tick_params(axis='y', which='both', length=0)
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1

del ax,axs,d_lim,dq_max_sc,dq_sca,fig,h_lim,i,j,q_lim,title,n,noise_temp,s,noise_sp,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q

#%% Create subplots for (dt,noise,sp)

dq_sca      = 1000
dq_max_sc   = 5
coeff_add   = 2

i = 0
for d in dt:
    dt_temp = df[(df['dt'] == d)]
    h_lim = round(max(dt_temp['rmse_h_max'])) + 2
    d_lim = round((max(dt_temp['rmse_d_max']) + (dq_max_sc/dq_sca))*dq_sca)
    q_lim = round((max(dt_temp['rmse_q_max']) + (dq_max_sc/dq_sca))*dq_sca)
    coeff_lim_min_h   = round(
       min(
           dt_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           dt_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           dt_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           dt_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           dt_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           dt_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )

    j = 0
    fig, axs = plt.subplots(3,len(noise),figsize=(10, 5),sharex="all",sharey='row')
    title = 'Minimum and maximum RMSE and number of coefficients\n(dt=' + str(d) + 'min)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Simulation period (hours)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)

    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[0,0].transAxes,
                  )
    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[1,0].transAxes,
                  )
    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[2,0].transAxes,
                  )
    
    for n in noise:        
        dt_noise = df[(df['dt'] == d) & (df['noise'] == n)].sort_values(by=['sp'])              
        
        axs[0,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(sp), right = max(sp))
        axs[0,j].set_xticks(np.arange(0, max(sp) + 28, 100))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('noise=' + str(n) + '%')
        axs[0,j].grid(axis='y')

        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_h_max'],
            marker="2",
            )
        axs_t0.tick_params(axis='y', which='both', length=0)
        
        axs[1,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_d_min']*dq_sca,
            label='minimun D',
            )
        axs[1,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_d_max']*dq_sca,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')

        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_d_max'],
            marker="2",
            )
        axs_t1.tick_params(axis='y', which='both', length=0)
        
        axs[2,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_q_min']*dq_sca,
            label='minimun P',
            )
        axs[2,j].plot(
            dt_noise['sp'],
            dt_noise['rmse_q_max']*dq_sca,
            label='maximun P',
            )
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')                

        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            dt_noise['sp'],
            dt_noise['n_coeff_q_max'],
            marker="2",
            )
        axs_t2.tick_params(axis='y', which='both', length=0)
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1

del ax,axs,d_lim,dq_max_sc,dq_sca,fig,h_lim,i,j,q_lim,title,n,d,dt_noise,dt_temp,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q,

#%% Create subplots for (dt,sp,noise)

dq_sca      = 1000
dq_max_sc   = 5
coeff_add   = 2

i = 0
for d in dt:
    dt_temp = df[(df['dt'] == d)]
    h_lim = round(max(dt_temp['rmse_h_max'])) + 2
    d_lim = round((max(dt_temp['rmse_d_max']) + (dq_max_sc/dq_sca))*dq_sca)
    q_lim = round((max(dt_temp['rmse_q_max']) + (dq_max_sc/dq_sca))*dq_sca)

    coeff_lim_min_h   = round(
       min(
           dt_temp[['n_coeff_h_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_h   = round(
       max(
           dt_temp[['n_coeff_h_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_d   = round(
       min(
           dt_temp[['n_coeff_d_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_d   = round(
       max(
           dt_temp[['n_coeff_d_max']].max(axis=0).tolist()
           ) + coeff_add
        )
    coeff_lim_min_q   = round(
       min(
           dt_temp[['n_coeff_q_min']].min(axis=0).tolist()
           ) - coeff_add
       )
    coeff_lim_max_q   = round(
       max(
           dt_temp[['n_coeff_q_max']].max(axis=0).tolist()
           ) + coeff_add
        )

    j = 0
    fig, axs = plt.subplots(3,len(noise),figsize=(10, 5),sharex="all",sharey='row')
    title = 'Minimum and maximum RMSE and number of coefficients\n(dt=' + str(d) + 'min)'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Noise in the pattern (%)', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)

    axs[0,0].text(-0.15, 0.5,
                  'Pressure Head (wmc)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[0,0].transAxes,
                  )
    axs[1,0].text(-0.15, 0.5,
                  'Demand (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[1,0].transAxes,
                  )
    axs[2,0].text(-0.15, 0.5,
                  'Flow (l/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation = 90,
                  transform=axs[2,0].transAxes,
                  )
    
    for s in sp:        
        dt_sp = df[(df['dt'] == d) & (df['sp'] == s)].sort_values(by=['noise'])              
        
        axs[0,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_h_min'],
            label='minimun H',
            )
        axs[0,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_h_max'],
            label='maximun H',            
            )
        #axs[0,j].set_xlim(left = min(noise), right = max(noise))
        axs[0,j].set_xticks(np.arange(0, max(noise) + 1, max(noise)/len(noise)))
        axs[0,j].set_ylim(top = h_lim)
        axs[0,j].set_title('sp=' + str(s) + 'hours')
        axs[0,j].grid(axis='y')

        axs_t0 = axs[0,j].twinx()
        axs_t0.set_ylim(bottom = coeff_lim_min_h, top = coeff_lim_max_h)
        axs_t0.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_h_min'],
            marker="2",            
            )
        axs_t0.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_h_max'],
            marker="2",
            )
        
        axs[1,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_d_min']*dq_sca,
            label='minimun D',
            )
        axs[1,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_d_max']*dq_sca,
            label='maximun H',
            )
        axs[1,j].set_ylim(top = d_lim)
        axs[1,j].grid(axis='y')

        axs_t1 = axs[1,j].twinx()
        axs_t1.set_ylim(bottom = coeff_lim_min_d, top = coeff_lim_max_d)
        axs_t1.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_d_min'],
            marker="2",            
            )
        axs_t1.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_d_max'],
            marker="2",
            )
        
        axs[2,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_q_min']*dq_sca,
            label='minimun P',
            )
        axs[2,j].plot(
            dt_sp['noise'],
            dt_sp['rmse_q_max']*dq_sca,
            label='maximun P',
            )
        axs[2,j].set_ylim(top = q_lim)
        axs[2,j].grid(axis='y')                

        axs_t2 = axs[2,j].twinx()
        axs_t2.set_ylim(bottom = coeff_lim_min_q, top = coeff_lim_max_q)
        axs_t2.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_q_min'],
            marker="2",            
            )
        axs_t2.scatter(
            dt_sp['noise'],
            dt_sp['n_coeff_q_max'],
            marker="2",
            )
        
        for ax in axs.flatten():
            ax.tick_params(axis='both', which='major', length = 0)
        
        plt.show()
        
        if i < 3:
            i += 1
        else:
            i = 0
        j += 1
        
del ax,axs,d_lim,dq_max_sc,dq_sca,fig,h_lim,i,j,q_lim,title,d,dt_temp,dt_sp,s,axs_t0,axs_t1,axs_t2,coeff_add
del coeff_lim_max_h,coeff_lim_max_d,coeff_lim_max_q,coeff_lim_min_h,coeff_lim_min_d,coeff_lim_min_q,