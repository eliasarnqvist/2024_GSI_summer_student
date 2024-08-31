# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# =============================================================================
# Import data
# =============================================================================

data_files = ['data\\2024-08-15_data_chA_find_tot_relation\\tek0009ALL.csv',
              'data\\2024-08-15_data_chA_find_tot_relation\\tek0010ALL.csv',
              'data\\2024-08-15_data_chA_find_tot_relation\\tek0011ALL.csv',
              'data\\2024-08-28_ABE_and_CD\\tek0012ALL.csv',
              'data\\2024-08-28_ABE_and_CD\\tek0013ALL.csv',
              'data\\2024-08-28_ABE_and_CD\\tek0014ALL.csv',
              'data\\2024-08-28_ABE_and_CD\\tek0015ALL.csv',
              'data\\2024-08-28_ABE_and_CD\\tek0016ALL.csv']

dfs = []

for i, file_path in enumerate(data_files):
    if i < 3:
        col_names = ['t', 'V1', 'V2']
    else:
        col_names = ['t', 'V1', 'V2', 'V3']
    
    df = pd.read_csv(file_path, skiprows=21, names=col_names)
    
    t_start = -1e-6
    t_stop = 4e-6

    idx1 = (df['t'] - t_start).abs().idxmin()
    idx2 = (df['t'] - t_stop).abs().idxmin()
    
    dfs.append(df.iloc[idx1:idx2+1])

# =============================================================================
# Find time over threshold
# =============================================================================

def linear_interpolation_x(x1, y1, x2, y2, y):
    x = x1 + ((x2 - x1) / (y2 - y1) * (y - y1))
    return x

def find_dt(t, U_smooth, threshold):
    try:
        # this returns the intex before the crossing
        crossing_index_leading = np.where((U_smooth[:-1] < threshold) & 
                                          (U_smooth[1:] >= threshold))[0][0]
        crossing_index_falling = np.where((U_smooth[:-1] >= threshold) & 
                                          (U_smooth[1:] < threshold))[0][0]
        
        t_1_leading = t[crossing_index_leading]
        t_2_leading = t[crossing_index_leading + 1]
        U_1_leading = U_smooth[crossing_index_leading]
        U_2_leading = U_smooth[crossing_index_leading + 1]
        t_leading = linear_interpolation_x(t_1_leading, U_1_leading, 
                                           t_2_leading, U_2_leading, 
                                           threshold)
        
        t_1_falling = t[crossing_index_falling]
        t_2_falling = t[crossing_index_falling + 1]
        U_1_falling = U_smooth[crossing_index_falling]
        U_2_falling = U_smooth[crossing_index_falling + 1]
        t_falling = linear_interpolation_x(t_1_falling, U_1_falling, 
                                           t_2_falling, U_2_falling, 
                                           threshold)
        
        dt = t_falling - t_leading
    except IndexError:
        # dt = None
        dt = 0
    return dt

threshold = 0.2

fig, ax = plt.subplots(figsize=(100/25, 100/25))

data_dict = {'dt':[], 'amp':[], 'err_dt_up':[], 'err_dt_down':[]}

for i, df in enumerate(dfs):
    t = df['t'].to_numpy()
    U = df['V1'].astype(float).to_numpy()
    U_smooth = np.convolve(U, np.ones(50)/50, mode='valid')
    
    ax.plot(t[:-49], U_smooth)
    
    amp = np.max(U)
    
    baseline_error = 4.915 / 1e3
    
    dt = find_dt(t, U_smooth, threshold)
    dt_big = find_dt(t, U_smooth + baseline_error, threshold)
    dt_small = find_dt(t, U_smooth - baseline_error, threshold)
    
    data_dict['dt'].append(dt*1e9)
    data_dict['amp'].append(amp)
    data_dict['err_dt_up'].append(abs(dt_big - dt) * 1e9)
    data_dict['err_dt_down'].append(abs(dt_small - dt) * 1e9)
    
    print(dt, dt_big, dt_small)

df = pd.DataFrame(data_dict)

df.to_csv('results/tot-to-amp.csv', index=False)

# =============================================================================
# Plot
# =============================================================================

# plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(100/inch_to_mm, 100/inch_to_mm))

ax.errorbar(data_dict['dt'], data_dict['amp'], 
            yerr=baseline_error, 
            xerr=[data_dict['err_dt_down'], data_dict['err_dt_up']],
            fmt='.', capsize=3, capthick=1, markersize=4, label='A', lw=1)

# ax.set_yticks(ticks=np.arange(-800, 700, 200))
# ax.tick_params(axis='y', labelsize=10)
# ax.set_xticks(ticks=np.arange(0, 3.5, 0.5))
# ax.tick_params(axis='x', labelsize=10)
# ax.set_xlim([-4, 1])
# ax.set_ylim([-900, 700])
ax.set_xlabel(r'Time-over-threshold (ns)', size=10)
ax.set_ylabel('Amplitude (mV)', size=10)

# ax.legend(frameon=False, loc='upper right', ncols=2, fontsize=10, 
#           handlelength=1, handletextpad=0.5, columnspacing=0.5)

plt.tight_layout(pad=0.5)

# save_name = 'oscilloscope_baselines_ABCDE'
# plt.savefig(f'figures\\{save_name}.jpg', dpi=300)
# plt.savefig(f'figures\\{save_name}.pdf')


