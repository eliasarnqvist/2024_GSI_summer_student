# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import data
# =============================================================================

data_folder = '2024-08-08_preamp_simulation'
file_name = 'charge_amp_shaper_0.01p_2.0p_0.01p.txt'
file_path = data_folder + "\\" + file_name

data_dict = {}
partial_dict = {}
i = 0

with open(file_path, 'r') as file:
    header = file.readline()
    for line in file:
        
        if "Step Information" in line:
            current_c6 = line.split('C6_value=')[1].split()[0]
            if current_c6[-1] == 'f':
                mult = 1e-15
            elif current_c6[-1] == 'p':
                mult = 1e-12
            current_c6 = float(current_c6[:-1]) * mult
            
            if i > 0:
                data_dict[i] = partial_dict
            partial_dict = {'t':np.array([]), 'U':np.array([]), 
                            'Q':current_c6, 'dt':[], 'thld':[], 'amp':[]}
            i += 1
        else:
            t = float(line.split()[0])
            U = float(line.split()[1])
            partial_dict['t'] = np.append(partial_dict['t'], t)
            partial_dict['U'] = np.append(partial_dict['U'], U)
    
    data_dict[i] = partial_dict
        
# =============================================================================
# Time over threshold
# =============================================================================

def linear_interpolation_x(x1, y1, x2, y2, y):
    x = x1 + ((x2 - x1) / (y2 - y1) * (y - y1))
    return x

all_thresholds = [0.1, 0.5, 1, 2, 3, 4, 5]

for i, this_threshold in enumerate(all_thresholds):
    
    threshold = -1.038822e+00 + this_threshold
    
    for key, value in data_dict.items():
        t = value['t']
        U = value['U']
        
        amp = np.max(U)
        
        try:
            # this returns the intex before the crossing
            crossing_index_leading = np.where((U[:-1] < threshold) & 
                                              (U[1:] >= threshold))[0][0]
            crossing_index_falling = np.where((U[:-1] >= threshold) & 
                                              (U[1:] < threshold))[0][0]
            
            t_1_leading = t[crossing_index_leading]
            t_2_leading = t[crossing_index_leading + 1]
            U_1_leading = U[crossing_index_leading]
            U_2_leading = U[crossing_index_leading + 1]
            t_leading = linear_interpolation_x(t_1_leading, U_1_leading, 
                                               t_2_leading, U_2_leading, 
                                               threshold)
            
            t_1_falling = t[crossing_index_falling]
            t_2_falling = t[crossing_index_falling + 1]
            U_1_falling = U[crossing_index_falling]
            U_2_falling = U[crossing_index_falling + 1]
            t_falling = linear_interpolation_x(t_1_falling, U_1_falling, 
                                               t_2_falling, U_2_falling, 
                                               threshold)
            
            dt = t_falling - t_leading
        except IndexError:
            # dt = None
            dt = 0
        
        data_dict[key]['thld'].append(this_threshold)
        data_dict[key]['dt'].append(dt)
        data_dict[key]['amp'] = amp

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
dd = {}
for i, this_threshold in enumerate(all_thresholds):
    xx = []
    yy = []
    for key, value in data_dict.items():
        x = value['Q']
        y = value['dt'][i]
        
        if x != None and y != None:
            x = x * 1e12
            y = y * 1e9
            
            xx.append(x)
            yy.append(y)
    dd[this_threshold] = {'x':xx, 'y':yy}
    ax.plot(xx, yy, label=this_threshold)
ax.set_xlabel('Deposited charge (pC)')
ax.set_ylabel('Time-over-threshold (ns)')
ax.legend(title='Threshold (V)', loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim([0, 2])
ax.set_ylim([0, None])
plt.tight_layout(pad=0.5)

# save_name = 'simulated_time-over-threshold'
# plt.savefig(f'figures\\{save_name}.jpg')
# plt.savefig(f'figures\\{save_name}.pdf')



fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
for key, value in data_dict.items():
    if value['Q'] == 2.7e-13:
        x = value['t']
        y = value['U']
        ax.plot(x, y)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
# ax.legend(title='Threshold (V)', loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xlim([0, 2])
# ax.set_ylim([0, None])
plt.tight_layout(pad=0.5)



fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
dd = {}
for i, this_threshold in enumerate(all_thresholds):
    xx = []
    yy = []
    for key, value in data_dict.items():
        x = value['Q']
        y = value['amp']
        
        if x != None and y != None:
            x = x * 1e12
            
            xx.append(x)
            yy.append(y)
    dd[this_threshold] = {'x':xx, 'y':yy}
    ax.plot(xx, yy, label=this_threshold)
ax.set_xlabel('Deposited charge (pC)')
ax.set_ylabel('Amplitude (V)')
ax.legend(title='Threshold (V)', loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xlim([0, 2])
# ax.set_ylim([0, None])
plt.tight_layout(pad=0.5)



fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
dd = {}
for i, this_threshold in enumerate(all_thresholds):
    xx = []
    yy = []
    for key, value in data_dict.items():
        x = value['dt'][i]
        y = value['amp']
        
        if x != None and y != None:
            x = x * 1e9
            
            xx.append(x)
            yy.append(y)
    dd[this_threshold] = {'x':xx, 'y':yy}
    ax.plot(xx, yy, label=this_threshold)
ax.set_xlabel('Time-over-threshold (ns)')
ax.set_ylabel('Amplitude (V)')
ax.legend(title='Threshold (V)', loc='center left', bbox_to_anchor=(1, 0.5))
# ax.set_xlim([0, 2])
# ax.set_ylim([0, None])
plt.tight_layout(pad=0.5)






