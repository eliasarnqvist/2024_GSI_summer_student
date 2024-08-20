# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct
import pickle

# =============================================================================
# Import data
# =============================================================================

with open('extracted_data\\2024-08-15_TEST-3_ABCDE_binary_extracted.pkl', 
          'rb') as file:
    loaded_dict = pickle.load(file)

# =============================================================================
# Select events
# =============================================================================

# Stop_1 should only be triggered once in the beginnning

remove_list = []

for sweep_number, sweep_data_dict in loaded_dict.items():
    
    for event_time, event_data_dict in sweep_data_dict.items():
        
        if event_data_dict['ch_w'] == 'STOP_1':
            # loaded_dict.pop(sweep_number)
            # remove_list.append(sweep_number)
            break

for sweep_number in remove_list:
    loaded_dict.pop(sweep_number)

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

# fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
fig = plt.figure(figsize=(160/inch_to_mm, 120/inch_to_mm))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))
number = 46
data = []
time = []
for key, value in loaded_dict[number].items():
    if value['ch_w'] == 'STOP_1':
        this_time = value['time']
        time.append(this_time)
        if value['edge_w'] == 'RISING':
            data.append(1)
        else:
            data.append(-1)
ax1.plot(time, data, ls='None', marker='o')
for key, value in loaded_dict[number].items():
    if value['ch_w'] == 'STOP_2':
        this_time = value['time']
        time.append(this_time)
        if value['edge_w'] == 'RISING':
            data.append(1)
        else:
            data.append(-1)
ax2.plot(time, data, ls='None', marker='o')
for key, value in loaded_dict[number].items():
    if value['ch_w'] == 'STOP_3':
        this_time = value['time']
        time.append(this_time)
        if value['edge_w'] == 'RISING':
            data.append(1)
        else:
            data.append(-1)
ax3.plot(time, data, ls='None', marker='o')
for key, value in loaded_dict[number].items():
    if value['ch_w'] == 'STOP_4':
        this_time = value['time']
        time.append(this_time)
        if value['edge_w'] == 'RISING':
            data.append(1)
        else:
            data.append(-1)
ax4.plot(time, data, ls='None', marker='o')
for key, value in loaded_dict[number].items():
    if value['ch_w'] == 'STOP_5':
        this_time = value['time']
        time.append(this_time)
        if value['edge_w'] == 'RISING':
            data.append(1)
        else:
            data.append(-1)
ax5.plot(time, data, ls='None', marker='o')
plt.tight_layout(pad=0.5)
# fig.subplots_adjust(hspace=0, wspace=0)


