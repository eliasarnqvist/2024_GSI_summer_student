
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

data_folder = r'data'
file_1 = r'2024-08-20_TEST-5_ABCDE_binary.lst'
filepath_1 = data_folder + "\\" + file_1

dict_data_sweep = {}

dict_channel = {0:'Zero', 1:'Counter real/live time', 2:'Counter data?',
                3:'Unused', 4:'Unused', 5:'Unused', 6:'START',
                7:'Reserved', 8:'STOP_1', 9:'STOP_2', 10:'STOP_3',
                11:'STOP_4', 12:'STOP_5', 13:'STOP_6', 14:'STOP_7',
                15:'STOP_8'}
dict_edge = {0:'RISING', 1:'FALLING'}

sweep_counter = 0
last_sweep = 0

with open(filepath_1, 'rb') as file:
    
    # Skip to the binary data
    for line in file:
        if line.strip() == b'[DATA]':
            break
    
    while True:
        # For "time_patch=1c" data is stored in 4 byte chunks (1 byte = 8 bits)
        byte_data = file.read(4)
        
        # For when we reach the end
        if len(byte_data) < 4:
            break
        
        value = struct.unpack('<I', byte_data)[0]
        if value != 0:
            # Fill with zeros that were not printed explicitly (32 bits total)
            bits_data = bin(value)[2:].zfill(32)
            # print(bits_data)
            
            channel = int(bits_data[28:32], 2)
            channel_word = dict_channel[channel]
            
            edge = int(bits_data[27])
            edge_word = dict_edge[edge]
            
            time = int(bits_data[11:27], 2)
            # Time is in units of the time bin size (bin size * 80 ps)
            
            sweep = int(bits_data[1:11], 2)
            
            loss = bits_data[0]
            
            # print(sweep, channel_word, edge_word, time, loss)
            
            if sweep != last_sweep:
                sweep_counter += 1
            last_sweep = sweep
            
            event_dict = {'sweep':sweep, 'sweep_counter':sweep_counter, 
                          'channel':channel, 'channel_w':channel_word, 
                          'edge':edge, 'edge_w':edge_word, 'time':time, 
                          'loss':loss}
            
            if sweep_counter not in dict_data_sweep:
                dict_data_sweep[sweep_counter] = {}
            dict_data_sweep[sweep_counter][time] = event_dict

# =============================================================================
# Event selection
# =============================================================================

# Vector of sweeps to remove
remove_these = []

# Only keep sweeps that make sense
for sweep_number, sweep_dict in dict_data_sweep.items():
    
    # First test the number of events in the sweep, which should be 10
    if len(sweep_dict) != 10:
        remove_these.append(sweep_number)
    else:
        # Rising edge events (START, STOP 2, 3, 4, 5)
        rising_events = [0, 0, 0, 0, 0]
        # Falling edge events (STOP 1, 2, 3, 4, 5)
        falling_events = [0, 0, 0, 0, 0]
        
        for event_time, event_dict in sweep_dict.items():
            if event_dict['edge_w'] == 'RISING':
                if event_dict['channel_w'] == 'START':
                    rising_events[0] += 1
                elif event_dict['channel_w'] == 'STOP_2':
                    rising_events[1] += 1
                elif event_dict['channel_w'] == 'STOP_3':
                    rising_events[2] += 1
                elif event_dict['channel_w'] == 'STOP_4':
                    rising_events[3] += 1
                elif event_dict['channel_w'] == 'STOP_5':
                    rising_events[4] += 1
            elif event_dict['edge_w'] == 'FALLING':
                if event_dict['channel_w'] == 'STOP_1':
                    falling_events[0] += 1
                elif event_dict['channel_w'] == 'STOP_2':
                    falling_events[1] += 1
                elif event_dict['channel_w'] == 'STOP_3':
                    falling_events[2] += 1
                elif event_dict['channel_w'] == 'STOP_4':
                    falling_events[3] += 1
                elif event_dict['channel_w'] == 'STOP_5':
                    falling_events[4] += 1
        
        if rising_events != [1, 1, 1, 1, 1] and falling_events != [1, 1, 1, 1, 1]:
            remove_these.append(sweep_number)

for sweep_number in remove_these:
    dict_data_sweep.pop(sweep_number)

# =============================================================================
# Time over threshold (optionally time under threshold)
# =============================================================================

dict_data_timing = {}

measurement_index = 0

for sweep_number, sweep_dict in dict_data_sweep.items():
    times = [0, 0, 0, 0, 0]
    
    for event_time, event_dict in sweep_dict.items():
        
        if event_dict['edge_w'] == 'RISING':
            if event_dict['channel_w'] == 'START':
                times[0] -= event_dict['time']
            elif event_dict['channel_w'] == 'STOP_2':
                times[1] -= event_dict['time']
            elif event_dict['channel_w'] == 'STOP_3':
                times[2] -= event_dict['time']
            elif event_dict['channel_w'] == 'STOP_4':
                times[3] -= event_dict['time']
            elif event_dict['channel_w'] == 'STOP_5':
                times[4] -= event_dict['time']
        elif event_dict['edge_w'] == 'FALLING':
            if event_dict['channel_w'] == 'STOP_1':
                times[0] += event_dict['time']
            elif event_dict['channel_w'] == 'STOP_2':
                times[1] += event_dict['time']
            elif event_dict['channel_w'] == 'STOP_3':
                times[2] += event_dict['time']
            elif event_dict['channel_w'] == 'STOP_4':
                times[3] += event_dict['time']
            elif event_dict['channel_w'] == 'STOP_5':
                times[4] += event_dict['time']
    
    # The time in ns
    times = [time * 80 / 1e3 for time in times]
    
    dict_data_timing[measurement_index] = times
    
    measurement_index += 1

# =============================================================================
# Save
# =============================================================================

# Save the event dictionary as a binary pickle file
save_name = 'selected_data\\' + file_1[:-4] + '_selected.pkl'
with open(save_name, 'wb') as file:
    pickle.dump(dict_data_sweep, file)

save_name = 'time_data\\' + file_1[:-4] + '_timing.pkl'
with open(save_name, 'wb') as file:
    pickle.dump(dict_data_timing, file)

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

# %%

# fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
fig = plt.figure(figsize=(160/inch_to_mm, 160/inch_to_mm))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))
vector_ch1 = []
vector_ch2 = []
vector_ch3 = []
vector_ch4 = []
vector_ch5 = []
for sweep_number, sweep_dict in dict_data_sweep.items():
    for event_time, event_dict in sweep_dict.items():
        if event_dict['channel_w'] in ['STOP_1', 'START']:
            vector_ch1.append(event_dict['time'])
        elif event_dict['channel_w'] == 'STOP_2':
            vector_ch2.append(event_dict['time'])
        elif event_dict['channel_w'] == 'STOP_3':
            vector_ch3.append(event_dict['time'])
        elif event_dict['channel_w'] == 'STOP_4':
            vector_ch4.append(event_dict['time'])
        elif event_dict['channel_w'] == 'STOP_5':
            vector_ch5.append(event_dict['time'])

this_histo, edges = np.histogram(vector_ch1, bins=4096, range=(0, 4096*8))
ax1.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch2, bins=4096, range=(0, 4096*8))
ax2.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch3, bins=4096, range=(0, 4096*8))
ax3.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch4, bins=4096, range=(0, 4096*8))
ax4.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch5, bins=4096, range=(0, 4096*8))
ax5.step(edges[:-1], this_histo, where='post')

plt.tight_layout(pad=0.5)
# fig.subplots_adjust(hspace=0, wspace=0)

# %%

fig = plt.figure(figsize=(160/inch_to_mm, 160/inch_to_mm))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))
vector_ch1 = []
vector_ch2 = []
vector_ch3 = []
vector_ch4 = []
vector_ch5 = []
for measurement_number, measurement_list in dict_data_timing.items():
    vector_ch1.append(measurement_list[0])
    vector_ch2.append(measurement_list[1])
    vector_ch3.append(measurement_list[2])
    vector_ch4.append(measurement_list[3])
    vector_ch5.append(measurement_list[4])

this_histo, edges = np.histogram(vector_ch1, bins=1000, range=(0, 1000))
ax1.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch2, bins=1000, range=(0, 1000))
ax2.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch3, bins=1000, range=(0, 1000))
ax3.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch4, bins=1000, range=(0, 1000))
ax4.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch5, bins=1000, range=(0, 1000))
ax5.step(edges[:-1], this_histo, where='post')

plt.tight_layout(pad=0.5)
# fig.subplots_adjust(hspace=0, wspace=0)




