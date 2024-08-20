
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
file_1 = r'2024-08-19_TEST-4_ABCDE_binary.lst'
filepath_1 = data_folder + "\\" + file_1

dict_data_channel = {}
dict_data_sweep = {}


dict_channel = {0:'Zero', 1:'Counter real/live time', 2:'Counter data?',
                3:'Unused', 4:'Unused', 5:'Unused', 6:'START',
                7:'Reserved', 8:'STOP_1', 9:'STOP_2', 10:'STOP_3',
                11:'STOP_4', 12:'STOP_5', 13:'STOP_6', 14:'STOP_7',
                15:'STOP_8'}
dict_edge = {0:'RISING', 1:'FALLING'}


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
            # Is time in units of time bins (8 * 80 ps) or 80 ps? Not sure yet
            
            sweep = int(bits_data[1:11], 2)
            
            loss = bits_data[0]
            
            print(sweep, channel_word, edge_word, time, loss)
            
            dict_ch = {'sweep':sweep, 'ch':channel, 'ch_w':channel_word, 
                       'edge':edge, 'edge_w':edge_word, 'time':time, 
                       'loss':loss}
            
            if sweep not in dict_data_sweep:
                dict_data_sweep[sweep] = {}
            dict_data_sweep[sweep][time] = dict_ch
            
            if channel_word not in dict_data_channel:
                dict_data_channel[channel_word] = {}
            dict_data_channel[channel_word][time] = dict_ch

# Save the event dictionary as a binary pickle file
save_name = 'extracted_data\\' + file_1[:-4] + '_extracted.pkl'
with open(save_name, 'wb') as file:
    pickle.dump(dict_data_sweep, file)

# An example where all channels are triggered
# example = dict_event[891]
# print(example)

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
vector = []
for key, value in dict_data_channel['STOP_1'].items():
    this_time = value['time']
    vector.append(this_time)
this_histo, edges = np.histogram(vector, bins=4096, range=(0, 4096*8))
ax1.step(edges[:-1], this_histo, where='post')
vector = []
for key, value in dict_data_channel['STOP_2'].items():
    this_time = value['time']
    vector.append(this_time)
this_histo, edges = np.histogram(vector, bins=4096, range=(0, 4096*8))
ax2.step(edges[:-1], this_histo, where='post')
vector = []
for key, value in dict_data_channel['STOP_3'].items():
    this_time = value['time']
    vector.append(this_time)
this_histo, edges = np.histogram(vector, bins=4096, range=(0, 4096*8))
ax3.step(edges[:-1], this_histo, where='post')
vector = []
for key, value in dict_data_channel['STOP_4'].items():
    this_time = value['time']
    vector.append(this_time)
this_histo, edges = np.histogram(vector, bins=4096, range=(0, 4096*8))
ax4.step(edges[:-1], this_histo, where='post')
vector = []
for key, value in dict_data_channel['STOP_5'].items():
    this_time = value['time']
    vector.append(this_time)
this_histo, edges = np.histogram(vector, bins=4096, range=(0, 4096*8))
ax5.step(edges[:-1], this_histo, where='post')

plt.tight_layout(pad=0.5)
# fig.subplots_adjust(hspace=0, wspace=0)


