
# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct
import pickle
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# =============================================================================
# Import data
# =============================================================================

data_folder = r'data'
file_1 = r'2024-08-28_meas_0_ToF_beam-center.lst'
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
        # For "time_patch=9" data is stored in 8 byte chunks (1 byte = 8 bits)
        byte_data = file.read(8)
        
        # For when we reach the end
        if len(byte_data) < 8:
            break
        
        value = struct.unpack('<Q', byte_data)[0]
        if value != 0:
            # Fill with zeros that were not printed explicitly (32 bits total)
            bits_data = bin(value)[2:].zfill(64)
            # print(bits_data)
            
            channel = int(bits_data[60:64], 2)
            channel_word = dict_channel[channel]
            
            edge = int(bits_data[59])
            edge_word = dict_edge[edge]
            
            time = int(bits_data[21:59], 2)
            # Time is in units of the time bin size (bin size * 80 ps)
            
            sweep = int(bits_data[1:21], 2)
            
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
    # (rising + falling for A, B, C, D, E)
    if len(sweep_dict) != 10:
        remove_these.append(sweep_number)
    else:
        # Rising edge events (START, STOP 2, 3, 4, 5)
        rising_events = [0, 0, 0, 0, 0]
        # Falling edge events (STOP 1, 2, 3, 4, 5)
        falling_events = [0, 0, 0, 0, 0]
        
        for event_time, event_dict in sweep_dict.items():
            if event_dict['edge_w'] == 'RISING':
                if event_dict['channel_w'] == 'STOP_1':
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
        
        if rising_events != [1, 1, 1, 1, 1] or falling_events != [1, 1, 1, 1, 1]:
            remove_these.append(sweep_number)

for sweep_number in remove_these:
    dict_data_sweep.pop(sweep_number)

# =============================================================================
# Time over threshold, amplitude, xy position, time of flight
# =============================================================================

surrogate_df = pd.read_csv('resources\\surrogate_functions.csv')

def surrogate_function(ToT, a, b, c, d):
    x = ToT
    amplitude = a + b / (x - c) + d / (x - c)**2
    return amplitude

def xy_position(a, b, c, d):
    x = (b + c) / (a + b + c + d)
    y = (a + b) / (a + b + c + d)
    return [x, y]

def CFT(t1, t2, f):
    return t1 + f * (t2 - t1)

dict_data_timing = {}

measurement_index = 0

for sweep_number, sweep_dict in dict_data_sweep.items():
    times = [0, 0, 0, 0, 0]
    times_E = [0, 0]
    amplitudes = [0, 0, 0, 0]
    
    for event_time, event_dict in sweep_dict.items():
        
        if event_dict['edge_w'] == 'RISING':
            if event_dict['channel_w'] == 'STOP_1':
                times[0] -= event_dict['time']
                times_E[0] = event_dict['time']
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
                times_E[1] = event_dict['time']
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
    times_E = [time * 80 / 1e3 for time in times_E]
    
    for j, time in enumerate(times[1:]):
        popts_list = np.fromstring(surrogate_df['popts'][0].strip('[]'), sep=' ')
        amplitude = surrogate_function(time, *popts_list)
        amplitudes[j] = amplitude
    
    xy = xy_position(amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3])
    
    ToF = times_E[0]
    CFDs = np.arange(0, 1.001, 0.01)
    # CFDs = np.array([0, 0.5, 1])
    ToFs = CFT(times_E[0], times_E[1], CFDs)
    
    dict_data_timing[measurement_index] = {'ToT':times,
                                           'amplitudes':amplitudes,
                                           'xy':xy,
                                           'ToF':ToF,
                                           'ToF_CFD':ToFs,
                                           'CFDs':CFDs}
    
    measurement_index += 1

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

def gaussian(T, a1, b1, c1):
    function = a1*np.exp(-(T-b1)**2/(2*c1**2))
    return function

# %%

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))

vector = np.zeros((len(dict_data_timing), len(CFDs)))
for measurement_number, measurement_dict in dict_data_timing.items():
    vector[measurement_number, :] = np.array(measurement_dict['ToF_CFD'])

FWHMs = np.array([])

for i in range(len(CFDs)):
    this_histo, edges = np.histogram(vector[:, i] / 1e3, bins=200, range=(41.96, 42.16))
    ax.step(edges[:-1], this_histo, where='post')
    
    amp_guess = np.max(this_histo)
    pos_guess = edges[int(np.argmax(this_histo))]
    
    popt, pcov = curve_fit(gaussian, edges[:-1], this_histo, 
                           sigma=np.sqrt(np.where(this_histo == 0, 1, this_histo)), 
                           p0=[amp_guess, pos_guess, 0.05])
    ax.plot(edges[:-1], gaussian(edges[:-1], *popt))
    FWHM = 2.35482 * abs(popt[2])
    FWHMs = np.append(FWHMs, FWHM)
    # text1 = 'FWHM = {:.1f}'.format(round(FWHM * 1e3, 2)) + ' ns'
    # ax.text(0.1, 0.9, text1, ha='left', va='center', transform=ax1.transAxes)
    # print(FWHM)

plt.tight_layout(pad=0.5)
# fig.subplots_adjust(hspace=0, wspace=0)

# %%

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))

ax.plot(CFDs, FWHMs)

best_CFD = CFDs[np.argmin(FWHMs)]
print(best_CFD, np.min(FWHMs))

# 0.57 is best
