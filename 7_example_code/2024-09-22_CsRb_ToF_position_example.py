
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
from matplotlib.colors import LogNorm

# =============================================================================
# Import data
# =============================================================================

data_folder = r'data'
file_1 = r'2024-08-29_meas_1_ToF_RbCs_15min.lst'
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

#%%

# =============================================================================
# Event selection
# =============================================================================

# Vector of sweeps to remove
remove_these = []

# Only keep sweeps that make sense
for sweep_number, sweep_dict in dict_data_sweep.items():
    
    # First test the number of events in the sweep, which should be 10, 20, 30
    # (rising + falling for A, B, C, D, E)
    num_events = len(sweep_dict)
    if num_events != 10 and num_events != 20 and num_events != 30:
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
        
        if not (all(x in [1, 2, 3] for x in rising_events) and 
                all(x in [1, 2, 3] for x in falling_events)):
            remove_these.append(sweep_number)

for sweep_number in remove_these:
    dict_data_sweep.pop(sweep_number)

#%%

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
    
    event_index = 0
    
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
        
        # Every tenth event we should move on to the next ion detection
        # (rising + falling for ABCDE for every ion detection, so 10 (0-9))
        if event_index % 9 == 0 and event_index > 0:
            # The time in ns
            times = [time * 80 / 1e3 for time in times]
            times_E = [time * 80 / 1e3 for time in times_E]
            # print(times_E)
            
            for j, time in enumerate(times[1:]):
                popts_list = np.fromstring(surrogate_df['popts'][0].strip('[]'), sep=' ')
                amplitude = surrogate_function(time, *popts_list)
                amplitudes[j] = amplitude
            
            xy = xy_position(amplitudes[0], amplitudes[1], amplitudes[2], amplitudes[3])
            
            ToF = [times_E[0], times_E[1]]
            CFDs = 0.57
            ToFs = CFT(times_E[0], times_E[1], 0.57)
            
            dict_data_timing[measurement_index] = {'ToT':times,
                                                   'amplitudes':amplitudes,
                                                   'xy':xy,
                                                   'ToF':ToF,
                                                   'ToF_CFD':ToFs,
                                                   'CFD':CFDs}
            
            times = [0, 0, 0, 0, 0]
            times_E = [0, 0]
            amplitudes = [0, 0, 0, 0]
            
            measurement_index += 1
        
        event_index += 1

#%%

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

cmap = plt.cm.viridis
cmap.set_under('white')

inch_to_mm = 25.4
colors = plt.cm.tab10

def gaussian(T, a1, b1, c1):
    function = a1*np.exp(-(T-b1)**2/(2*c1**2))
    return function

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

this_histo, edges = np.histogram(vector_ch1, bins=4096, range=(41497.6/0.08, 41497.6/0.08 + 4096*8))
ax1.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch2, bins=4096, range=(41497.6/0.08, 41497.6/0.08 + 4096*8))
ax2.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch3, bins=4096, range=(41497.6/0.08, 41497.6/0.08 + 4096*8))
ax3.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch4, bins=4096, range=(41497.6/0.08, 41497.6/0.08 + 4096*8))
ax4.step(edges[:-1], this_histo, where='post')

this_histo, edges = np.histogram(vector_ch5, bins=4096, range=(41497.6/0.08, 41497.6/0.08 + 4096*8))
ax5.step(edges[:-1], this_histo, where='post')

plt.tight_layout(pad=0.5)
ax1.set_title('Histograms of trigger crossing times')

# %%

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
vector_ch1 = []
vector_ch2 = []
vector_ch3 = []
vector_ch4 = []
vector_ch5 = []
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_ch1.append(measurement_dict['ToT'][0])
    vector_ch2.append(measurement_dict['ToT'][1])
    vector_ch3.append(measurement_dict['ToT'][2])
    vector_ch4.append(measurement_dict['ToT'][3])
    vector_ch5.append(measurement_dict['ToT'][4])

this_histo, edges = np.histogram(vector_ch1, bins=100, range=(0, 1000))
ax.step(edges[:-1], this_histo, where='post', label='E')

this_histo, edges = np.histogram(vector_ch2, bins=100, range=(0, 1000))
ax.step(edges[:-1], this_histo, where='post', label='A')

this_histo, edges = np.histogram(vector_ch3, bins=100, range=(0, 1000))
ax.step(edges[:-1], this_histo, where='post', label='B')

this_histo, edges = np.histogram(vector_ch4, bins=100, range=(0, 1000))
ax.step(edges[:-1], this_histo, where='post', label='C')

this_histo, edges = np.histogram(vector_ch5, bins=100, range=(0, 1000))
ax.step(edges[:-1], this_histo, where='post', label='D')

ax.set_xlabel('Time over threshold (ns)', size=12)
ax.legend()

plt.tight_layout(pad=0.5)

# %%

fig, ax = plt.subplots(1, 3, figsize=(148/inch_to_mm,55/inch_to_mm), 
                       sharey=True)

vector_1 = []
vector_2 = []
vector_3 = []
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_1.append(measurement_dict['ToF'][0])
    vector_2.append(measurement_dict['ToF'][1])
    vector_3.append(measurement_dict['ToF_CFD'])

# this_histo, edges = np.histogram(vector_1, bins=5000, range=(23e3, 55e3))
# ax.step(edges[:-1], this_histo, where='post', lw=1.5)

# this_histo, edges = np.histogram(vector_2, bins=5000, range=(23e3, 55e3))
# ax.step(edges[:-1], this_histo, where='post', lw=1.5)

this_histo, edges = np.histogram(vector_3, bins=20000, range=(33e3, 43e3))
edges_us = edges / 1e3
ax[0].step(edges_us[:-1], this_histo, where='post', lw=1)
ax[1].step(edges_us[:-1], this_histo, where='post', lw=1)
ax[2].step(edges_us[:-1], this_histo, where='post', lw=1)

ax[0].spines.right.set_visible(False)
ax[1].spines.left.set_visible(False)
ax[1].spines.right.set_visible(False)
ax[2].spines.left.set_visible(False)
ax[1].yaxis.set_ticks_position('none')
ax[2].yaxis.tick_right()

offs = 0.04

lower = 33.385
ax[0].set_xlim(lower, lower + offs)
low = np.argmin(np.abs(edges_us - lower))
high = np.argmin(np.abs(edges_us - lower - offs))
popt, pcov = curve_fit(gaussian, edges_us[low:high], this_histo[low:high], 
                       sigma=np.sqrt(np.where(this_histo[low:high] == 0, 1, this_histo[low:high])), 
                       p0=[500, lower + offs/2, 0.01])
FWHM = 2.35482 * abs(popt[2]) * 1e3
position = popt[1]
print(FWHM)
# ax[0].plot(edges_us[:-1], gaussian(edges_us[:-1], *popt))
text = ('$T\!oF = {:.3f}$'.format(round(position, 3)) + r' \textmu s' + 
        '\n$F\!W\!H\!M = {:.1f}$'.format(round(FWHM, 1)) + ' ns')
ax[0].text(0.42, 0.5, text, ha='left', va='center', transform=ax[0].transAxes)

lower = 33.793
ax[1].set_xlim(lower, lower + offs)
low = np.argmin(np.abs(edges_us - lower))
high = np.argmin(np.abs(edges_us - lower - offs))
popt, pcov = curve_fit(gaussian, edges_us[low:high], this_histo[low:high], 
                       sigma=np.sqrt(np.where(this_histo[low:high] == 0, 1, this_histo[low:high])), 
                       p0=[500, lower + offs/2, 0.01])
FWHM = 2.35482 * abs(popt[2]) * 1e3
position = popt[1]
print(FWHM)
# ax[1].plot(edges_us[:-1], gaussian(edges_us[:-1], *popt))
text = ('$T\!oF = {:.3f}$'.format(round(position, 3)) + r' \textmu s' + 
        '\n$F\!W\!H\!M = {:.1f}$'.format(round(FWHM, 1)) + ' ns')
ax[1].text(0.32, 0.2, text, ha='left', va='center', transform=ax[1].transAxes)

lower = 42.065
ax[2].set_xlim(lower, lower + offs)
low = np.argmin(np.abs(edges_us - lower))
high = np.argmin(np.abs(edges_us - lower - offs))
popt, pcov = curve_fit(gaussian, edges_us[low:high], this_histo[low:high], 
                       sigma=np.sqrt(np.where(this_histo[low:high] == 0, 1, this_histo[low:high])), 
                       p0=[500, lower + offs/2, 0.01])
FWHM = 2.35482 * abs(popt[2]) * 1e3
position = popt[1]
print(FWHM)
# ax[2].plot(edges_us[:-1], gaussian(edges_us[:-1], *popt))
text = ('$T\!oF = {:.3f}$'.format(round(position, 3)) + r' \textmu s' + 
        '\n$F\!W\!H\!M = {:.1f}$'.format(round(FWHM, 1)) + ' ns')
ax[2].text(0.3, 0.55, text, ha='left', va='center', transform=ax[2].transAxes)

ax[0].set_zorder(3)
ax[1].set_zorder(2)
ax[2].set_zorder(1)

text = '$\mathrm{^{85}Rb}$'
ax[0].text(0.21, 0.88, text, ha='left', va='center', transform=ax[0].transAxes)
text = '$\mathrm{^{87}Rb}$'
ax[1].text(0.13, 0.36, text, ha='left', va='center', transform=ax[1].transAxes)
text = '$\mathrm{^{133}Cs}$'
ax[2].text(0.08, 0.92, text, ha='left', va='center', transform=ax[2].transAxes)

ax[0].set_yticks([i*200 for i in range(7)])
ax[0].set_ylim([-50, 1150])

ax[0].set_ylabel('Counts per bin', size = 10)
ax[1].set_xlabel(r'Time-of-flight (\textmu s)')

d = 0.5
kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=8,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax[0].plot([1, 1], [0, 1], transform=ax[0].transAxes, **kwargs)
ax[1].plot([0, 0], [0, 1], transform=ax[1].transAxes, **kwargs)
ax[1].plot([1, 1], [0, 1], transform=ax[1].transAxes, **kwargs)
ax[2].plot([0, 0], [0, 1], transform=ax[2].transAxes, **kwargs)

plt.tight_layout(pad=0.5)
fig.subplots_adjust(hspace=0, wspace=0.03)

# %%

fig, ax = plt.subplots(figsize=(100/inch_to_mm,70/inch_to_mm))

vector_x = np.array([])
vector_y = np.array([])
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_x = np.append(vector_x, measurement_dict['xy'][0])
    vector_y = np.append(vector_y, measurement_dict['xy'][1])

histo, ex, ey = np.histogram2d(vector_x, vector_y,
                               bins = (500, 500),
                               range = [[0, 1], [0, 1]])

cax = ax.pcolormesh(ex, ey, histo.T, cmap=cmap, 
                    norm=mcolors.Normalize(vmin=0.001), rasterized=True)

ax.set_xlabel('Position $x$', size=10)
ax.set_ylabel('Position $y$', size=10)

plt.tight_layout(pad=0.5)

# %%

fig, ax = plt.subplots(figsize=(100/inch_to_mm,70/inch_to_mm))

vector_x = np.array([])
vector_y = np.array([])
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_x = np.append(vector_x, measurement_dict['xy'][0])
    vector_y = np.append(vector_y, measurement_dict['ToF_CFD'])

histo, ex, ey = np.histogram2d(vector_x, vector_y / 1e3,
                               bins = (500, 500),
                               range = [[0, 1], [33, 43]])

cax = ax.pcolormesh(ex, ey, histo.T, cmap=cmap, 
                    norm=mcolors.Normalize(vmin=0.001), rasterized=True)

ax.set_xlabel('Position $x$', size=10)
ax.set_ylabel('ToF', size=10)

plt.tight_layout(pad=0.5)

# %%

fig, ax = plt.subplots(figsize=(100/inch_to_mm,70/inch_to_mm))

vector_x = np.array([])
vector_y = np.array([])
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_x = np.append(vector_x, measurement_dict['ToF_CFD'])
    vector_y = np.append(vector_y, measurement_dict['xy'][1])

histo, ex, ey = np.histogram2d(vector_x / 1e3, vector_y,
                               bins = (500, 500),
                               range = [[33, 43], [0, 1]])

cax = ax.pcolormesh(ex, ey, histo.T, cmap=cmap, 
                    norm=mcolors.Normalize(vmin=0.001), rasterized=True)

ax.set_xlabel('ToF', size=10)
ax.set_ylabel('Position $y$', size=10)

plt.tight_layout(pad=0.5)


# %%

fig, ax = plt.subplots(4, 4, figsize=(120/inch_to_mm,100/inch_to_mm),
                       gridspec_kw={'width_ratios': [3, 1, 1, 1], 
                                    'height_ratios': [3, 1, 1, 1]},
                       sharex='col', sharey='row')

for i in range(3):
    for j in range(3):
        ax[i+1, j+1].set_visible(False)
        ax[i+1, j+1].set_axis_off()
        ax[i+1, j+1].set_zorder(0)

vector_x = np.array([])
vector_y = np.array([])
vector_ToF = np.array([])
for measurement_number, measurement_dict in dict_data_timing.items():
    vector_x = np.append(vector_x, measurement_dict['xy'][0])
    vector_y = np.append(vector_y, measurement_dict['xy'][1])
    vector_ToF = np.append(vector_ToF, measurement_dict['ToF_CFD'])

histo, ex, ey = np.histogram2d(vector_x, vector_y,
                               bins = (300, 300),
                               range = [[0, 1], [0, 1]])
# cax = ax[0, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
cax = ax[0, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)

histo, ex, ey = np.histogram2d(vector_x, vector_ToF / 1e3,
                               bins = (300, 5000),
                               range = [[0, 1], [33, 43]])
# cax = ax[1, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
# cax = ax[2, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
# cax = ax[3, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
cax = ax[1, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)
cax = ax[2, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)
cax = ax[3, 0].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)

histo, ex, ey = np.histogram2d(vector_ToF / 1e3, vector_y,
                               bins = (5000, 300),
                               range = [[33, 43], [0, 1]])
# cax = ax[0, 1].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
# cax = ax[0, 2].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
# cax = ax[0, 3].pcolormesh(ex, ey, histo.T, cmap=cmap, 
#                           norm=mcolors.Normalize(vmin=0.001), rasterized=True)
cax = ax[0, 1].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)
cax = ax[0, 2].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)
cax = ax[0, 3].pcolormesh(ex, ey, histo.T, cmap=cmap, 
                          norm=LogNorm(), rasterized=True)

ax[0, 0].set_xlim([0.3, 0.8])
ax[0, 0].set_ylim([0.25, 0.75])

offs = 0.04
lower = 33.385
ax[1, 0].set_ylim(lower, lower + offs)
ax[0, 1].set_xlim(lower, lower + offs)
lower = 33.793
ax[2, 0].set_ylim(lower, lower + offs)
ax[0, 2].set_xlim(lower, lower + offs)
lower = 42.065
ax[3, 0].set_ylim(lower, lower + offs)
ax[0, 3].set_xlim(lower, lower + offs)

ax[3, 0].set_xlabel('Position x')
ax[0, 0].set_ylabel('Position y')
ax[0, 2].set_xlabel('ToF (us)')
ax[2, 0].set_ylabel('ToF (us)')

ax[0, 1].xaxis.set_tick_params(labelbottom=True)
ax[0, 2].xaxis.set_tick_params(labelbottom=True)
ax[0, 3].xaxis.set_tick_params(labelbottom=True)

ax[1, 0].invert_yaxis()
ax[2, 0].invert_yaxis()
ax[3, 0].invert_yaxis()

plt.tight_layout(pad=0.5)
fig.subplots_adjust(hspace=0, wspace=0)

# save_name = 'ToF_position_CsRb'
# plt.savefig(f'figures\\{save_name}.jpg', dpi=300)
# plt.savefig(f'figures\\{save_name}.pdf')
