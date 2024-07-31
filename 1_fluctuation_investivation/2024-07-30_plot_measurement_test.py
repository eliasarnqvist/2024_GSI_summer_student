# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Import data
# =============================================================================

data_folder = r'C:\Users\Elias\Work\Skolarbete universitet\2024xS GSI\Data\TOF_fluctation_check'

file_1 = r'20240726_002_VC_logging_overweekend.csv'
filepath_1 = data_folder + "\\" + file_1
df_1 = pd.read_csv(filepath_1, delimiter=";")

# Calculate the seconds after the first timestamp
df_1['TimeStamp'] = pd.to_datetime(df_1['TimeStamp'], format='mixed')
df_1['Seconds_After_First'] = (df_1['TimeStamp'] - df_1['TimeStamp'].iloc[0]).dt.total_seconds()

# Revove the big jump in collected data, where it did not record
df_1 = df_1[:2909]

# Do Fourier transform
time = df_1['Seconds_After_First'].to_numpy()
amplitude = df_1['MgnTof_sml_HVp_U'].to_numpy()
fft_values = np.fft.fft(amplitude)
fft_freq = np.fft.fftfreq(len(time), time[1] - time[0])

# =============================================================================
# Plot data
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
ax.plot(df_1['Seconds_After_First'], df_1['MgnTof_sml_HVp_U'], lw=1)
plt.tight_layout(pad=0.5)




fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
x = fft_freq
y = fft_values.real
sorting = np.argsort(x)
x = x[sorting]
y = y[sorting]
ax.step(x, y, lw=1, where='mid')
# ax.step(fft_freq, fft_values.imag, lw=1, where='mid')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)')
plt.tight_layout(pad=0.5)


# =============================================================================
# FFT everything
# =============================================================================

# min_index = 15
# max_index = min_index + 2
this_index = 16

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(110/inch_to_mm,140/inch_to_mm))
for index, column in enumerate(df_1.columns):
    # if index > min_index and index < max_index:
    if index == this_index:
        col_amplitude = df_1[column].to_numpy()
        fft_values = np.fft.fft(col_amplitude)
        
        y = np.sqrt(fft_values.real**2 + fft_values.imag**2)
        # y = fft_values.real
        y = y[sorting]
        # ax.step(x, y, lw=1, where='mid')
        ax1.plot(x, y, lw=1)
        
        ax2.plot(df_1['Seconds_After_First'], df_1[column], lw=1)
        
        print(column)
        
ax1.set_yscale('log')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Fourier transform magnitude')
ax1.set_xlim([-1e-3, max(x)])
ax1.set_ylim([1e-3, 1e6])

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude (?)')

plt.tight_layout(pad=0.5)



"""
1 - signal, no osc
4 - something at 0.06 Hz
7 - small at 0.06
8 - small at 0.06
10 - small at 0.06
11 - small at 0.06
14 - small at 0.06
16 - something at 0.003 (!) - Ref_E6_U
19 - again 0.003 - Ref_E7_U
23 - 0.06
25 - 0.06
26 - 0.06
40 - very small at 0.003
58 - discontinuity butno oscillations - Li_Def_4_U
67 - noise decreases after a while strangely - Ref_E3_U

"""








