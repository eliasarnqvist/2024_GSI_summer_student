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

file_1 = r'20240726_002_DI2_Cs_overweekend_500T.txt'
filepath_1 = data_folder + "\\" + file_1
col_names = ['MS (s)', 'time (us)', 'mass (u)']
df_1 = pd.read_csv(filepath_1, skiprows=13, delim_whitespace=True, names=col_names)

#%%

masses = df_1['mass (u)'].to_numpy()
hist, bin_edges = np.histogram(masses, bins=1000)

# %%

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
ax.step(bin_edges[:-1], hist, where='post', lw=1)
ax.set_xlabel('Mass (a.u.)')
plt.tight_layout(pad=0.5)

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
ax.step(df_1.index.to_numpy(), df_1['mass (u)'], where='post', lw=1)
plt.tight_layout(pad=0.5)
ax.set_xlabel('Time (?)')
ax.set_ylabel('Mass (a.u.)')


