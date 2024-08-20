# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# Import data
# =============================================================================

data_folder = r'data'

file_1 = r'2024-08-15_data_chA_find_tot_relation\\tek0007ALL.csv'
filepath_1 = data_folder + "\\" + file_1
# col_names = ['t', 'V']
col_names = ['t', 'V1', 'V2']
df_1 = pd.read_csv(filepath_1, skiprows=21, names=col_names)

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
ax.plot(df_1['t'], df_1['V1'])
ax.plot(df_1['t'], df_1['V2'])
ax.set_xlim([-2e-6, 4e-6])
plt.tight_layout(pad=0.5)


