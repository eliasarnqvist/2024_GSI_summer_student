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

file_1 = r'2024-08-28_ABE_and_CD\\tek0012ALL.csv'
filepath_1 = data_folder + "\\" + file_1
# col_names = ['t', 'V']
col_names = ['t', 'V1', 'V2', 'V3']
df_1 = pd.read_csv(filepath_1, skiprows=21, names=col_names)

file_1 = r'2024-08-28_ABE_and_CD\\tek0013ALL.csv'
filepath_1 = data_folder + "\\" + file_1
# col_names = ['t', 'V']
col_names = ['t', 'V1', 'V2', 'V3']
df_2 = pd.read_csv(filepath_1, skiprows=21, names=col_names)

# =============================================================================
# Plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(73/inch_to_mm, 55/inch_to_mm))
ax.plot(df_1['t']*1e6 + 0.26, df_1['V1'] * 1e3, label='A', lw=1)
ax.plot(df_1['t']*1e6 + 0.26, df_1['V2'] * 1e3, label='B', lw=1)
ax.plot(df_2['t']*1e6 + 0.26, df_2['V1'] * 1e3, label='C', lw=1) # actually A...
ax.plot(df_2['t']*1e6 + 0.26, df_2['V2'] * 1e3, label='D', lw=1) # actually B...
ax.plot(df_1['t']*1e6 + 0.26, df_1['V3'] * 1e3, label='E', lw=1)
ax.set_yticks(ticks=np.arange(-800, 700, 200))
ax.tick_params(axis='y', labelsize=10)
ax.set_xticks(ticks=np.arange(0, 3.5, 0.5))
ax.tick_params(axis='x', labelsize=10)
ax.set_xlim([-0.2, 2.7])
ax.set_ylim([-900, 700])
ax.set_xlabel(r'Time (\textmu s)', size=10)
ax.set_ylabel('Voltage (mV)', size=10)

ax.legend(frameon=False, loc='upper right', ncols=2, fontsize=10, 
          handlelength=1, handletextpad=0.5, columnspacing=0.5)

plt.tight_layout(pad=0.5)

save_name = 'oscilloscope_preamp_ABCDE'
plt.savefig(f'figures\\{save_name}.jpg', dpi=300)
plt.savefig(f'figures\\{save_name}.pdf')


