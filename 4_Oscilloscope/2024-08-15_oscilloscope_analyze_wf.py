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
ax.plot(df_1['t']*1e6, df_1['V1'], label='A')
ax.plot(df_1['t']*1e6, df_1['V2'], label='E')
ax.set_xlim([-1, 3])
ax.set_xlabel('Time (us)')
ax.set_ylabel('Voltage (V)')
ax.legend(frameon=False)
plt.tight_layout(pad=0.5)

save_name = 'oscilloscope_preamp_A_E'
plt.savefig(f'figures\\{save_name}.jpg', dpi=300)
plt.savefig(f'figures\\{save_name}.pdf')


