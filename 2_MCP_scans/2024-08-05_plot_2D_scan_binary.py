# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct

# =============================================================================
# Import data
# =============================================================================

data_folder = r'C:\Users\Elias\Work\Skolarbete universitet\2024xS GSI\Data\2D MCP'

file_1 = r'MCP data 2023\20231113_001_Cs_ions_after_optmising_sled_position.mp'
filepath_1 = data_folder + "\\" + file_1
# col_names = ['x', 'y', 'counts']
# df_1 = pd.read_csv(filepath_1, skiprows=34, delim_whitespace=True, names=col_names)

with open(filepath_1, 'rb') as file:
    
    for _ in range(21):
        file.readline()
    
    test = struct.unpack('l', file.read(4))[0]


#%%

x_len = 1024
y_len = 1024

xx = np.arange(0, x_len, 1) - 0.5
yy = np.arange(0, y_len, 1) - 0.5

zz = np.zeros([x_len, y_len])

for i, row in df_1.iterrows():
    this_x = row['x']
    this_y = row['y']
    this_counts = row['counts']
    zz[this_y, this_x] = this_counts

# coarse (128x128)
coarse_size = 128
xx_coarse = np.arange(0, coarse_size, 1) - 0.5
yy_coarse = np.arange(0, coarse_size, 1) - 0.5

div = int(x_len / coarse_size)

zz_coarse = zz.reshape(128, 8, 128, 8).sum(axis=(1, 3))



# %%

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

cmap = plt.cm.viridis
cmap.set_under('white')



# fig, ax = plt.subplots(figsize=(110/inch_to_mm,70/inch_to_mm))
fig, ax = plt.subplots(figsize=(73/inch_to_mm,70/inch_to_mm))

# cax = ax.pcolormesh(xx, yy, zz, cmap=cmap, 
#                     norm=mcolors.Normalize(vmin=0.001), rasterized=True)
cax = ax.pcolormesh(xx_coarse, yy_coarse, zz_coarse, cmap=cmap, 
                    norm=mcolors.Normalize(vmin=0.001), rasterized=True)

# cbar = fig.colorbar(cax, location='right', pad=0.05)
# cbar.set_label('Counts per bin')
# ax.step(bin_edges[:-1], hist, where='post', lw=1)
ax.set_xlabel('x', size=12)
ax.set_ylabel('y', size=12)
plt.tight_layout(pad=0.5)

save_name = 'MCP_2D_scan_test'
plt.savefig(f'figures\\{save_name}.jpg')
plt.savefig(f'figures\\{save_name}.pdf')


