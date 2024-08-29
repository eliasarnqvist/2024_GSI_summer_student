# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =============================================================================
# Import flip data
# =============================================================================

filename = 'coin_flip_data.txt'
with open(filename, 'r') as file:
    content = file.read()

# Measured
measured_heads = content.count('h')
measured_tails = content.count('t')
measured_tosses = measured_heads + measured_tails
measured_frac = measured_tails / measured_tosses
measured_error = np.sqrt(measured_tails) / measured_tosses

# Expected
expected_frac = 0.5
expected_tails = expected_frac * measured_tosses
expected_error = np.sqrt(expected_tails) / measured_tosses

# =============================================================================
# Make a plot
# =============================================================================

plt.close('all')

inch_to_mm = 25.4
colors = plt.cm.tab10

fig, ax = plt.subplots(figsize=(73/inch_to_mm, 60/inch_to_mm))
x = ['Summer student\ntechnique', 'Conventional\nmethod']
y = [measured_frac, expected_frac]
err = [measured_error, expected_error]

ax.plot([-1, 2], [measured_frac, measured_frac], 'gray', ls='--', lw=1)
ax.plot([-1, 2], [expected_frac, expected_frac], 'gray', ls='--', lw=1)
ax.errorbar(x, y, yerr=err, fmt='.', capsize=3, capthick=1, markersize=8)

ax.set_xlim(-0.5, len(x) - 0.5)
ax.set_ylim([0, 1])

ax.text(0.6, 0.55, r'$P\approx0.5$', transform=ax.transAxes, ha='center')
ax.text(0.45, 0.85, f'$P={measured_frac:.3f}$', transform=ax.transAxes, ha='center')
# ax.text(0.1, 0.1, f'Number of tosses: {measured_tosses}', transform=ax.transAxes, ha='left')

ax.set_ylabel('Probability', size=10)
# ax.set_xlabel('Method')

plt.tight_layout(pad=0.5)

save_name = 'coin_flip_results'
plt.savefig(f'figures\\{save_name}.jpg', dpi=300)
plt.savefig(f'figures\\{save_name}.pdf')


