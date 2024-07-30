# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:58:17 2024

@author: Elias
"""

# =============================================================================
# RUN THIS ONCE TO CHANGE TO THE LaTeX FONT !!!
# =============================================================================

# why is it so damn difficult to change the font
# finally found a way to do it...

# installed fonts are in:
# C:\Users\Elias\AppData\Local\Microsoft\Windows\Fonts

# edit rc file in:
# C:\Program Files\Spyder\pkgs\matplotlib\mpl-data

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib as mpl

# Find the path to your installed TTF font file
font_path = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# Choose the font file you want to use
# font_path = font_path[0]  # Choose the first font file, for example

# Create a FontProperties object with the font file
font_name = r"C:\Users\Elias\AppData\Local\Microsoft\Windows\Fonts\cmunrm.ttf"
font_name_math = r"C:\Users\Elias\AppData\Local\Microsoft\Windows\Fonts\cmunti.ttf"
# custom_font = font_manager.FontProperties(fname=font_name)

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=font_name).get_name()]

# rcParams['mathtext.fontset'] = 'cm'
# rcParams['mathtext.it'] = font_manager.FontProperties(fname=font_name_math).get_name()

plt.rcParams['text.usetex'] = True

params = {'text.latex.preamble': r'\usepackage{textcomp}'}   
plt.rcParams.update(params) 

# Use the custom font in your plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel(r'X-axis $a+bx$ (\textmu)', fontsize=12)
plt.ylabel(r'Y-axis $\alpha$', fontsize=12)
plt.title('Custom Font Plot', fontsize=12)
plt.text(1, 5, r'$\displaystyle\int_0^\infty e^{-x}\,dx$', fontsize=12)
plt.show()

# reset the hard work you just did
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)


