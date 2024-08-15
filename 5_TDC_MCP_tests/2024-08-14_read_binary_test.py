
# -*- coding: utf-8 -*-

# Elias Arnqvist
# GSI, summer 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct
import pickle

# =============================================================================
# Import data
# =============================================================================

data_folder = r'data'
file_1 = r'2024-08-14_TEST-2_ABCDE_sweep-binary.lst'
filepath_1 = data_folder + "\\" + file_1

dict_event = {}

dict_channel = {0:'Zero', 1:'Counter real/live time', 2:'Counter data?',
                3:'Unused', 4:'Unused', 5:'Unused', 6:'START',
                7:'Reserved', 8:'STOP_1', 9:'STOP_2', 10:'STOP_3',
                11:'STOP_4', 12:'STOP_5', 13:'STOP_6', 14:'STOP_7',
                15:'STOP_8'}
dict_edge = {0:'RISING', 1:'FALLING'}


with open(filepath_1, 'rb') as file:
    
    # Skip to the binary data
    for line in file:
        if line.strip() == b'[DATA]':
            break
    
    while True:
        # For "time_patch=1c" data is stored in 4 byte chunks
        byte_data = file.read(4)
        
        if len(byte_data) < 4:
            break
        
        value = struct.unpack('<I', byte_data)[0]
        if value != 0:
            bits_data = bin(value)[2:].zfill(32)
            # print(bits_data)
        
            channel = int(bits_data[28:32], 2)
            channel_word = dict_channel[channel]
            
            edge = int(bits_data[27])
            edge_word = dict_edge[edge]
            
            time = int(bits_data[11:27], 2)
            # Is time in units of time bins (8 * 80 ps) or 80 ps?
            
            sweep = int(bits_data[1:11], 2)
            
            loss = bits_data[0]
            
            print(sweep, channel_word, edge_word, time, loss)
            
            dict_ch = {'sweep':sweep, 'ch':channel, 'ch_w':channel_word, 
                       'edge':edge, 'edge_w':edge_word, 'time':time, 
                       'loss':loss}
            if sweep not in dict_event:
                dict_event[sweep] = {}
            
            dict_event[sweep][time] = dict_ch

save_name = 'extracted_data\\' + file_1[:-4] + '_extracted.pkl'
with open(save_name, 'wb') as file:
    pickle.dump(dict_event, file)


