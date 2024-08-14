
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

data_folder = r'data'
file_1 = r'2024-08-14_TEST_AE_sweep-binary.lst'
filepath_1 = data_folder + "\\" + file_1

with open(filepath_1, 'rb') as file:
    
    for line in file:
        if line.strip() == b'[DATA]':
            break
    
    for i in range(15):
        # value = struct.unpack('I', file.read(4))[0]
        value = file.read(4)
        print(value)
        
# %%

def byte_to_bit(byte_data):
    bits_data = bin(int.from_bytes(byte_data, byteorder='big'))[2:]
    return bits_data

testbyte = b'\x06\x00 \x00'
value = struct.unpack('I', testbyte)[0]
print(value)








