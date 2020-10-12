# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:48:50 2020

@author: Nick Dreitzler

Purpose Reads weight values from the weights file
"""
import struct
import numpy as np

byte_order = 'little'
size_float = 8

file = open("weights.bin","rb")

#Read in metadata
pixals_in_image     = int.from_bytes(file.read(4),byte_order)
num_hidden_nodes    = int.from_bytes(file.read(4),byte_order)
output_nodes        = int.from_bytes(file.read(4),byte_order)

#Read in weight_hidden
buffer = file.read(pixals_in_image*num_hidden_nodes*size_float)
weight_hidden = struct.iter_unpack('d'*num_hidden_nodes, buffer)
weight_hidden = np.array([*weight_hidden])

#Read in weight_output
buffer = file.read(output_nodes*num_hidden_nodes*size_float)
weight_output = struct.iter_unpack('d'*output_nodes, buffer)
weight_output = np.array([*weight_output])


print("weight_hidden")
print(weight_hidden)
print()
print("weight_output")
print(weight_output)
print()

file.close()
