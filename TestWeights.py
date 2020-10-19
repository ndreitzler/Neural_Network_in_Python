# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 18:48:18 2020

@author: dreit
"""

import struct
import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Read weights from file
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

#Read bias_hidden
buffer = file.read(num_hidden_nodes*size_float)
bias_hidden = struct.unpack('d'*num_hidden_nodes, buffer)
bias_hidden = np.array([*bias_hidden])

#Read in weight_output
buffer = file.read(output_nodes*num_hidden_nodes*size_float)
weight_output = struct.iter_unpack('d'*output_nodes, buffer)
weight_output = np.array([*weight_output])

#Read bias_output
buffer = file.read(output_nodes*num_hidden_nodes*size_float)
bias_output = struct.unpack('d'*output_nodes, buffer)
bias_output = np.array([*bias_output])

file.close();

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Read in Test Figures

byte_order = 'big'

file = open('t10k-images.idx3-ubyte','rb')

#Discard first 4 bytes, ie the magic number 
#Assumes magic number will always be b'\x00\x00\x08\x03'
file.seek(4)

#get the number of items, number of rows, and number of columns from the file
num_items = int.from_bytes(file.read(4), byte_order)
rows = int.from_bytes(file.read(4), byte_order)
cols = int.from_bytes(file.read(4), byte_order)

pixals_in_image = rows*cols

data = file.read(pixals_in_image*num_items)

#Assumes data will always be an unsigned byte, ie the 3rd byte of the magic number is always \x08
test_figures = struct.iter_unpack("%dB" % (pixals_in_image), data)
test_figures = np.array([*test_figures])

file.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Read in Test Numbers

output_nodes = 10 #Because we need a node for each 0-9
file = open('t10k-labels.idx1-ubyte','rb')

#Discard first 4 bytes, ie the magic number 
#Assumes magic number will always be b'\x00\x00\x08\x01'
file.seek(4)

#get the number of items from the file
num_items_out = int.from_bytes(file.read(4), byte_order)

if(num_items_out != num_items):
    print("Item sizes do not match")
    exit(-1)

data = file.read(num_items)
test_numbers = struct.unpack("%dB" % (num_items), data)
test_numbers = np.array([*test_numbers])
    
file.close()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Run Test

correct = 0
for i in range(10):
    #print(test_figures[i])
    r1 = np.dot(test_figures[i], weight_hidden) + bias_hidden
    #print(r1)
    r2 = sigmoid(r1)
    #print(r2)
    r3 = np.dot(r2, weight_output) + bias_output
    #print(r3)
    r4 = sigmoid(r3)    
    #print(r4)

    guess = 0
    max_val = r4[0]
    for k in range(1,output_nodes):
        if(r4[k] > max_val):
            guess = k
            max_val = r4[k]
    print(guess)
    if(guess == test_numbers[i]):
        correct += 1
     
    #print(r4, guess) 
    
percent_correct = correct / num_items

print('Num Items: \t', num_items)
print('Correct: \t', correct)
print('Percent: \t', percent_correct, ' %')     
    
'''
#taking Inputs
single_point = np.array([0,0])

#1st step 
result1 = np.dot(single_point, weight_hidden)

#2nd step
result2 = sigmoid(result1)

#3rd step
result3 = np.dot(result2, weight_output)
result4 = sigmoid(result3)
print(result4)
'''