#Import Libraries
import numpy as np
import struct

#%%
byte_order = 'big'

file = open('testin.bin','rb')

#Discard first 4 bytes, ie the magic number 
#Assumes magic number will always be b'\x00\x00\x08\x03'
file.seek(4)

#get the number of items, number of rows, and number of columns from the file
num_items = int.from_bytes(file.read(1), byte_order)
rows = int.from_bytes(file.read(1), byte_order)
cols = int.from_bytes(file.read(1), byte_order)

total_bytes = num_items*rows*cols

data = file.read(total_bytes)

#Assumes data will always be an unsigned byte, ie the 3rd byte of the magic number is always \x08
input_figures = struct.iter_unpack("%dB" % (rows*cols), data)
input_figures = np.array([*input_figures])

#print(input_figures)
           
file.close()


#%%

output_nodes = 5
file = open('testout.bin','rb')

#Discard first 4 bytes, ie the magic number 
#Assumes magic number will always be b'\x00\x00\x08\x01'
file.seek(4)

#get the number of items from the file
num_items_out = int.from_bytes(file.read(1), byte_order)

if(num_items_out != num_items):
    print("Item sizes do not match")
    exit(-1)

data = file.read(num_items)
target_numbers = struct.unpack("%dB" % (num_items), data)
target_numbers = np.array([*target_numbers])
output_figures = np.zeros((num_items,output_nodes),np.uint8)
for i in range(num_items):
    output_figures[i][target_numbers[i]] = 1
    
#print(output_figures )

file.close()
