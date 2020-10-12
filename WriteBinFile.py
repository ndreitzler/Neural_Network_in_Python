#File to create 2 simple binary files in the form of the MNIST handwritten dataset
#Nick Dreitzler

import struct

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Create input file
magic_number = b'\x00\x00\x08\x03'
num_items = 4
rows = 2
cols = 2

file = open('testin.bin','wb')

#Write Magic number
file.write(magic_number)

#Write number of 'images'
file.write(bytes([num_items,rows,cols]))

#Write data 
file.write(struct.pack('BBBB',0,0,0,1))
file.write(struct.pack('BBBB',1,0,1,1))
file.write(struct.pack('BBBB',0,0,1,1))
file.write(struct.pack('BBBB',1,0,0,1))

file.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Create target output file
file = open('testout.bin','wb')

magic_number = b'\x00\x00\x08\x01'

#Write Magic number
file.write(magic_number)

#Write number of items to target output file
file.write(bytes([num_items]))

#Write Desired outputs
file.write(struct.pack('BBBB',0,2,3,4))

file.close()



#The magic number is an integer (MSB first). The first 2 bytes are always 0.
#
#The third byte codes the type of the data:
#0x08: unsigned byte
#0x09: signed byte
#0x0B: short (2 bytes)
#0x0C: int (4 bytes)
#0x0D: float (4 bytes)
#0x0E: double (8 bytes)
#
#The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
#
#The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
#
#The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.