#Sample Neural Network with one hidden layer
#Main logic of neural network was created following a guide created by Pratik Shukla and Roberto Iriondo
#Guide used: Building Neural Networks with Python Code and Math in Detail — II

#Import Libraries
import numpy as np
import struct
import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Read from target outputs
byte_order = 'big'

file = open('train-images.idx3-ubyte','rb')

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
input_figures = struct.iter_unpack("%dB" % (pixals_in_image), data)
input_figures = np.array([*input_figures])
            
file.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Read from target outputs
output_nodes = 10 #Because we need a node for each 0-9
file = open('train-labels.idx1-ubyte','rb')

#Discard first 4 bytes, ie the magic number 
#Assumes magic number will always be b'\x00\x00\x08\x01'
file.seek(4)

#get the number of items from the file
num_items_out = int.from_bytes(file.read(4), byte_order)

if(num_items_out != num_items):
    print("Item sizes do not match")
    exit(-1)

data = file.read(num_items)
target_numbers = struct.unpack("%dB" % (num_items), data)
target_numbers = np.array([*target_numbers])


#Convert target output ints into one hot encoding
output_figures = np.zeros((num_items,output_nodes),np.uint8)
for i in range(num_items):
    output_figures[i][target_numbers[i]] = 1
    
#print(output_figures )

file.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Set weights and bias

num_hidden_nodes = 16 #Arbitray
#Generate random weights for hidden layer, creates a pixals_in_image x num_hidden_nodes matrix
weight_hidden = np.random.rand(pixals_in_image,num_hidden_nodes)
#generate randome weights for output layer, create a num_hidden_nodes matrix x 1 matrix
weight_output = np.random.rand(num_hidden_nodes,output_nodes)

#bias weight
#bias = 0.3

#learning rate
lr = 0.005


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Main Logic for neural network
#Running our code 10000 times

time_taken = -time.time()

for epoch in range(10000):
    #input for hidden layer
    input_hidden = np.dot(input_figures, weight_hidden)
        
    #output from hidden layer
    output_hidden = sigmoid(input_hidden)
        
    #Input for output layer
    input_op = np.dot(output_hidden, weight_output)
        
    #Output for output layer
    output_op = sigmoid(input_op)
    
    #Phase 1#############################################################
    #Calculateing error
    error_out = np.power((output_op-output_figures),2) / 2
    #print(error_out.sum())
    
    
    #Calculating derivative:
    derror_douto = output_op - output_figures
    douto_dino = sigmoid_der(input_op)
    dino_dwo = output_hidden
    
    derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)
    
    #Phase 2#############################################################
    #derivatives for phase 2
    derror_dino = derror_douto * douto_dino
    dino_douth = weight_output
    derror_douth = np.dot(derror_dino, dino_douth.T)
    douth_dinh = sigmoid_der(input_hidden)
    dinh_dwh = input_figures
    derror_wh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)
    
    #update weights
    weight_hidden -= lr * derror_wh
    weight_output -= lr * derror_dwo

time_taken += time.time()
print("This took %d second", time_taken)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
#print weights
'''
print("weight_hidden")
print(weight_hidden)
print()
print("weight_output")
print(weight_output)
print()
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#save weights
#init buffer   
buffer = np.zeros(3, np.uint32)
struct.pack_into('iii',buffer,0,pixals_in_image,num_hidden_nodes,output_nodes)

file = open("weights.bin","wb")

file.write(buffer)

for i in range(pixals_in_image):
    file.write(struct.pack('d'*num_hidden_nodes, *(weight_hidden)[i]))

for i in range(num_hidden_nodes):
    file.write(struct.pack('d'*output_nodes, *(weight_output)[i]))

file.close();

#%%

#print(input_figures)

#print(output_figures)

#%%
'''
#Testing

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


#taking Inputs
single_point = np.array([0,1])

#1st step 
result1 = np.dot(single_point, weight_hidden)

#2nd step
result2 = sigmoid(result1)

#3rd step
result3 = np.dot(result2, weight_output)
result4 = sigmoid(result3)
print(result4)


#taking Inputs
single_point = np.array([1,0])

#1st step 
result1 = np.dot(single_point, weight_hidden)

#2nd step
result2 = sigmoid(result1)

#3rd step
result3 = np.dot(result2, weight_output)
result4 = sigmoid(result3)
print(result4)

#taking Inputs
single_point = np.array([1,1])

#1st step 
result1 = np.dot(single_point, weight_hidden)

#2nd step
result2 = sigmoid(result1)

#3rd step
result3 = np.dot(result2, weight_output)
result4 = sigmoid(result3)
print(result4)
'''
