#Import Libraries
import numpy as np

#Define input features
#input_features = np.array([[1,0,0,1],[1,0,0,0],[0,0,1,1],
#                           [0,1,0,0],[1,1,0,0],[0,0,1,1],
#                           [0,0,0,1],[0,0,1,0]])
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])

#print (input_features.shape)
#print(input_features)

#define target output
#target_output = np.array([[1,1,0,0,1,1,0,0]])
#target_output = target_output.reshape(8,1)
target_output = np.array([[0,1,1,0]])
target_output = target_output.reshape(4,1)
#print(target_output.shape)
#print(target_output)

#define weights
weight_hidden = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
weight_output = np.array([[0.7],[0.8],[0.9]])
#print(weights.shape)
#print(weights)

#bias weight
#bias = 0.3

#learning rate
lr = 0.05


#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivative of sigmoid function
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

#Main Logic for neural network
#Running our code 10000 times
    
for epoch in range(200000):
    #input for hidden layer
    input_hidden = np.dot(input_features, weight_hidden)
        
    #output from hidden layer
    output_hidden = sigmoid(input_hidden)
        
    #Input for output layer
    input_op = np.dot(output_hidden, weight_output)
        
    #Output for output layer
    output_op = sigmoid(input_op)
    
    #Phase 1#############################################################
    #Calculateing error
    error_out = np.power((output_op-target_output),2) / 2
    #print(error_out.sum())
    
    
    #Calculating derivative:
    derror_douto = output_op - target_output
    douto_dino = sigmoid_der(input_op)
    dino_dwo = output_hidden
    
    derror_dwo = np.dot(dino_dwo.T, derror_douto * douto_dino)
    
    #Phase 2#############################################################
    #derivatives for phase 2
    derror_dino = derror_douto * douto_dino
    dino_douth = weight_output
    derror_douth = np.dot(derror_dino, dino_douth.T)
    douth_dinh = sigmoid_der(input_hidden)
    dinh_dwh = input_features
    derror_wh = np.dot(dinh_dwh.T, douth_dinh * derror_douth)
    
    #update weights
    weight_hidden -= lr * derror_wh
    weight_output -= lr * derror_dwo
            
        
print(weight_hidden)
print(weight_output)
print(" ")

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