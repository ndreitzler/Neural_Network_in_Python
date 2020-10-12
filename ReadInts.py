

file = open('train-labels.idx1-ubyte','rb')

print(int.from_bytes(file.read(4),'big'))
print(int.from_bytes(file.read(4),'big'))

print(int.from_bytes(file.read(1),'big'))
print(int.from_bytes(file.read(1),'big'))
print(int.from_bytes(file.read(1),'big'))

file.close()