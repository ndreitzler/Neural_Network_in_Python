from PIL import Image

img = Image.new('L',(28,28), color = 255)

file = open('train-images.idx3-ubyte','rb')

#np.fromfile(file, dtype=int, count=4, sep="")
print(int.from_bytes(file.read(4),'big'))
print(int.from_bytes(file.read(4),'big'))
print(int.from_bytes(file.read(4),'big'))
print(int.from_bytes(file.read(4),'big'))


#text = file.read(784)
for x in range(0,28):
    for y in range(0,28):
        img.putpixel((y,x), abs(int.from_bytes(file.read(1),"big") - 255))
        img.save('pil_black.png')
        
for x in range(0,28):
    for y in range(0,28):
        img.putpixel((y,x),  abs(int.from_bytes(file.read(1),"big") - 255))
        img.save('pil_black2.png')
        
for x in range(0,28):
    for y in range(0,28):
        img.putpixel((y,x),  abs(int.from_bytes(file.read(1),"big") - 255))
        img.save('pil_black3.png')
        
file.close()