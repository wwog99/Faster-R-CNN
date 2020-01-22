import cv2
import torch
import numpy as np
"""
image treatment
1. get resize ratio of each image => we will use it when resizing the box
2. resize image and save it(name will be processed_img_*)
"""
image_path = "test/img_"
imageset = np.array([])
resize_image = (512,512)
resize_x = []
resize_y = []
for i in range(6):
    image = cv2.imread(image_path+str(876+i)+".jpg")
    resize_x.append(resize_image[0]/image.shape[1])
    resize_y.append(resize_image[1]/image.shape[0])
    ###resize by ratio###
    image = cv2.resize(image, (0,0), fx=(resize_x[i]), fy=resize_y[i], interpolation=cv2.INTER_LINEAR_EXACT)
    np.save("processed/processed_img_"+str(i+1),image)  # save image as npy

"""
label treatment
1. get text file and save it
2. translate into bounding box coordination form (x1,y1,x2,y2)
   > we should resize the box 
"""
#1. get text file
label = []
text_path = "test/gt_img_"
for i in range(6):
    text = []
    file = open(text_path+str(876+i)+".txt")
    #read file from start to end
    while(1):
        line = file.readline()

        try:escape=line.index('\n')
        except:escape=len(line)
        if line: text.append(line[0:escape])
        else:
            break
    #text has [string, string, string, ...]
    file.close()
    #label has [text1, text2, ...]
    label.append(text)

#2. translate text into bounding box coordination
for i in range(len(label)):
    for j in range(len(label[i])):
        split_string = label[i][j].split(',')
        x1 = int(int(split_string[0])*resize_x[i])
        y1 = int(int(split_string[1])*resize_y[i])
        x2 = int(int(split_string[4])*resize_x[i])
        y2 = int(int(split_string[5])*resize_y[i])
        coord = [x1,y1,x2,y2]
        label[i][j] = coord

label = np.asarray(label)
print("label is :", label)
np.save("processed/processed_label", label) #save label as npy
#"""

"""
Test for resizing
>it will test we resize bounding box and image corretly.
>it will save the image which is resized and resized box on it
>image file would be boxed_img_*
"""
pro_path = "processed/processed_img_"
for i in range(6):
    image = np.load(pro_path+str(1+i)+".npy")
    for j in range(len(label[i])):
        image = cv2.rectangle(image, (label[i][j][0],label[i][j][1]), (label[i][j][2],label[i][j][3]), (255,255,255), 3)
    cv2.imwrite("processed/boxed_img_"+str(i+1)+".jpg", image)
    #print("after reshape: ",image.shape)
#"""