import csv
import os
from os import rename

count = 0
path = './data/total_Img/'

image_Name = []
image_Label = []

image_List = os.listdir(path)

total_Image_Info_File = open('./data/Total_File_Infor.csv', 'r', encoding='utf-8')
total_Image_Info = csv.reader(total_Image_Info_File)

for info in total_Image_Info:
    image_Name.append(info[0].replace('.vir', ''))
    image_Label.append(info[1])


for image in image_List:
    for i in range(0, len(image_Name)):
        if image.replace('.jpg', '') == image_Name[i]:
            if image_Label[i] == '1':
                rename(path + image, path + 'M' + str(count) + '.jpg')
            else:
                rename(path + image, path + 'B' + str(count) + '.jpg')
            count += 1