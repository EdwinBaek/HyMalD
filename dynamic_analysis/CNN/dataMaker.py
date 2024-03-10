import os
import csv
from os import rename
import shutil

def get_ExtractedFile_Data():
    extracted_Image_List = os.listdir('./data/total_Img/')
    image_Name = []
    image_Data = []
    countOfIndex = 1
    numberOfMalware = 0
    numberOfBenign = 0

    total_Image_Data_File = open('./data/labels/total_Image_Info.csv', 'r', encoding='utf-8')
    total_Image_Data = csv.reader(total_Image_Data_File)

    for name in extracted_Image_List:
        image_Name.append(name.split('.jpg')[0])

    print('image_Name size : ' + str(len(image_Name)))

    for data in total_Image_Data:
        if data[0].split('.vir')[0] in image_Name:
            image_Data.append([data[0].split('.vir')[0], data[1], countOfIndex])
            countOfIndex += 1
            if data[1] == '0':
                numberOfBenign += 1
            else:
                numberOfMalware += 1

    print('image_Data size : ' + str(len(image_Data)))
    print('Number of Malware : ' + str(numberOfMalware))
    print('Number of Benign : ' + str(numberOfBenign))

    extracted_Image_Data_File = open('./data/labels/128_extracted_Image_Info.csv', 'w', newline='', encoding='utf-8')
    data_Writer = csv.writer(extracted_Image_Data_File)

    for data in image_Data:
        data_Writer.writerow(data)

    total_Image_Data_File.close()
    extracted_Image_Data_File.close()
    print('>>>>>>>>>> Successful creation of the extracted image data file in csv format')

def rename_Move(old_File_Name, new_File_Name, old_Path, new_Path):
    rename(old_File_Name, new_File_Name)
    shutil.move(old_Path, new_Path)


def classification_Data():
    numberOfMalware, numberOfBenign, countOfIndex, countOfBenign, countOfMalware = 0, 0, 0, 0, 0
    train_ID, valid_ID, test_ID, train_Label, valid_Label, test_Label = [], [], [], [], [], []

    image_Path = './data/total_Img/'
    train_Path = './data/img_size_128/train/'
    valid_Path = './data/img_size_128/valid/'
    test_Path = './data/img_size_128/test/'

    data_Info_File = open('./data/labels/128_extracted_Image_Info.csv', 'r', encoding='utf-8')
    data_Info = csv.reader(data_Info_File)

    '''
    Data set is not static
    '''

    for data in data_Info:
        if data[1] == '0':                                                  # if benign
            countOfBenign += 1
            if countOfBenign <= 1792:                                       # train data set
                train_ID.append(data[2])
                train_Label.append('0')
                rename_Move(image_Path+data[0]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            train_Path+data[2]+'.jpg')
            elif countOfBenign > 1792 and countOfBenign <= 2015:            # valid data set
                valid_ID.append(data[2])
                valid_Label.append('0')
                rename_Move(image_Path+data[0]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            valid_Path+data[2]+'.jpg')
            else:                                                           # test data set
                test_ID.append(data[2])
                test_Label.append('0')
                rename_Move(image_Path + data[0] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            test_Path + data[2] + '.jpg')
        elif data[1] == '1':                                                # if malware
            countOfMalware += 1
            if countOfMalware <= 2236:                                      # train data set
                train_ID.append(data[2])
                train_Label.append('1')
                rename_Move(image_Path + data[0] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            train_Path + data[2] + '.jpg')
            elif countOfMalware > 2236 and countOfMalware <= 2515:          # valid data set
                valid_ID.append(data[2])
                valid_Label.append('1')
                rename_Move(image_Path+data[0]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            image_Path+data[2]+'.jpg',
                            valid_Path+data[2]+'.jpg')
            else:                                                           # test data set
                test_ID.append(data[2])
                test_Label.append('1')
                rename_Move(image_Path + data[0] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            image_Path + data[2] + '.jpg',
                            test_Path + data[2] + '.jpg')

        countOfIndex += 1

    print('count of classification file : ' + str(countOfIndex))
    print('train_ID size : ' + str(len(train_ID)))
    print('valid_ID size : ' + str(len(valid_ID)))
    print('test_ID size : ' + str(len(test_ID)))
    print('train_Label size : ' + str(len(train_Label)))
    print('valid_Label size : ' + str(len(valid_Label)))
    print('test_Label size : ' + str(len(test_Label)))
    print('>>>>>>>>>> Successfully classify the dataset into train, valid, and test data')

    write_Classification_Data('./data/labels/128_train_ID.csv', train_ID)
    write_Classification_Data('./data/labels/128_valid_ID.csv', valid_ID)
    write_Classification_Data('./data/labels/128_test_ID.csv', test_ID)
    write_Classification_Data('./data/labels/128_train_Label.csv', train_Label)
    write_Classification_Data('./data/labels/128_valid_Label.csv', valid_Label)
    write_Classification_Data('./data/labels/128_test_Label.csv', test_Label)

    data_Info_File.close()

    print('>>>>>>>>>> Completion of saving each data ID and label information in csv format')

def write_Classification_Data(path, data):
    write_File = open(path, 'w', encoding='utf-8')

    for value in data:
        write_File.write(str(value)+'\n')

    write_File.close()

def __dataMakerMain__():
    '''
    If you want to create ID and label for image data,
    run get_ExtractedFile_Data(), classification_Data() method
    '''

    get_ExtractedFile_Data()
    classification_Data()

__dataMakerMain__()