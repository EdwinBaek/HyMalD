import random
import os
import csv
import numpy as np

def alldata_csv_maker(image_path, ID_label_path):
    csv_file = open(ID_label_path, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)

    train_label = './labels/trainSet.csv'
    pre_label = './labels/preSet.csv'
    final1_label = './labels/finalSet1.csv'
    final2_label = './labels/finalSet2.csv'

    img_file_list = os.listdir(image_path)    # filename.json.csv.csv.png

    modified_list = []
    for file in img_file_list:
        modified_list.append(file)
    print("modified_list")
    print(modified_list)

    train_csv = open(train_label, 'r', encoding='utf-8')
    train_rdr = csv.reader(train_csv)
    for line1 in train_rdr:
        csv_writer.writerow([line1[0], line1[1]])
    train_csv.close()

    pre_csv = open(pre_label, 'r', encoding='utf-8')
    pre_rdr = csv.reader(pre_csv)
    for line2 in pre_rdr:
        csv_writer.writerow([line2[0], line2[1]])
    pre_csv.close()

    final1_csv = open(final1_label, 'r', encoding='utf-8')
    final1_rdr = csv.reader(final1_csv)
    for line3 in final1_rdr:
        csv_writer.writerow([line3[0], line3[1]])
    final1_csv.close()

    final2_csv = open(final2_label, 'r', encoding='utf-8')
    final2_rdr = csv.reader(final2_csv)
    for line4 in final2_rdr:
        csv_writer.writerow([line4[0], line4[1]])
    final2_csv.close()


def dataset_id_maker(image_path, ID_label_path):
    csv_file = open(ID_label_path, 'r', encoding='utf-8')
    csv_rdr = csv.reader(csv_file)

    alldata_name_list = []
    alldata_label_list = []
    for line in csv_rdr:
        alldata_name_list.append(line[0])
        alldata_label_list.append(line[1])

    img_file_list = os.listdir(image_path)    # filename.vir.png
    index_list = []
    for file in img_file_list:
        modified = file.replace(".jpg", "")
        list_index = alldata_name_list.index(modified)    # list_index + 1 is label csv file row index
        index_list.append(list_index)

    img_name_list = []
    img_label_list = []
    for i in index_list:
        img_name_list.append(alldata_name_list[i])
        img_label_list.append(alldata_label_list[i])
    name_arr = np.array(img_name_list)
    label_arr = np.array(img_label_list)

    print("name arr")
    print(name_arr)
    print("label_arr ")
    print(label_arr)

    return name_arr, label_arr


def filename_modify(image_path):
    img_file_list = os.listdir(image_path)  # filename.json.csv.csv.png
    for file in img_file_list:
        origin_file_dir = os.path.join(image_path, file)

        modified = file.replace(".jpg", ".vir.jpg")
        print("modifed")
        print(modified)

        dst = os.path.join(image_path, modified)
        print("dst")
        print(dst)
        os.rename(origin_file_dir, dst)


def dataset_maker(name_arr, label_arr):
    print("name_array print")
    print(name_arr)
    print("label_arr print")
    print(label_arr)

    train_id, test_id, valid_id, train_label, test_label, valid_label = [], [], [], [], [], []
    count_index = 0     # for문 index count variable
    ben_cnt = 0
    mal_cnt = 0
    for i in label_arr:
        if i == '0':      # if benign
            if ben_cnt <= 120:
                train_id.append(name_arr[count_index])
                train_label.append(int(0))
                ben_cnt += 1
            elif ben_cnt > 120 and ben_cnt <= 150:
                valid_id.append(name_arr[count_index])
                valid_label.append(int(0))
                ben_cnt += 1
            else:
                test_id.append(name_arr[count_index])
                test_label.append(int(0))
                ben_cnt += 1
        elif i == '1':    # if malware
            if mal_cnt <= 120:
                train_id.append(name_arr[count_index])
                train_label.append(int(1))
                mal_cnt += 1
            elif mal_cnt > 120 and mal_cnt <= 150:
                valid_id.append(name_arr[count_index])
                valid_label.append(int(1))
                mal_cnt += 1
            else:
                test_id.append(name_arr[count_index])
                test_label.append(int(1))
                mal_cnt += 1

        count_index += 1

    return train_id, test_id, valid_id, train_label, test_label, valid_label


if __name__ == '__main__':
    image_path = './img/MB'
    ID_label_path = './labels/MB_labels1.csv'
    label_path = './data/imagelabels.mat'
    setid_path = './data/setid.mat'

    #filename_modify(image_path)
    #alldata_csv_maker(image_path, ID_label_path)

