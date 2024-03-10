import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter

import sys
import os
import csv
import numpy as np
import scipy.io
from PIL import Image
from sppnet import SPPNet

save_path = './data/sppnet_test.pth'
image_path = './img/MB'
ID_label_path = './labels/MB_labels1.csv'

BATCH = 1
EPOCH = 2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("device print")
print(device)
print("torch.cuda.is_available() print")
print(torch.cuda.is_available())

class MyDataset(Dataset):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __init__(self, image_path, mode, transform=None):
        """
        image_00001.jpg
        image_00002.jpg
        image_00003.jpg
        """

        if mode == 'train':
            self.labels = train_label     #trnid -> 1x6149 trnid에는 그 이미지가 몇 번째 라벨인지가 들어있다.
            self.images = ['./img/MB/%s.jpg' % i for i in train_id]    # 1에서 6149까지의 데이터셋
        elif mode == 'valid':
            self.labels = valid_label
            self.images = ['./img/MB/%s.jpg' % i for i in valid_id]
        else:
            self.labels = test_label
            self.images = ['./img/MB/%s.jpg' % i for i in test_id]

        self.transform = transform



    def __getitem__(self, index):
        label = self.labels[index]      # 배열 형태로 만든 것을 index화 시킨다.
        image = self.images[index]      # index화 시킴-> 얻는 것은 filename임
        print("before", image)
        if self.transform is not None:
            image = self.transform(Image.open(image))   #transform은 이미지를 변형할 때 쓴다. /// filename을 open하여 이미지를 얻는다.
        print("after", image)
        return image, label

    def __len__(self):
        return len(self.labels)


def train(model, device, train_loader, criterion, optimizer, epoch):
    # train part
    model.train()
    train_loss = 0
    print("씨발씨발")
    # batch_idx는 여기서 나온다.
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()

        output = model(image)
        label = label.type(torch.LongTensor).cuda()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()    # 200개까지의 loss를 다 더한다.

        if (batch_idx + 1) % 200 == 0:
            train_loss /= 200

            print('Train Epoch: %d [%d/%d (%.4f%%)\tLoss: %.4f]'% (
                epoch, (batch_idx + 1) * len(image), len(train_loader.dataset),
                100. * (batch_idx + 1) * len(image) / len(train_loader.dataset), train_loss)) # 그리고 loss 출력
            train_loss = 0



def valid(model, device, valid_loader, criterion, epoch):
    model.eval()
    total_true = 0
    total_loss = 0
    with torch.no_grad():
        for image, label in valid_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)

            label = label.type(torch.LongTensor).cuda()
            loss = criterion(output, label)
            pred = torch.max(output, 1)[1]

            total_loss += (pred.view(label.size()).data == label.data).sum().item()
            total_loss += loss.item()

    accuracy = total_true / len(valid_loader.dataset)
    loss = total_loss / len(valid_loader.dataset)
    print('\nValidation Epoch: %d ====> Accuracy: [%d/%d (%.4f%%)]\tAverage loss: %.4f\n' % (epoch, total_true, len(valid_loader.dataset), 100. * accuracy, loss))


def dataset_maker(name_arr, label_arr):
    #print("name_array print")
    #print(name_arr)
    #print("label_arr print")
    #print(label_arr)

    train_id, test_id, valid_id, train_label, test_label, valid_label = [], [], [], [], [], []
    count_index = 0     # for문 index count variable
    ben_cnt = 0
    mal_cnt = 0
    for i in label_arr:
        if i == '0':      # if benign
            ben_cnt += 1
            if ben_cnt <= 294:
                train_id.append(name_arr[count_index])
                train_label.append(int(0))
            elif ben_cnt > 294 and ben_cnt <= 331:
                valid_id.append(name_arr[count_index])
                valid_label.append(int(0))
            else:
                test_id.append(name_arr[count_index])
                test_label.append(int(0))
        elif i == '1':    # if malware
            mal_cnt += 1
            if mal_cnt <= 1112:
                train_id.append(name_arr[count_index])
                train_label.append(int(1))
            elif mal_cnt > 1112 and mal_cnt <= 1250:
                valid_id.append(name_arr[count_index])
                valid_label.append(int(1))
            else:
                test_id.append(name_arr[count_index])
                test_label.append(int(1))

        count_index += 1
    print("mal_cnt")
    print(mal_cnt)
    print("ben_cnt")
    print(ben_cnt)

    return train_id, test_id, valid_id, train_label, test_label, valid_label


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

    csv_file.close()
    del alldata_name_list
    del alldata_label_list
    del img_name_list
    del img_label_list

    return name_arr, label_arr



if __name__ == '__main__':
    name_arr, label_arr = dataset_id_maker(image_path, ID_label_path)
    train_id, test_id, valid_id, train_label, test_label, valid_label = dataset_maker(name_arr, label_arr)
    '''
    print(np.array(train_id).shape)
    print(np.array(train_label).shape)
    print(np.array(test_id).shape)
    print(np.array(test_label).shape)
    print(np.array(valid_id).shape)
    print(np.array(valid_label).shape)
    '''

    # dataset의 크기는 각기 다르다.
    train_dataset = MyDataset(image_path,
                              mode='train', transform=
                              transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])) # imagenet mean, std
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    print('Train size:', len(train_loader))

    valid_dataset = MyDataset(image_path,
                             mode='valid', transform=
                             transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False)
    print('valid size:', len(valid_loader))

    test_dataset = MyDataset(image_path,
                             mode='test', transform=
                             transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]))
    test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
    print('Test size:', len(test_loader))

    model = SPPNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCH + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        torch.cuda.empty_cache()
        torch.save(model, save_path)
        valid(model, device, valid_loader, criterion, epoch)
        torch.cuda.empty_cache()
