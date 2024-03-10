import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.optimize.tests.test_lbfgsb_hessinv import test_1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, TensorBoard
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import glob
import numpy as np
import os.path as path
import imageio
import keras.backend.tensorflow_backend as K
import csv
import tensorflow as tf
from sklearn import metrics

tf.compat.v1.disable_eager_execution()

# Hyperparams
SAVE_NAME = 'C:/Users/ucloud/Desktop/Dynamic Data/1. CNN/1. 128 x 128/128_Result.csv'
MODEL_SUMMARY_FILE = 'C:/Users/ucloud/Desktop/Dynamic Data/1. CNN/1. 128 x 128/CNN_Entropy_128_model_summary.txt'
IMAGE_SIZE = 224  # 128, 256, 384, 512, 768
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)
EPOCHS = 1
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.6
THRESHOLD = 0.5
PATIENCE = 40
N_TO_VISUALIZE = 10

def getData(path):
    getFile = open(path, 'r', encoding='utf-8')
    getData = csv.reader(getFile)
    getList = []

    for data in getData:
        getList += data
    getFile.close()

    return getList

# data preprocessing
train_ID = getData('./data/labels/128_train_ID.csv')
valid_ID = getData('./data/labels/128_valid_ID.csv')
test_ID = getData('./data/labels/128_test_ID.csv')

train_Label = getData('./data/labels/128_train_Label.csv')
valid_Label = getData('./data/labels/128_valid_Label.csv')
test_Label = getData('./data/labels/128_test_Label.csv')

train_Label = list(map(float, train_Label))
valid_Label = list(map(float, valid_Label))
test_Label = list(map(float, test_Label))

train_labels = np.array(train_Label)
train_images = ['./data/img_size_128/train/%s.jpg' % i for i in train_ID]
valid_labels = np.array(valid_Label)
valid_images = ['./data/img_size_128/valid/%s.jpg' % i for i in valid_ID]
test_labels = np.array(test_Label)
test_images = ['./data/img_size_128/test/%s.jpg' % i for i in test_ID]

# TRAINING_LOGS_FILE = 'DASC_training_logs.csv'
# TEST_FILE = 'DASC_NoOptimizer_Train60_Test40_test_file.txt'
MODEL_FILE = 'DASC_NoOptimizer_Train60_model.h5'

trimg = [imageio.imread(path) for path in train_images]
trimg = np.asarray(trimg)
trimg = trimg / 255

valimg = [imageio.imread(path) for path in valid_images]
valimg = np.asarray(valimg)
valimg = valimg / 255

teimg = [imageio.imread(path) for path in test_images]
teimg = np.asarray(teimg)
teimg = teimg / 255

with K.tf.device('/gpu:0'):
    model = Sequential()
    # Conv2D(컨볼루션 필터의 수, (행, 열), padding = 'valid' or padding = 'same', input_shape = (행, 열, 채널 수), activation = 'relu'
    model.add(Conv2D(96, (7, 7), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1'))

    model.add(Conv2D(256, (5, 5), strides=(4, 4), padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
    model.add(Conv2D(1024, (3, 3), strides=(1, 1), padding='same'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Dropout(THRESHOLD))
    model.add(Dense(4096))
    model.add(Dropout(THRESHOLD))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    with open(MODEL_SUMMARY_FILE, "w") as fh:
        model.summary(print_fn=lambda line: fh.write(line + "\n"))

    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=1, mode='auto')

    # tensorboard
    LOG_DIRECTORY_ROOT = ""
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}/".format(LOG_DIRECTORY_ROOT, now)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True)

    callbacks = [early_stopping]

    # Training & Validation
    print("train start...")
    hist = model.fit(trimg, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1, validation_data=(valimg, valid_labels))


    test_loss, test_acc = model.evaluate(teimg, test_labels, batch_size=16)
    print('test loss : ' + str(test_loss))
    print('test accuracy : ' + str(test_acc))

    print('---')
    test_Label = list(map(int, test_Label))
    y_true = test_Label
    y_pred_list = model.predict_classes(teimg)

    y_pred = []
    for value in y_pred_list:
        y_pred += (list(value))

    print(y_true)
    print(y_pred)

    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    cm_list = []
    cm_list.append(cm[0][0])
    cm_list.append(cm[0][1])
    cm_list.append(cm[1][0])
    cm_list.append(cm[1][1])
    print(cm_list)

    write_file = open(SAVE_NAME, 'w', newline='', encoding='utf-8')
    wr = csv.writer(write_file)

    wr.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    for index in range(0, len(hist.history['loss'])):
        row = [index + 1, hist.history['loss'][index], hist.history['accuracy'][index], hist.history['val_loss'][index], hist.history['val_accuracy'][index]]
        wr.writerow(row)

    wr.writerow(['-'])
    wr.writerow(['test_loss', 'test_acc', 'tp', 'fn', 'fp', 'tn'])
    wr.writerow([test_loss, test_acc, cm_list[0], cm_list[1], cm_list[2], cm_list[3]])

    write_file.close()
print("model save...")
model.save(MODEL_FILE)
print("model save complete!")

