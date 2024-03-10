from __future__ import print_function
import time
from datetime import timedelta
import os
print("CUDA_VISIBLE_DEVICES 0 setting")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import FastText
from gensim.models import word2vec
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding
from Generator import DataGenerator

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('##')
                if content:
                    contents.append(content.split(' '))
                    labels.append(label)
            except:
                pass
    return contents, labels


def read_vocab(vocab_dir):
    """word to id list generation"""
    with open_file(vocab_dir) as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def read_category():
    """class to id"""
    categories = ['0', '1']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """file to id"""
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        tempo = []
        for x in contents[i]:
            if x in word_to_id:
                tempo.append(word_to_id[x])

        data_id.append(tempo)
        label_id.append(cat_to_id[labels[i]])

    x_len = []
    for i in data_id:  # i is all contents of one file
        if len(i) < 600:
            x_len.append(len(i))
        else:
            x_len.append(600)

    # pad_sequences Unified length
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, maxlen=max_length, padding='post', truncating='post')
    #y_onehot = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))
    label_data = np.array(label_id)
    return x_pad, label_data, x_len


# embedding model load
def embedding_load(model_name, vocab_dir, model_select, embedding_dim):
    print("embedding model loading...")

    # changable
    word_vocab_name = '%s' % vocab_dir
    word2vec_file_name = './embedding_model/%s' % model_name

    with open(word_vocab_name) as fp:
        words = [_.strip() for _ in fp.readlines()]

    # changable
    embedding_matrix = np.zeros((len(words), embedding_dim))  # 300 600 900
    if model_select == 'fasttext':
        print("fasttext model loading...")
        print("vocab name = %s" % str(vocab_dir))
        print("word2vec_file_name = %s" % str(word2vec_file_name))
        model = FastText.load(word2vec_file_name)
    else:
        print("word2vec model loading...")
        print("vocab name = %s" % str(vocab_dir))
        print("word2vec_file_name = %s" % str(word2vec_file_name))
        model = word2vec.Word2Vec.load(word2vec_file_name)


    for i in range(len(words)):
        if words[i] in model.wv.vocab:
            embedding_vector = model.wv[words[i]]
            embedding_matrix[i] = embedding_vector

    print("embedding model load Done!")

    return model, embedding_matrix


def LSTM_model(input_x, input_y, x_val, y_val, x_test, y_test, embedding_model, embedding_matrix, embedding_dim, model_name, layer, cell1, cell2):
    #tensorboard_callback = kr.callbacks.TensorBoard(log_dir="logs")
    checkpointer = kr.callbacks.ModelCheckpoint(filepath='./Model/%s_%d_layer_%d_cell1_%d_cell2_' % (model_name, layer, cell1, cell2) + '{epoch:02d}-{val_loss:.4f}' + '.h5', monitor='val_loss', verbose=1, save_best_only=True)

    if layer == 1:
        print("Train start %d layer_%d_cell1_%d_cell2!" % (layer, cell1, cell2))
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(len(embedding_model.wv.vocab) + 1, embedding_dim, input_length=600,
                                weights=[embedding_matrix], trainable=False))
            model.add(Bidirectional(LSTM(units=int(cell1/2), dropout=0.2)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy'])
            history = model.fit(input_x,
                                input_y,
                                batch_size=32,
                                epochs=100,
                                callbacks=[checkpointer],
                                validation_data=(x_val, y_val),
                                shuffle=True,
                                validation_batch_size=1)
            kr.backend.clear_session()

            # model testing
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1)
            print("loss and metrics ====>> " + str(loss_and_metrics))

    if layer == 2:
        print("Train start %d layer_%d_cell1_%d_cell2!" % (layer, cell1, cell2))
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(len(embedding_model.wv.vocab) + 1, embedding_dim, input_length=600,
                                weights=[embedding_matrix], trainable=False))
            model.add(Bidirectional(LSTM(units=int(cell1/2), dropout=0.2, return_sequences=True)))
            model.add(Bidirectional(LSTM(units=int(cell2/2), dropout=0.2)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy'])
            history = model.fit(input_x,
                                input_y,
                                batch_size=32,
                                epochs=100,
                                callbacks=[checkpointer],
                                validation_data=(x_val, y_val),
                                shuffle=True,
                                validation_batch_size=1)
            kr.backend.clear_session()

            # model testing
            loss_and_metrics = model.evaluate(x_test, y_test, batch_size=1)
            print("loss and metrics ====>> " + str(loss_and_metrics))

    with open('./logs/%s_%d_layer_%d_cells1_%d_cells2_log.txt' % (model_name, layer, cell1, cell2), mode='w') as f:
        f.write('test log ===> ' + str(loss_and_metrics) + '\n')
        f.close()

    # model save
    model.save('./save/%s_%d_layer_%d_cells1_%d_cells2.h5' % (model_name, layer, cell1, cell2))

    print("print result visualization...")
    # 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
    plt.plot(history.history['accuracy'], color='b', label='train_acc')
    plt.plot(history.history['val_accuracy'], color='g', label='valid_acc')
    plt.plot(history.history['loss'], color='y', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='valid_loss')
    plt.title('Model accuracy & losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.twinx().set_ylabel('Accuracy')
    plt.legend(['Train_loss', 'Valid_loss'], loc='upper left')
    plt.legend(['Train_acc', 'Valid_acc'], loc='lower left')
    plt.savefig('./train_graph/%s_%d_layer_%d_cell1_%d_cell2.png' % (model_name, layer, cell1, cell2))


def LSTM_model2(x_train, y_train, x_val, y_val, x_test, y_test, embedding_model, embedding_matrix, embedding_dim, model_name, train_batch, valid_batch, test_batch, layer, cell1, cell2):
    train_gen = DataGenerator(x_train, y_train, batch_size=train_batch, dim=(600,), n_classes=2, shuffle=True)
    train_step = int(np.floor(len(x_train) / train_batch))
    valid_gen = DataGenerator(x_val, y_val, batch_size=valid_batch, dim=(600,), n_classes=2, shuffle=True)
    valid_step = int(np.floor(len(x_val) / valid_batch))
    test_gen = DataGenerator(x_test, y_test, batch_size=test_batch, dim=(600,), n_classes=2, shuffle=True)
    #tensorboard_callback = kr.callbacks.TensorBoard(log_dir="logs")
    checkpointer = kr.callbacks.ModelCheckpoint(filepath='./Model/%s_%d_layer_%d_cell1_%d_cell2_' % (model_name, layer, cell1, cell2) + '{epoch:02d}-{val_loss:.4f}' + '.h5', monitor='val_loss', verbose=1, save_best_only=True)
    #earlystopping_callback = kr.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    if layer == 1:
        print("Train start %d layer_%d_cell1_%d_cell2!" % (layer, cell1, cell2))
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(len(embedding_model.wv.vocab) + 1, embedding_dim, input_length=600,
                                weights=[embedding_matrix], trainable=False))
            model.add(Bidirectional(LSTM(units=int(cell1/2), dropout=0.2)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy'])
            history = model.fit_generator(
                generator=train_gen,
                steps_per_epoch=train_step,
                epochs=100,
                callbacks=[checkpointer],
                validation_data=valid_gen,
                validation_steps=valid_step
            )
            kr.backend.clear_session()

            # model testing
            loss_and_metrics = model.evaluate_generator(generator=test_gen)
            print("loss and metrics ====>> " + str(loss_and_metrics))

    elif layer == 2:
        print("Train start %d_layer_%d_cell1_%d_cell2!" % (layer, cell1, cell2))
        with tf.device('/gpu:0'):
            model = Sequential()
            model.add(Embedding(len(embedding_model.wv.vocab) + 1, embedding_dim, input_length=600,
                                weights=[embedding_matrix], trainable=False))
            model.add(Bidirectional(LSTM(units=int(cell1/2), dropout=0.2, return_sequences=True)))
            model.add(Bidirectional(LSTM(units=int(cell2/2), dropout=0.2)))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=['accuracy'])
            history = model.fit_generator(
                generator=train_gen,
                steps_per_epoch=train_step,
                epochs=100,
                callbacks=[checkpointer],
                validation_data=valid_gen,
                validation_steps=valid_step
            )
            kr.backend.clear_session()

            # model testing
            loss_and_metrics = model.evaluate_generator(generator=test_gen)
            print("loss and metrics ====>> " + str(loss_and_metrics))

    with open('./logs/%s_%d_layer_%d_cells1_%d_cells2_log.txt' % (model_name, layer, cell1, cell2), mode='w') as f:
        f.write('test log ===> ' + str(loss_and_metrics) + '\n')
        f.close()

    # model save
    model.save('./save/%s_%d_layer_%d_cells1_%d_cells2.h5' % (model_name, layer, cell1, cell2))

    print("print result visualization...")
    # 학습 정확성 값과 검증 정확성 값을 플롯팅 합니다.
    plt.plot(history.history['accuracy'], color='b', label='train_acc')
    plt.plot(history.history['val_accuracy'], color='g', label='valid_acc')
    plt.plot(history.history['loss'], color='y', label='train_loss')
    plt.plot(history.history['val_loss'], color='r', label='valid_loss')
    plt.title('Model accuracy & losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.twinx().set_ylabel('Accuracy')
    plt.legend(['Train_loss', 'Valid_loss'], loc='upper left')
    plt.legend(['Train_acc', 'Valid_acc'], loc='lower left')
    plt.savefig('./train_graph/%s_%d_layer_%d_cell1_%d_cell2.png' % (model_name, layer, cell1, cell2))


def train_model(seq_length, embedding_dim, train_dir, test_dir, valid_dir, vocab_dir, model_name, model_select, train_batch, valid_batch, test_batch, layer, cell1, cell2):
    '''
    # memory dynamic allocate
    gpus = tf.config.experimental.list_physical_devices('GPU')   # GPU check
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    
    # memory only allocate n GB
    gpus = tf.config.experimental.list_physical_devices('GPU')   # GPU check
    try:
        print("GPU memory allocate...")
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    '''
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    vocab_size = len(words)

    x_train, y_train, x_train_length = process_file(train_dir, word_to_id, cat_to_id, seq_length)  # def process_file(filename, word_to_id, cat_to_id, max_length=600)
    x_val, y_val, x_valid_length = process_file(valid_dir, word_to_id, cat_to_id, seq_length)
    x_test, y_test, x_test_length = process_file(test_dir, word_to_id, cat_to_id, seq_length)


    # load embedding model
    EBD_model, embedding_matrix = embedding_load(model_name, vocab_dir, model_select, embedding_dim)

    print('embedding_matrix shape : ' + str(embedding_matrix.shape))
    print('x_train shape : ' + str(x_train.shape))
    #print('x_train shape [1] : ' + str(x_train.shape[1]))
    print('y_train shape : ' + str(y_train.shape))
    print('x_val shape : ' + str(x_val.shape))
    print('y_val shape : ' + str(y_val.shape))
    print('x_test shape : ' + str(x_test.shape))
    print('y_test shape : ' + str(y_test.shape))

    # train & validation part
    #LSTM_model(x_train, y_train, x_val, y_val, x_test, y_test, EBD_model, embedding_matrix, embedding_dim, model_name, layer, cell1, cell2)
    LSTM_model2(x_train, y_train, x_val, y_val, x_test, y_test, EBD_model, embedding_matrix, embedding_dim, model_name, train_batch, valid_batch, test_batch, layer, cell1, cell2)

def train_main(seq_length, train_dir, test_dir, valid_dir, vocab_dir, train_batch, valid_batch, test_batch, method):
    embedding_dim1 = 100
    embedding_dim2 = 200
    embedding_dim3 = 300

    # method 1 - 1 layer & 128 cells - word2vec 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_100' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_200' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_300' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)

    # method 1 - 1 layer & 128 cells - fasttext 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_100' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_200' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_300' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=128, cell2=0)
    

    # method 1 - 1 layer & 256 cells - word2vec 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_100' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_200' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_300' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)


    # method 1 - 1 layer & 256 cells - fasttext 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_100' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_200' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_300' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=1, cell1=256, cell2=0)


    # method 1 - 2 layer & 128-128 cells - word2vec 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_100' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_200' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_300' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)


    # method 1 - 2 layer & 128-128 cells - fasttext 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_100' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_200' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_300' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=128, cell2=128)


    # method 1 - 2 layer & 256-128 cells - word2vec 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_100' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_200' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_300' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)


    # method 1 - 2 layer & 256-128 cells - fasttext 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_100' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_200' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_300' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=128)


    # method 1 - 2 layer & 256-256 cells - word2vec 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_100' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_200' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_word2vec_model_300' % method,
                    'word2vec', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)


    # method 1 - 2 layer & 256-256 cells - fasttext 100, 200, 300 train
    train_model(seq_length, embedding_dim1, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_100' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)
    train_model(seq_length, embedding_dim2, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_200' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)
    train_model(seq_length, embedding_dim3, train_dir, test_dir, valid_dir, vocab_dir, '%d_fasttext_model_300' % method,
                    'fasttext', train_batch, valid_batch, test_batch, layer=2, cell1=256, cell2=256)


def main():
    '''
    # memory only allocate n GB
    print("GPU memory set...")
    gpus = tf.config.experimental.list_physical_devices('GPU')  # GPU check
    try:
        print("GPU memory allocate...")
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
    '''
    # hyperparam
    seq_length = 600
    train_batch = 32
    valid_batch = 8
    test_batch = 8

    base_dir = './data/train'
    train_dir3 = os.path.join(base_dir, '3_train.txt')
    test_dir3 = os.path.join(base_dir, '3_test.txt')
    valid_dir3 = os.path.join(base_dir, '3_valid.txt')
    vocab_dir3 = os.path.join(base_dir, '3_word_vocab.txt')

    train_main(seq_length, train_dir3, test_dir3, valid_dir3, vocab_dir3, train_batch, valid_batch, test_batch, method=3)


main()