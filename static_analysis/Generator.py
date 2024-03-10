import numpy as np
import tensorflow.keras as kr

class DataGenerator(kr.utils.Sequence):
    def __init__(self, X, y, batch_size, dim, n_classes, shuffle=True):
        self.X = X
        self.y = y if y is not None else y
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __data_generation(self, X_list, y_list):
        X = np.empty((self.batch_size, *self.dim), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        if y is not None:
            # 지금 같은 경우는 MNIST를 로드해서 사용하기 때문에
            # 배열에 그냥 넣어주면 되는 식이지만,
            # custom image data를 사용하는 경우
            # 이 부분에서 이미지를 batch_size만큼 불러오게 하면 됩니다.
            for i, (img, label) in enumerate(zip(X_list, y_list)):
                X[i] = img
                y[i] = label
                #kr.utils.to_categorical(y, num_classes=self.n_classes)
            return X, y

        else:
            for i, img in enumerate(X_list):
                X[i] = img

            return X

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X_list = [self.X[k] for k in indexes]

        if self.y is not None:
            y_list = [self.y[k] for k in indexes]
            X, y = self.__data_generation(X_list, y_list)
            steps = self.__len__()
            return X, y
        else:
            y_list = None
            X = self.__data_generation(X_list, y_list)
            steps = self.__len__()
            return X