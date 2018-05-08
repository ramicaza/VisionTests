#!/usr/bin/env python3

import os
from scipy.misc import imread
import numpy as np
import cv2

class Data:
    def _get_images(self):
        bg_files = []
        lanes_files = []
        potholes_files = []
        bg_path = os.path.join(os.getcwd(),'background_train')
        lanes_path = os.path.join(os.getcwd(),'lanes_train')
        potholes_path = os.path.join(os.getcwd(),'potholes_train')
        X = []
        Y = []
        for (dirpath, dirnames, filenames) in os.walk(bg_path):
            bg_files.extend(filenames)
            break
        for (dirpath, dirnames, filenames) in os.walk(lanes_path):
            lanes_files.extend(filenames)
            break
        for (dirpath, dirnames, filenames) in os.walk(potholes_path):
            potholes_files.extend(filenames)
            break
        for f in bg_files:
            image = imread(os.path.join(bg_path, f))
            X.append(image)
            Y.append(0)
        for f in lanes_files:
            image = imread(os.path.join(lanes_path, f))
            X.append(image)
            Y.append(1)
        for f in potholes_files:
            image = imread(os.path.join(potholes_path, f))
            X.append(image)
            Y.append(2) # label pothols as
        for i in range(len(X)):
            X[i] = cv2.resize(X[i],(32, 32), interpolation=cv2.INTER_CUBIC)
        return np.array(X), np.array(Y)

    def _unison_shuffled_copies(self,a, b):
        assert len(a) == len(b)
        np.random.seed(6969) #deterministic
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def get_partitions(self,train,valid,test):
        assert(abs(train + valid + test-1) < 10**-10)
        X,Y = self._get_images()
        X,Y = self._unison_shuffled_copies(X,Y)
        end_train = int(len(X)*train)
        end_valid = int(len(X)*(train+valid)) #currently only using train and test
        x_train = X[:end_train]
        y_train = Y[:end_train]
        x_valid = X[end_train:end_valid]
        y_valid = Y[end_train:end_valid]
        x_test = X[end_valid:]
        y_test = Y[end_valid:]

        return (x_train,y_train), (x_valid,y_valid), (x_test,y_test)

# unit tests
if __name__ == "__main__":
    d = Data()
    train,valid,test = d.get_partitions(0.7,0.2,0.1)
    print(train[0].shape)
    print(train[1].shape)
    print(valid[0].shape)
    print(valid[1].shape)
    print(test[0].shape)
    print(test[1].shape)
