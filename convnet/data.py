#!/usr/bin/env python3

import os
from scipy.misc import imread
import numpy as np

class Data:
    def _get_images(self):
        pos_files = []
        neg_files = []
        posp = os.path.join(os.getcwd(),'lanes_pos')
        negp = os.path.join(os.getcwd(),'lanes_neg')
        X = []
        Y = []
        for (dirpath, dirnames, filenames) in os.walk(posp):
            pos_files.extend(filenames)
            break
        for (dirpath, dirnames, filenames) in os.walk(negp):
            neg_files.extend(filenames)
            break
        for f in pos_files:
            image = imread(os.path.join(posp, f))
            X.append(image)
            Y.append(1)
        for f in neg_files:
            image = imread(os.path.join(negp, f))
            X.append(image)
            Y.append(0)
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
