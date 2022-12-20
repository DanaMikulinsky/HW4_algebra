from PIL import Image
import numpy as np
import pickle
import time

batch1_path = "./cifar-10-python/cifar-10-batches-py/data_batch_1"
batch2_path = "./cifar-10-python/cifar-10-batches-py/data_batch_2"
batch3_path = "./cifar-10-python/cifar-10-batches-py/data_batch_3"
batch4_path = "./cifar-10-python/cifar-10-batches-py/data_batch_4"
batch5_path = "./cifar-10-python/cifar-10-batches-py/data_batch_5"
test_path = "./cifar-10-python/cifar-10-batches-py/test_batch"


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,  encoding='bytes')
    return dict


def center(X):
    n = X.shape[0]
    meanVec = np.zeros(n)
    for i in range(n):
        meanVec[i] = np.mean(X[i, :])
    Xcentered = np.zeros(X.shape)
    for j in range(X.shape[1]):
        Xcentered[:, j] = X[:, j] - meanVec
    return Xcentered


def getPCA(X, s):
    svd = np.linalg.svd(X)
    U = svd[0][:, 0:s]
    Ut = np.transpose(U)
    Xs = np.matmul(np.matmul(U, Ut), X)
    return Xs


batch1_data = np.transpose(np.asarray(unpickle(batch1_path)[b'data']))
batch2_data = np.transpose(np.asarray(unpickle(batch2_path)[b'data']))
batch3_data = np.transpose(np.asarray(unpickle(batch3_path)[b'data']))
batch4_data = np.transpose(np.asarray(unpickle(batch4_path)[b'data']))
batch5_data = np.transpose(np.asarray(unpickle(batch5_path)[b'data']))
test_data = np.transpose(np.asarray(unpickle(test_path)[b'data']))

batches = [batch1_data, batch2_data, batch3_data, batch4_data, batch5_data, test_data]
test1 = center(batch1_data)