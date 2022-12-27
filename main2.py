from PIL import Image
import numpy as np
import pickle
import time

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


def turnGrey(A):
    """
    turning a color image (represented by a 1-D array)
    to a greyscale image (also represented by a 1-D array)
    """
    final = np.zeros(A.shape[0], A.shape[1]/3)
    for i in range(A.shape[0]):
        img = A[i, :]
        img_reshaped = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        actual_img = Image.fromarray(img_reshaped.astype('uint8'))
        grey_img = actual_img.convert("L")
        grey_img_mat = np.array(grey_img)
        final[i, :] = grey_img_mat.flatten()
    return final


def gather_data(paths):
    if len(paths) == 0:
        return
    if len(paths) == 1:
        full_data = unpickle(paths)[b'data']
        full_labels = unpickle(paths)[b'labels']
    else:
        full_data = unpickle(paths[0])[b'data']
        full_labels = unpickle(paths[0])[b'labels']
        for i in range(1, len(paths)):
            batch_data = unpickle(paths[i])[b'data']
            batch_labels = unpickle(paths[i])[b'labels']
            full_labels = full_labels + batch_labels
            full_data = np.concatenate((full_data, batch_data), axis=0)
    return full_data, np.asarray(full_labels)


if __name__ == '__main2__':
    batch1_path = "./cifar-10-python/cifar-10-batches-py/data_batch_1"
    batch2_path = "./cifar-10-python/cifar-10-batches-py/data_batch_2"
    batch3_path = "./cifar-10-python/cifar-10-batches-py/data_batch_3"
    batch4_path = "./cifar-10-python/cifar-10-batches-py/data_batch_4"
    batch5_path = "./cifar-10-python/cifar-10-batches-py/data_batch_5"
    test_path = "./cifar-10-python/cifar-10-batches-py/test_batch"
    paths = [batch1_path, batch2_path, batch3_path, batch4_path, batch5_path]
    tmp_data, labels = gather_data(paths)
    data = turnGrey(tmp_data)

