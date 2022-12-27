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


def reductDim(X, U, s):
    Ut = np.transpose(U[:, 0:s])
    Xs = np.matmul(Ut, X)
    return Xs


def turnGrey(A):
    """
    turning a color image (represented by a 1-D array)
    to a greyscale image (also represented by a 1-D array)
    """
    final = np.zeros((A.shape[0], int(A.shape[1]/3)))
    for i in range(A.shape[0]):
        img = A[i, :]
        img_reshaped = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        actual_img = Image.fromarray(img_reshaped.astype('uint8'))
        grey_img = actual_img.convert("L")
        grey_img_mat = np.array(grey_img)
        final[i, :] = grey_img_mat.flatten()
    return final


def gather_data(paths):
    """
    :param paths: a list of paths to the batches
    :return: 1 full matrix that holds all the pictures and 1 array with the corresponding labels
    """
    if len(paths) == 0:
        return
    if len(paths) == 1:
        full_data = unpickle(paths[0])[b'data']
        full_labels = unpickle(paths[0])[b'labels']
    else:
        full_data = unpickle(paths[0])[b'data']
        full_labels = unpickle(paths[0])[b'labels']
        for i in range(1, len(paths)):
            batch_data = unpickle(paths[i])[b'data']
            batch_labels = unpickle(paths[i])[b'labels']
            full_labels = full_labels + batch_labels
            full_data = np.concatenate((full_data, batch_data), axis=0)
    return full_data, np.asarray(full_labels)


def KNN(y, X_data, X_labels, k):
    tmp = (X_data.transpose() - y).transpose()
    distances = np.linalg.norm(tmp, axis=0)
    labeled = np.row_stack((distances, X_labels.T))
    sorted_labeled = labeled[:, labeled[0].argsort()]
    nearest_labels = np.asarray(sorted_labeled[1, 0:k], dtype='int')
    label_y = np.bincount(nearest_labels).argmax()
    return label_y


if __name__ == '__main__':
    batch1_path = "./cifar-10-python/cifar-10-batches-py/data_batch_1"
    batch2_path = "./cifar-10-python/cifar-10-batches-py/data_batch_2"
    batch3_path = "./cifar-10-python/cifar-10-batches-py/data_batch_3"
    batch4_path = "./cifar-10-python/cifar-10-batches-py/data_batch_4"
    batch5_path = "./cifar-10-python/cifar-10-batches-py/data_batch_5"
    test_path = "./cifar-10-python/cifar-10-batches-py/test_batch"
    paths = [batch1_path, batch2_path, batch3_path, batch4_path, batch5_path]
    #paths = [batch1_path]
    tmp_data, labels = gather_data(paths)
    test_data_orig = unpickle(test_path)[b'data']
    test_labels = unpickle(test_path)[b'labels']
    test_data_grey = turnGrey(test_data_orig)
    test_data = center(np.transpose(test_data_grey))
    data = center(np.transpose(turnGrey(tmp_data)))
        #now every data point (image) is a column & the matrix is centered
    print(data.shape)
    s_vals = [1, 5, 20, 55, 100, 500, 1000]
    k_vals = [1, 5, 10, 55, 100, 360]
    errors = np.zeros((len(k_vals), len(s_vals)))
    svd = np.linalg.svd(data)
    for i in range(len(s_vals)):
        dataS = reductDim(data, svd[0], s_vals[i])
        testS = reductDim(test_data, svd[0], s_vals[i])
        for j in range(len(k_vals)):
            y_labels = []
            for h in range(testS.shape[1]):
                y_labels.append(KNN(testS[:, h], dataS, labels, k_vals[j]))
            errors[j, i] = np.sum(np.asarray(y_labels) != np.asarray(test_labels))/len(y_labels)
            #the error for some s_vals[i] & k_vals[j] is in row j, col i

    print(errors)