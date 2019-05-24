import numpy as np
import h5py 
import skimage.transform as sc
from sklearn.model_selection import train_test_split


def resize(arr):
    im = []
    for i in range(arr.shape[0]):    
        img = sc.resize(arr[i], (88, 88), preserve_range=True)
        im.append(img)
    im = np.array(im)
    return im


def load_data():
    print('load_data')
    data = h5py.File('LowRes_13434_overlapping_pairs.h5', 'r')   #loading data
    data = data.get('dataset_1')
    images = data[:, :, :, 0]
    labels = data[:, :, :, 1]
    images = resize(images)
    labels = resize(labels)
    images = np.expand_dims(images, -1)
    labels = np.expand_dims(labels, -1)
    print("Images shape:{}".format(images.shape))
    print("Labels shape:{}".format(labels.shape))
    # splitting into train and test sets
    # images_train = images[:images.shape[0]-20]
    # images_test = images[images.shape[0]-20:]
    # labels_train = labels[:labels.shape[0]-20]
    # labels_test = labels[labels.shape[0]-20:]
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels,
                                                                            test_size=0.2,
                                                                            random_state=0)
    return images_train, labels_train, images_test, labels_test
