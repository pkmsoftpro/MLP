import numpy as np
import pickle
import os
#import download
#from dataset import one_hot_encoded

from os import listdir
#from skimage.io import imread
from PIL import Image

#from matplotlib import pyplot as plt

########################################################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = ""

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, "cifar-10-batches-py/", filename)  #


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')

    return data


def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size]) #-1 means auto set the number of that dim
                                                               # and the second dimension is divided sequentially

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])  

    return images


def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images. from the dictionary by key
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array. ## so these are class labels
    cls = np.array(data[b'labels'])

    # Convert the images.
    images = _convert_images(raw_images)

    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


def maybe_download_and_extract():
    """
    Download and extract the CIFAR-10 data-set if it doesn't already exist
    in data_path (set this variable first to the desired path).
    """

    download.maybe_download_and_extract(url=data_url, download_dir=data_path)


def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names


def load_training_data():
    """
    Load all the training-data for the CIFAR-10 data-set.
    The data-set is split into 5 data-files which are merged here.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # Number of images in this batch. # the length should be the number of the rows
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array. # we only care about the 1st dimension of the 4d array
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch # because it only has one dimension

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls #, one_hot_encoded(class_numbers=cls, num_classes=num_classes)


def load_test_data():
    """
    Load all the test-data for the CIFAR-10 data-set.
    Returns the images, class-numbers and one-hot encoded class-labels.
    """

    images, cls = _load_data(filename="test_batch") # because it only has one batch, so no for loop needed

    return images, cls #, one_hot_encoded(class_numbers=cls, num_classes=num_classes)

########################################################################

if __name__ == "__main__":
    files = [f for f in listdir('.') if f.endswith('.png')]

   #maybe_download_and_extract()
    names = load_class_names()
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    #print (X_train.shape)
    #print (X_train.dtype)
    #print (names[Y_train[1]])

    #print (y_train.shape)
    #print (y_train[:10])

    #from matplotlib import pyplot as plt
    #plt.imshow(X_train[1])
    #plt.show()

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    #print (Y_train.shape)
    #print (Y_train[:10])

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))
    #print (model.output_shape)

    model.add(Conv2D(32, (5, 5), activation='relu')) # why we need a conv filter twice?
    model.add(MaxPooling2D(pool_size=(2,2))) # ?still need to read about dropout?
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', # loss function
              optimizer='adam',    # optimizer
              metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)

    for f in files:
#        f_data = imread(f)
#        print(model.predict(f))

#   f_data = imread(f)
        f_data = Image.open(f)
#    print(np.shape(f_data))
#    print(f_data.mode)
        if f_data.mode != 'RGB':
            f_data = f_data.convert('RGB')
#    print(np.shape(f_data))
            f_data_change = np.array(f_data.resize((32,32),Image.ANTIALIAS))
#    print(np.shape(f_data_change))
            f_input = np.array([f_data_change])
            #from matplotlib import pyplot as plt
            #plt.imshow(f_data_change)
            #plt.show()
            predict_result = model.predict(f_input)
            print(predict_result)
            name_list = [ np.where(r==1)[0][0] for r in predict_result]
            for j in name_list:
                print(f,names[j])
            



