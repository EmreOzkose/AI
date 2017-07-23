import numpy as np
import matplotlib.pyplot as plt

def read_data():
    train_data_file = open("Data/train.csv","r")
    test_data_file = open("Data/test.csv","r")
    test_data_file_2 = open('Data/sample_submission.csv','r')

    m = 42000;
    n = 28*28;

    train_data_x = np.zeros((m, n))
    train_data_y = np.zeros((m, 1))

    index = 0
    for i in train_data_file.readlines()[1:]:
        i = i.strip("\n").split(",")
        train_data_y[index] = int(i[0])
        train_data_x[index] = np.asarray(i[1:],dtype=np.float)

        index+=1

    m = 28000;
    n = 28 * 28;

    test_data_x = np.zeros((m, n))
    test_data_y = np.zeros((m, 1))

    index = 0
    for i in test_data_file.readlines()[1:]:
        i = i.strip("\n").split(",")
        test_data_x[index] = np.asarray(i, dtype=np.float)

        index += 1

    index = 0
    for i in test_data_file_2.readlines()[1:]:
        i = i.strip("\n").split(",")
        test_data_y[index] = int(i[1])

        index+=1

    return train_data_x,  train_data_y, test_data_x, test_data_y

def plot_data(input_image):
    x = np.reshape(input_image, (28, 28))

    plt.imshow(x, cmap='gray')
    plt.show()

def read_cifar_data():
    paths=['Data/cifar-10-batches-py/data_batch_1',
           'Data/cifar-10-batches-py/data_batch_2',
           'Data/cifar-10-batches-py/data_batch_3',
           'Data/cifar-10-batches-py/data_batch_4',
           'Data/cifar-10-batches-py/data_batch_5'
           ]

    m = 50000
    n = 32*32*3

    train_data_x = np.zeros((m, n))
    train_data_y = np.zeros((m, 1))

    bottom = 0
    top = 10000
    for i in paths:
        x = unpickle(i)
        train_data_x[bottom:top] = x[bytes('data', 'utf-8')]
        row = np.asarray(x[bytes('labels', 'utf-8')])

        train_data_y[bottom:top] = row.reshape((10000,1))

        bottom += 10000; top += 10000

    return train_data_x, train_data_y

def plot_cifar_data(input_image):
    print(input_image.shape)

    image = np.zeros((int(len(input_image)/3), 1))

    for i in range(0,int(len(input_image)/3)):
        r = input_image[i]
        g = input_image[i+1024]
        b = input_image[i+2048]

        image[int(i)] = r/255 + g/255 + b/255

    image = np.reshape(image, (32, 32))
    plt.imshow(image)
    plt.show()

def array_to_one(input_array):
    (m, n) = input_array.shape
    return np.reshape(input_array, (m*n,1))

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

