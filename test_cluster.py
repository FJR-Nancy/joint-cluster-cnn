from cluster import *
from mnist import MNIST

def test_cluster(dataset, feature):
    '''
    -To test the cluster algorithm
        dataset: 'mnist-test', 'mnist-train', 'pie', 'usps', 'coil20', 'coil100'
        feature: 'no', 'lbp', 'hog'
    '''

    if 'mnist-test' in dataset:
        mndata = MNIST('.')
        fea, gnd = mndata.load_testing()
        #gnd = gnd[0:1000]
        #fea = fea[0:1000]
        K = 10
        imageSize = 28
        cluster(fea, gnd, K, imageSize, feature)

    elif 'mnist-train' in dataset:
        mndata = MNIST('.')
        fea, gnd = mndata.load_training()
        # fea = loadMNISTImages('train-images-idx3-ubyte')'
        # gnd = loadMNISTLabels('train-labels-idx1-ubyte')
        # load('MNIST_10kTrain.mat')
        gnd = gnd[0:1000]
        fea = fea[0:1000]
        K = 10
        imageSize = 28
        cluster(fea, gnd, K, imageSize, feature)
    '''
    elif 'pie' in dataset:
        load('PIE_pose27.mat')
        K = 68
        imageSize = 32
        cluster(fea, gnd, K, imageSize, feature)
    
    elif 'usps' in dataset:
        load('USPS.mat')
        gnd = gnd[0:1000]
        fea = fea[0:1000]
        K = 10
        imageSize = 16
        cluster(fea, gnd, K, imageSize, feature)
    
    elif 'coil20' in dataset:
        load('COIL20.mat')
        K = 20
        imageSize = 32
        cluster(fea, gnd, K, imageSize, feature)
    
    elif 'coil100' in dataset:
        load('COIL100.mat')
        fea = float(fea)
        K = 100
        imageSize = 32
        cluster(fea, gnd, K, imageSize, feature)
    '''