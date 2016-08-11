from tensorflow.contrib.layers import convolution2d, batch_norm, fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from scipy.sparse import coo_matrix
import math


K = 10
Ks = 20  # the number of nearest neighbours of a sample
Kc = 5  # the number of nearest clusters of a cluster
a = 1.0
l = 1.0  # lambda
alpha = 0
gamma = 2.0

def calA(fea):
    a = 1.0
    Ks = 20 # the number of nearest neighbours of a sample
    Ns = np.size(fea, 0)

    # Initialization with K-means
    # Nc = max(int(math.ceil(Ns / 4)), K)
    # C = []
    # for i in range(Nc):
    #     C.append([])
    # for i in range(Ns):
    #     C[indexClusters[i]].append(i)

    # Calculate Dis
    Dis = squareform(pdist(fea))

    # Calculate W
    indexDis = Dis.argsort(axis = 1)[:, 0 : Ks + 1]
    sortedDis = np.sort(Dis, axis = 1)[:, 0 : Ks + 1]

    sig2 = np.mean(np.mean(sortedDis)) * a
    XI = np.transpose(np.tile(range(Ns), (Ks + 1, 1)))
    W = coo_matrix((np.exp(-sortedDis.flatten()*(1 / sig2)), (XI.flatten(), indexDis.flatten())), shape = (Ns, Ns)).toarray()

    # Calculate A
    asymA = np.zeros((Ns,Ns))
    A = np.zeros((Ns,Ns))
    for j in range(Ns):
        for i in range(j - 1):
            asymA[j, i] = np.dot(np.sum(W[i, :][ :, j], 0), np.sum(W[j, :][ :, i], 1)) / math.pow(np.size(i, 0), 2)
            asymA[i, j] = np.dot(np.sum(W[j, :][ :, i], 0), np.sum(W[i, :][ :, j], 1)) / math.pow(np.size(j, 0), 2)
            A[i, j] = asymA[i, j] + asymA[j, i]
            A[j, i] = A[i, j]

    return A


def merge_cluster():
    fea = mnist.train.images
    Ns = np.size(fea, 0)

    # Initialization with K-means
    Nc = max(int(math.ceil(Ns / 4)), K)
    indexClusters = KMeans(Nc).fit_predict(fea)
    C = []
    for i in range(Nc):
        C.append([])
    for i in range(Ns):
        C[indexClusters[i]].append(i)

    # Calculate Dis
    Dis = squareform(pdist(fea))

    # Calculate W
    indexDis = Dis.argsort(axis=1)[:, 0: Ks + 1]
    sortedDis = np.sort(Dis, axis=1)[:, 0: Ks + 1]

    sig2 = np.mean(np.mean(sortedDis)) * a
    XI = np.transpose(np.tile(range(Ns), (Ks + 1, 1)))
    W = coo_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
                   shape=(Ns, Ns)).toarray()

    # Calculate A
    asymA = np.zeros((Nc, Nc))
    A = np.zeros((Nc, Nc))
    for j in range(Nc):
        for i in range(j - 1):
            asymA[j, i] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1)) / math.pow(
                np.size(C[i], 0), 2)
            asymA[i, j] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1)) / math.pow(
                np.size(C[j], 0), 2)
            A[i, j] = asymA[i, j] + asymA[j, i]
            A[j, i] = A[i, j]

    # Find two clusters with the smallest loss
    np.fill_diagonal(A, - float('Inf'))
    indexA = A.argsort(axis=1)[:, ::-1][:, 0: Kc]
    sortedA = np.sort(A, axis=1)[:, ::-1][:, 0: Kc]

    minLoss = float('Inf')
    for i in range(Nc):
        loss = - (1 + l) * sortedA[i, 0] + sum(sortedA[i, 1: Kc]) * l / (Kc - 1)
        if loss < minLoss:
            minLoss = loss
            minIndex1 = min(i, indexA[i, 0])
            minIndex2 = max(i, indexA[i, 0])

    # Merge
    cluster1 = C[minIndex1]
    cluster2 = C[minIndex2]
    new_cluster = cluster1 + cluster2

    # update the merged cluster and its affinity
    C[minIndex1] = new_cluster
    asymA[minIndex1, 0: Nc] = asymA[minIndex1, 0: Nc] + asymA[minIndex2, 0: Nc]
    len1 = np.size(cluster1, 0)
    len2 = np.size(cluster2, 0)
    asymA[0: Nc, minIndex1] = asymA[0: Nc, minIndex1] * (1 + alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2) \
                              + asymA[0: Nc, minIndex2] * (1 + alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)

    A[minIndex1, :] = asymA[minIndex1, :] + asymA[:, minIndex1]
    A[:, minIndex1] = A[minIndex1, :]

    # replace the second cluster to be merged with the last cluster of the cluster array
    if (minIndex2 != Nc):
        C[minIndex2] = C[-1]
        asymA[0: Nc, minIndex2] = asymA[0: Nc, Nc - 1]
        asymA[minIndex2, 0: Nc] = asymA[Nc - 1, 0: Nc]
        A[0: Nc, minIndex2] = A[0: Nc, Nc - 1]
        A[minIndex2, 0: Nc] = A[Nc - 1, 0: Nc]

    # remove the last cluster
    C.pop()
    asymA[0: Nc, Nc - 1] = 0
    asymA[Nc - 1, 0: Nc] = 0
    A[0: Nc, Nc - 1] = 0
    A[Nc - 1, 0: Nc] = 0

    # generate sample labels
    labels = np.ones((Ns, 1))
    for i in range(np.size(C)):
        labels[C[i]] = i

    return C, indexA, labels

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [50,28,28,50])
net = convolution2d(x_image, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding = 'VALID', normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 24 * 24
net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID', data_format='NHWC', name=None) # 12 * 12
#net = max_pool2d(net)
net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding = 'VALID', normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 8 * 8
net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 1, 1, 1], 'VALID', data_format='NHWC', name=None)  # 4 * 4
#net = max_pool2d(net)
#tf.nn.relu(features, name=None)
net = fully_connected(net, num_outputs=160, activation_fn=tf.nn.relu)
net = tf.nn.l2_normalize(net, 0)

net_shape = net.get_shape().as_list()
net_reshape = tf.reshape(net, [net_shape[0], net_shape[1] * net_shape[2] * net_shape[3]])

A = tf.py_func(calA, [net_reshape], [tf.float32])
C, indexA, labels = tf.py_func(merge_cluster, [], [tf.int32, tf.int32, tf.int32])

loss = tf.Variable(tf.zeros([1])) # loss function
for i in range(50):
    j = tf.gather(labels, i) #labels[i]
    x_k = tf.gather(C, tf.gather(indexA, j)) #C[indexA[j]]
    Aij = tf.gather(tf.gather(A, i), tf.gather(C, j)) #A[i, C[j]]
    Aik = tf.gather(tf.gather(A, i), x_k) #A[i, x_k]
    loss = loss + - l / (Kc - 1) * (gamma * tf.reduce_sum(Aij) + tf.reduce_sum(Aik))

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
    0.01,  # Base learning rate.
    batch * 50,  # Current index into the dataset.
    50,  # Decay step.
    0.99995,  # Decay rate.
    staircase=True)

train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

