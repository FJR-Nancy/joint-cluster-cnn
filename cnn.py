import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm, fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_mldata
import math
from bestMap import *
from MutualInfo import *
from Kmeans import *
import timeit
import logging

K = 10
Ks = 20  # the number of nearest neighbours of a sample
Kc = 5  # the number of nearest clusters of a cluster
a = 1.0
l = 1.0  # lambda
alpha = 0
gamma = 2.0
eta = 0.9
epochs = 20

def calA_sample_based(fea):
    Ns = fea.get_shape().as_list()[0]

    # Calculate Dis
    r = tf.reshape(tf.reduce_sum(fea * fea, 1), [-1, 1])
    Dis = -(r - 2 * tf.matmul(fea, tf.transpose(fea)) + tf.transpose(r))

    # Calculate W
    sortedDis, indexDis = tf.nn.top_k(Dis, k = Ks + 1, sorted=True)
    sig2 = tf.reduce_mean(tf.reduce_mean(-sortedDis)) * a
    XI = tf.tile(tf.reshape(tf.range(0, Ns, 1), [-1]), [Ks + 1])
    sparse_indices = tf.transpose([XI, tf.reshape(indexDis, [-1])])
    W = tf.sparse_to_dense(sparse_values = tf.exp(tf.reshape(sortedDis,[-1])*(1 / sig2)),
                           sparse_indices = sparse_indices,
                           output_shape = [Ns, Ns], validate_indices=False)

    # Calculate A
    A = tf.mul(W, tf.transpose(W))
    return A

def calA_cluster_based(fea, C):
    Ns = np.size(fea, 0)

    logger2.info('%.2f s,Begin to fit neighbour graph', timeit.default_timer() - tic)
    # Calculate Dis
    # Dis = squareform(pdist(fea))
    # r = tf.reshape(tf.reduce_sum(fea * fea, 1), [-1, 1])
    # Dis = -(r - 2 * tf.matmul(fea, tf.transpose(fea)) + tf.transpose(r))
    # sess.run(Dis)
    # print Dis.eval()
    neigh = NearestNeighbors(n_neighbors=Ks+1, n_jobs=-1).fit(fea)
    logger2.info('%.2f s,finished fitting, begin to calculate Dis', timeit.default_timer() - tic)

    Dis = neigh.kneighbors_graph(X=fea, mode='distance').power(2)
    logger2.info('%.2f s,finished the calculation of Dis, begin to calculate W', timeit.default_timer() - tic)

    # Calculate W
    # indexDis = Dis.argsort(axis=1)[:, 0: Ks + 1]
    # sortedDis = np.sort(Dis, axis=1)[:, 0: Ks + 1]
    sig2 = Dis.sum()/(Ks * Ns) * a
    # XI = np.transpose(np.tile(range(Ns), (Ks + 1, 1)))
    # W = coo_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
    #                shape=(Ns, Ns)).toarray()
    W_scr = (-Dis * (1 / sig2)).expm1()
    I = csr_matrix((np.ones(Ks * Ns), W_scr.nonzero()))
    W = (W_scr+I).toarray() # exp-1?????

    logger2.info('%.2f s,finished the calculation of W, begin to calculate A', timeit.default_timer() - tic)

    # Calculate A
    asymA = np.zeros((Nc, Nc))
    A = np.zeros((Nc, Nc))
    for j in range(Nc):
        for i in range(j):
            if np.size(C[i], 0)!=0:
                asymA[j, i] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1)) / math.pow(
                np.size(C[i], 0), 2)
            if np.size(C[j], 0)!=0:
                asymA[i, j] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1)) / math.pow(
                np.size(C[j], 0), 2)
            A[i, j] = asymA[i, j] + asymA[j, i]
            A[j, i] = A[i, j]
    return A, asymA

def merge_cluster(A, asymA, C):
    # Find two clusters with the smallest loss
    Nc = np.size(C)
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

    # Update the merged cluster and its affinity
    C[minIndex1] = new_cluster
    asymA[minIndex1, 0: Nc] = asymA[minIndex1, 0: Nc] + asymA[minIndex2, 0: Nc]
    len1 = np.size(cluster1, 0)
    len2 = np.size(cluster2, 0)
    asymA[0: Nc, minIndex1] = asymA[0: Nc, minIndex1] * (1 + alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2) \
                              + asymA[0: Nc, minIndex2] * (1 + alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)

    A[minIndex1, :] = asymA[minIndex1, :] + asymA[:, minIndex1]
    A[:, minIndex1] = A[minIndex1, :]

    # Replace the second cluster to be merged with the last cluster of the cluster array
    if (minIndex2 != Nc):
        C[minIndex2] = C[-1]
        asymA[0: Nc, minIndex2] = asymA[0: Nc, Nc - 1]
        asymA[minIndex2, 0: Nc] = asymA[Nc - 1, 0: Nc]
        A[0: Nc, minIndex2] = A[0: Nc, Nc - 1]
        A[minIndex2, 0: Nc] = A[Nc - 1, 0: Nc]

    # Remove the last cluster
    C.pop()
    asymA_new = asymA[0 : Nc - 1, 0 : Nc - 1]
    A_new = A[0 : Nc - 1, 0 : Nc - 1]

    return A_new, asymA_new, C

def model(data):
    # CNN
    net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                        normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 24 * 24
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 12 * 12

    net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                        normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 8 * 8
    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 4 * 4

    net = fully_connected(net, num_outputs=160, activation_fn=tf.nn.relu)
    net = tf.nn.l2_normalize(net, 3)

    net_shape = net.get_shape().as_list()
    net_reshape = tf.reshape(net, [net_shape[0], net_shape[1] * net_shape[2] * net_shape[3]])

    return net_reshape

def get_labels(C):
    # Generate sample labels
    labels = np.zeros(Ns, np.int)
    for i in range(np.size(C)):
        labels[C[i]] = i

    return labels

sess = tf.InteractiveSession()

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='logfile.txt',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger1 = logging.getLogger('K-means')
logger2 = logging.getLogger('Cluster')
logger3 = logging.getLogger('CNN')
logger4 = logging.getLogger('Result')

tic=timeit.default_timer()

# Input data
# mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
# images = mnist.train.images
# gnd = mnist.train.labels
# images = mnist.train.images[0:1000]
# gnd = mnist.train.labels[0:1000]

usps = fetch_mldata("USPS")
images, gnd = usps.data, usps.target.astype(np.int) - 1

fea = images

Ns = np.size(fea, 0)
batch_size = Ns/epochs

x = tf.placeholder(tf.float32, shape=[None, 256])
x_image = tf.reshape(x, [Ns,16,16,1])
x_image_batch = tf.reshape(x, [batch_size,16,16,1])

#gnd = np.argmax(labels_gnd, axis=1)

# Initialization with K-means
# Nc = 10
Nc = max(int(math.ceil(Ns / 10)), K) #20 or 50
logger1.info('%.2f s,Begin K means...', timeit.default_timer() - tic)
#centroids, indexClusters = TFKMeansCluster(np.float64(fea), K)
indexClusters = MiniBatchKMeans(Nc, max_iter=100, max_no_improvement=10, tol=0.0001).fit_predict(fea) #optimize?????
logger1.info('%.2f s,K means is completed', timeit.default_timer() - tic)

C = []
for i in range(Nc):
    C.append([])
for i in range(Ns):
    C[indexClusters[i]].append(i)

logger2.info('%.2f s,Begin to calculate A based on clusters...', timeit.default_timer() - tic)
# A = calA_sample_based(tf.convert_to_tensor(fea))
A_clu, asymA = calA_cluster_based(fea, C)
logger2.info('%.2f s,Calculation of A based on clusters is completed', timeit.default_timer() - tic)

ts = 0
t = 0
p = 0
Np = np.ceil(eta * Nc)

while Nc > K:
    logger2.info('%.2f s, Timestep: %d', timeit.default_timer() - tic, t)
    t = t + 1
    Nc = Nc - 1
    A_clu, asymA, C = merge_cluster(A_clu, asymA, C)

    if t == ts + Np:
        indexA = A_clu.argsort(axis=1)[:, ::-1][:, 0: Kc]
        labels = get_labels(C)
        features = model(x_image)

        for i in range(epochs):
            logger3.info('%.2f s, Period: %d, epochs: %d', timeit.default_timer() - tic, p, i)
            fea_batch = model(x_image_batch)
            A_sam = calA_sample_based(fea_batch)
            # sortedA, indexA = tf.nn.top_k(A_sam, k = Kc + 1, sorted=True)

            # Loss function
            loss = tf.Variable(tf.zeros([1]))
            for j in range(batch_size):
                Cid = labels[i * batch_size + j]
                x_j = np.array(C[Cid])
                Aij = tf.gather(tf.gather(A_sam, j), x_j[np.logical_and(x_j>=i*batch_size, x_j<j)], validate_indices=False)  # A[i, C[Cid]], here include itself  ???index outbound???
                x_k = []
                for n in range(Kc): #??????Kc cluster or samples??????
                    x_k = x_k + C[indexA[Cid][n]]
                    # sample = tf.gather(tf.gather(indexA, Cid), n + 1)
                    # if tf.reshape(tf.equal(tf.gather(labels, sample),Cid), [-1]):
                    #     x_k = x_k + sample
                x_k = np.array(x_k)
                Aik = tf.gather(tf.gather(A_sam, j), x_k[np.logical_and(x_k>=i*batch_size, x_k<j)], validate_indices=False)  # A[j, x_k]
                loss = loss + - l / (Kc - 1) * (gamma * tf.reduce_sum(Aij) + tf.reduce_sum(Aik))

            # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
            batch = tf.Variable(0)
            # Decay once per epoch, using an exponential schedule starting at 0.01.
            learning_rate = tf.train.exponential_decay(
                0.01,  # Base learning rate.
                batch * 50,  # Current index into the dataset.
                50,  # Decay step.
                0.99995,  # Decay rate.
                staircase=True)

            train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
            sess.run(tf.initialize_all_variables())
            #batch = mnist.train.next_batch(batch_size)
            batch = images[batch_size*i : batch_size*(i+1), :]
            train_step.run(feed_dict={x: batch})

        fea = sess.run(features, feed_dict={x: images}) # Get the new feature representation
        A_clu, asymA = calA_cluster_based(fea, C) # Update A based on the new feature representation
        ts = t
        Np = np.ceil(eta * Nc)
        p = p + 1
        logger3.info('%.2f s, Timestep: %d, Period: %d', timeit.default_timer() - tic, t, p)

labels = get_labels(C)
labels_maped = bestMap(gnd, labels)
AC = float(np.count_nonzero(gnd == labels_maped)) / np.size(gnd, 0) # Evaluate AC: accuracy
MIhat = MutualInfo(gnd, labels_maped) # Evaluate MIhat: nomalized mutual information
logger4.info('%.2f s, AC: %f, MIhat: %f', timeit.default_timer() - tic, AC, MIhat)
