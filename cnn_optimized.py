import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm, fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
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

    logger2.info('%.2f s,Begin to calculate Dis', timeit.default_timer() - tic)
    # Calculate Dis
    r = tf.reshape(tf.reduce_sum(fea * fea, 1), [-1, 1])
    Dis = -(r - 2 * tf.matmul(fea, tf.transpose(fea)) + tf.transpose(r))
    logger2.info('%.2f s,finished calculation of Dis', timeit.default_timer() - tic)

    # Calculate W
    sortedDis, indexDis = tf.nn.top_k(Dis, k=Ks + 1, sorted=True)
    sig2 = tf.reduce_mean(tf.reduce_mean(-sortedDis)) * a
    XI = tf.tile(tf.reshape(tf.range(0, Ns, 1), [-1]), [Ks + 1])
    sparse_indices = tf.transpose([XI, tf.reshape(indexDis, [-1])])
    W = tf.sparse_to_dense(sparse_values=tf.exp(tf.reshape(sortedDis, [-1]) * (1 / sig2)),
                           sparse_indices=sparse_indices,
                           output_shape=[Ns, Ns], validate_indices=False)
    logger2.info('%.2f s,finished calculation of W', timeit.default_timer() - tic)

    # Calculate A
    asymA = tf.Variable(tf.zeros([Nc * Nc, 1]))
    A = tf.Variable(tf.zeros([Nc, Nc]))
    for i in range(Nc):
        logger2.info('%.2f s,i:%d', timeit.default_timer() - tic, i)
        for j in range(i - 1):
            asymA_ij = tf.matmul(tf.reshape(tf.reduce_sum(tf.gather(tf.gather(W, C[i]), C[j])),[1, -1]),
                                 tf.reshape(tf.reduce_sum(tf.gather(tf.gather(W, C[j]), C[i])),[-1, 1])) / math.pow(np.size(C[i], 0), 2)
            asymA_ji = tf.matmul(tf.reshape(tf.reduce_sum(tf.gather(tf.gather(W, C[j]), C[i])),[1, -1]),
                                 tf.reshape(tf.reduce_sum(tf.gather(tf.gather(W, C[i]), C[j])),[-1, 1])) / math.pow(np.size(C[j], 0), 2)
            tf.scatter_update(asymA, i*Nc+j, tf.reshape(asymA_ij,[-1]))
            test = tf.scatter_update(asymA, j*Nc+i, tf.reshape(asymA_ji,[-1]))

            sess.run(test)

    print asymA.eval()

    asymA = tf.reshape(asymA, [Nc, Nc])
    A = asymA + tf.transpose(asymA)
    return A, asymA

def merge_cluster(A, asymA, C_flattened, C_index):
    # Find two clusters with the smallest loss

    asymA = tf.Variable(asymA)
    C_flattened = tf.Variable(C_flattened)
    C_index = tf.Variable(C_index)

    #np.fill_diagonal(A, - float('Inf'))
    # indexA = A.argsort(axis=1)[:, ::-1][:, 0: Kc]
    # sortedA = np.sort(A, axis=1)[:, ::-1][:, 0: Kc]
    sortedA, indexA = tf.nn.top_k(A, k = Kc + 1, sorted=True)

    min_loss = float('Inf')
    min_index1 = 0
    min_index2 = 0
    for i in range(Nc):
        loss = - (1 + l) * tf.gather(tf.gather(sortedA,i), 1) + tf.reduce_sum(tf.gather(tf.gather(sortedA,i), tf.range(2,Kc+1))) * l / (Kc - 1)

        # def loss_smaller():
        #     # min_loss = loss
        #     # min_index1 = tf.minimum(i, tf.gather(tf.gather(indexA, i), 1))
        #     # min_index2 = tf.maximum(i, tf.gather(tf.gather(indexA, i), 1))
        #     return min_loss
        #
        # def loss_larger():
        #     return min_loss
        #
        # min_loss= tf.cond(loss<min_loss, loss_smaller, loss_larger)

        min_loss = tf.select(loss<min_loss, loss, min_loss)
        min_index1 = tf.select(loss < min_loss, tf.minimum(i, tf.gather(tf.gather(indexA, i), 1)), min_index1)
        min_index2 = tf.select(loss < min_loss, tf.maximum(i, tf.gather(tf.gather(indexA, i), 1)), min_index2)

        sess.run(loss)
        sess.run(min_loss)
        sess.run(min_index1)
        sess.run(min_index2)

    print loss.eval(),min_loss.eval(),min_index1.eval(), min_index2.eval()

    # Merge
    # cluster1 = C[min_index1]
    # cluster2 = C[min_index2]
    # new_cluster = cluster1 + cluster2
    len2 = tf.gather(C_index, min_index2)- tf.gather(C_index, min_index2-1)
    cluster2 = tf.gather(C_flattened, tf.range(tf.gather(C_index, min_index2-1), tf.gather(C_index, min_index2)))
    tf.scatter_update(C_flattened, tf.range(tf.gather(C_index, min_index1)+len2, tf.gather(C_index, min_index2)),
                      tf.gather(C_flattened, tf.range(tf.gather(C_index, min_index1), tf.gather(C_index, min_index2-1))))
    tf.scatter_add(C_index, tf.range(tf.gather(C_index, min_index1), tf.gather(C_index, min_index2-1)), len2)

    # Update the merged cluster and its affinity
    # C[min_index1] = new_cluster
    tf.scatter_add(asymA, min_index1, tf.gather(asymA,min_index2))
    if min_index1!=0:
        len1 = tf.gather(C_index, min_index1)- tf.gather(C_index, min_index1-1)
    else:
        len1= tf.gather(C_index, min_index1)
    # len2 = np.size(cluster2, 0)

    asymA_transpose = tf.Variable(tf.transpose(asymA))
    asymA_index1 = tf.gather(asymA, min_index1) * (1 + alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2) \
                              + tf.gather(asymA, min_index2) * (1 + alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)
    tf.scatter_update(asymA_transpose, min_index1, asymA_index1)
    asymA = tf.Variable(tf.transpose(asymA_transpose))

    # Replace the second cluster to be merged with the last cluster of the cluster array
    if (min_index2 != Nc):
        C[min_index2] = C[-1]
        tf.scatter_update(asymA, min_index2, tf.gather(asymA, Nc - 1))
        asymA_transpose = tf.Variable(tf.transpose(asymA))
        tf.scatter_update(asymA_transpose, min_index2, tf.gather(asymA_transpose, Nc - 1))
        asymA = tf.transpose(asymA_transpose)

    # Remove the last cluster
    C.pop()
    asymA_new = tf.Variable(tf.zeros([Nc-1, Nc-1]))
    tf.scatter_update(asymA_new, tf.range(Nc-1), tf.transpose(tf.gather(tf.transpose(tf.gather(asymA, tf.range(Nc-1))), tf.range(Nc-1))))

    A = asymA + tf.transpose(asymA)

    return A, asymA_new, C

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
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# images = mnist.train.images
# labels_gnd = mnist.train.labels
images = mnist.train.images[0:1000]
labels_gnd = mnist.train.labels[0:1000]
fea = images

Ns = np.size(fea, 0)
batch_size = Ns/epochs

x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [Ns,28,28,1])
x_image_batch = tf.reshape(x, [batch_size,28,28,1])

gnd = np.argmax(labels_gnd, axis=1)

# Initialization with K-means
Nc = 11
# Nc = max(int(math.ceil(Ns / 4)), K) #20 or 10
logger1.info('%.2f s,Begin K means...', timeit.default_timer() - tic)
#centroids, indexClusters = TFKMeansCluster(np.float64(fea), K)
indexClusters = MiniBatchKMeans(Nc, max_iter=100, max_no_improvement=10, tol=0.0001).fit_predict(fea) #optimize?????
logger1.info('%.2f s,K means is completed', timeit.default_timer() - tic)

C = []
for i in range(Nc):
    C.append([])
for i in range(Ns):
    C[indexClusters[i]].append(i)

C_flattened = tf.Variable(tf.zeros([Ns]))
C_index = tf.Variable(tf.zeros([Nc]))
len_total = 0
for i in range(Nc):
    len = np.size(C[i], 0)
    len_total = len_total + len
    tf.scatter_update(C_index, i, len_total)
    tf.scatter_update(C_flattened, tf.range(len_total, len_total + len), C[i])

logger2.info('%.2f s,Begin to calculate A based on clusters...', timeit.default_timer() - tic)
# A = calA_sample_based(tf.convert_to_tensor(fea))
A_clu, asymA = calA_cluster_based(fea, C)
logger2.info('%.2f s,Calculation of A based on clusters is completed', timeit.default_timer() - tic)

ts = 0
t = 0
p = 0
Np = np.ceil(eta * Nc)
sess.run(tf.initialize_all_variables())

while Nc > K:
    logger2.info('%.2f s, Timestep: %d', timeit.default_timer() - tic, t)
    t = t + 1
    Nc = Nc - 1
    A_clu, asymA, C = merge_cluster(A_clu, asymA, C_flattened, C_index)

    #if t == ts + Np:
    if True:
        indexA = A_clu.argsort(axis=1)[:, ::-1][:, 0: Kc]
        labels = get_labels(C)
        features = model(x_image)

        for i in range(epochs):
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
            batch = mnist.train.next_batch(batch_size)
            train_step.run(feed_dict={x: batch[0]})

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
