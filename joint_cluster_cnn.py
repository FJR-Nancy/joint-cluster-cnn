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


class joint_cluster_cnn():

    K = 10
    Ks = 20  # the number of nearest neighbours of a sample
    Kc = 5  # the number of nearest clusters of a cluster
    a = 1.0
    l = 1.0  # lambda
    alpha = 0
    gamma = 2.0
    eta = 0.9
    epochs = 20

    def __init__(self, dataset, testmode = False):
        self.tic = timeit.default_timer()

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
        self.logger = logging.getLogger('')

        self.logger.info('%.2f s, Begin to extract dataset', timeit.default_timer() - self.tic)
        # Input data
        if 'mnist' in dataset:
            mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
            self.images = mnist.train.images
            self.gnd = mnist.train.labels
            image_size = 28
            self.K = 10
            self.logger.info('%.2f s, Finished extracting MNIST dataset', timeit.default_timer() - self.tic)
            # images = mnist.train.images[0:1000]
            # gnd = mnist.train.labels[0:1000]
        elif 'mnist-all' in dataset:
            mnist = fetch_mldata("MNIST")
            self.images = mnist.data
            self.gnd = mnist.target.astype(np.int) - 1
            image_size = 28
            self.K = 10
            self.logger.info('%.2f s, Finished extracting MNIST dataset', timeit.default_timer() - self.tic)
        elif 'usps' in dataset:
            usps = fetch_mldata("USPS")
            self.images = usps.data
            self.gnd = usps.target.astype(np.int) - 1
            image_size = 16
            self.K = 10
            self.logger.info('%.2f s, Finished extracting USPS dataset', timeit.default_timer() - self.tic)
        elif 'coil20' in dataset:
            coil20 = fetch_mldata("coil20")
            self.images = coil20.data
            self.gnd = coil20.target.astype(np.int) - 1
            image_size = 32
            self.K = 20
            self.logger.info('%.2f s, Finished extracting COIL20 dataset', timeit.default_timer() - self.tic)


        self.Ns = np.size(self.images, 0)
        self.batch_size = self.Ns / self.epochs

        if testmode == False:
            self.Nc = max(int(math.ceil(self.Ns / 10)), self.K)  # 10 or 20
        else:
            self.Nc = 12

        self.x = tf.placeholder(tf.float32, shape=[None, image_size * image_size])
        self.x_image = tf.reshape(self.x, [self.Ns, image_size, image_size, 1])
        self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size, image_size, 1])


    def calA_sample_based(self, fea):
        Ns = fea.get_shape().as_list()[0]

        # Calculate Dis
        r = tf.reshape(tf.reduce_sum(fea * fea, 1), [-1, 1])
        Dis = -(r - 2 * tf.matmul(fea, tf.transpose(fea)) + tf.transpose(r))

        # Calculate W
        sortedDis, indexDis = tf.nn.top_k(Dis, k = self.Ks + 1, sorted=True)
        sig2 = tf.reduce_mean(tf.reduce_mean(-sortedDis)) * self.a
        XI = tf.tile(tf.reshape(tf.range(0, Ns, 1), [-1]), [self.Ks + 1])
        sparse_indices = tf.transpose([XI, tf.reshape(indexDis, [-1])])
        W = tf.sparse_to_dense(sparse_values = tf.exp(tf.reshape(sortedDis,[-1])*(1 / sig2)),
                               sparse_indices = sparse_indices,
                               output_shape = [Ns, Ns], validate_indices=False)

        # Calculate A
        A = tf.mul(W, tf.transpose(W))
        return A

    def calA_cluster_based(self, fea, C):
        Ns = np.size(fea, 0)

        self.logger.info('%.2f s, Begin to fit neighbour graph', timeit.default_timer() - self.tic)
        # Calculate Dis
        # Dis = squareform(pdist(fea))
        # r = tf.reshape(tf.reduce_sum(fea * fea, 1), [-1, 1])
        # Dis = -(r - 2 * tf.matmul(fea, tf.transpose(fea)) + tf.transpose(r))
        # sess.run(Dis)
        # print Dis.eval()
        neigh = NearestNeighbors(n_neighbors=self.Ks+1, n_jobs=-1).fit(fea)
        self.logger.info('%.2f s, Finished fitting, begin to calculate Dis', timeit.default_timer() - self.tic)

        Dis = neigh.kneighbors_graph(X=fea, mode='distance').power(2)
        self.logger.info('%.2f s, Finished the calculation of Dis, begin to calculate W', timeit.default_timer() - self.tic)

        # Calculate W
        # indexDis = Dis.argsort(axis=1)[:, 0: Ks + 1]
        # sortedDis = np.sort(Dis, axis=1)[:, 0: Ks + 1]
        sig2 = Dis.sum()/(self.Ks * Ns) * self.a
        # XI = np.transpose(np.tile(range(Ns), (Ks + 1, 1)))
        # W = coo_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
        #                shape=(Ns, Ns)).toarray()
        W_scr = (-Dis * (1 / sig2)).expm1()
        I = csr_matrix((np.ones(self.Ks * Ns), W_scr.nonzero()))
        W = (W_scr+I).toarray() # exp-1?????

        self.logger.info('%.2f s, Finished the calculation of W, begin to calculate A', timeit.default_timer() - self.tic)

        # Calculate A
        asymA = np.zeros((self.Nc, self.Nc))
        A = np.zeros((self.Nc, self.Nc))
        for j in range(self.Nc):
            self.logger.info('%.2f s, Calculating A..., j: %d', timeit.default_timer() - self.tic, j)

            for i in range(j):
                if np.size(C[i], 0)!=0:
                    asymA[j, i] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1)) / math.pow(
                    np.size(C[i], 0), 2)
                if np.size(C[j], 0)!=0:
                    asymA[i, j] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1)) / math.pow(
                    np.size(C[j], 0), 2)
                A[i, j] = asymA[i, j] + asymA[j, i]
                A[j, i] = A[i, j]
        self.logger.info('%.2f s, Calculation of A based on clusters is completed', timeit.default_timer() - self.tic)

        return A, asymA

    def merge_cluster(self, A, asymA, C):

        # Find two clusters with the smallest loss
        Nc = np.size(C)
        np.fill_diagonal(A, - float('Inf'))
        indexA = A.argsort(axis=1)[:, ::-1][:, 0: self.Kc]
        sortedA = np.sort(A, axis=1)[:, ::-1][:, 0: self.Kc]

        minLoss = float('Inf')
        for i in range(Nc):
            loss = - (1 + self.l) * sortedA[i, 0] + sum(sortedA[i, 1: self.Kc]) * self.l / (self.Kc - 1)
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
        asymA[0: Nc, minIndex1] = asymA[0: Nc, minIndex1] * (1 + self.alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2) \
                                  + asymA[0: Nc, minIndex2] * (1 + self.alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)

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

    def model(self, data):
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

    def get_labels(self, C):
        # Generate sample labels
        labels = np.zeros(self.Ns, np.int)
        for i in range(np.size(C)):
            labels[C[i]] = i

        return labels


    def train(self):
        sess = tf.InteractiveSession()

        # Initialization with K-means
        self.logger.info('%.2f s, Begin K means...', timeit.default_timer() - self.tic)
        indexClusters = MiniBatchKMeans(self.Nc, max_iter=100, max_no_improvement=10, tol=0.0001).fit_predict(self.images) #optimize?????
        self.logger.info('%.2f s, K means is completed', timeit.default_timer() - self.tic)

        C = []
        for i in range(self.Nc):
            C.append([])
        for i in range(self.Ns):
            C[indexClusters[i]].append(i)

        A_clu, asymA = self.calA_cluster_based(self.images, C)

        ts = 0
        t = 0
        p = 0
        Np = np.ceil(self.eta * self.Nc)

        while self.Nc > self.K:
            self.logger.info('%.2f s, Timestep: %d', timeit.default_timer() - self.tic, t)
            t = t + 1
            self.Nc = self.Nc - 1
            A_clu, asymA, C = self.merge_cluster(A_clu, asymA, C)

            if False:
            #if t%10==1:
            #if t == ts + Np:
                indexA = A_clu.argsort(axis=1)[:, ::-1][:, 0: self.Kc]
                labels = self.get_labels(C)
                features = self.model(self.x_image)

                for i in range(self.epochs):
                    self.logger.info('%.2f s, Period: %d, epochs: %d', timeit.default_timer() - self.tic, p, i)
                    fea_batch = self.model(self.x_image_batch)
                    A_sam = self.calA_sample_based(fea_batch)
                    # sortedA, indexA = tf.nn.top_k(A_sam, k = Kc + 1, sorted=True)

                    # Loss function
                    loss = tf.Variable(tf.zeros([1]))
                    for j in range(self.batch_size):
                        Cid = labels[i * self.batch_size + j]
                        x_j = np.array(C[Cid])
                        Aij = tf.gather(tf.gather(A_sam, j), x_j[np.logical_and(x_j>=i*self.batch_size, x_j<j)], validate_indices=False)  # A[i, C[Cid]], here include itself  ???index outbound???
                        x_k = []
                        for n in range(self.Kc): #??????Kc cluster or samples??????
                            x_k = x_k + C[indexA[Cid][n]]
                            # sample = tf.gather(tf.gather(indexA, Cid), n + 1)
                            # if tf.reshape(tf.equal(tf.gather(labels, sample),Cid), [-1]):
                            #     x_k = x_k + sample
                        x_k = np.array(x_k)
                        Aik = tf.gather(tf.gather(A_sam, j), x_k[np.logical_and(x_k>=i*self.batch_size, x_k<j)], validate_indices=False)  # A[j, x_k]
                        loss = loss + - self.l / (self.Kc - 1) * (self.gamma * tf.reduce_sum(Aij) + tf.reduce_sum(Aik))

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
                    #batch = mnist.train.next_batch(batch_size)
                    batch = self.images[self.batch_size * i: self.batch_size * (i + 1), :]
                    sess.run(tf.initialize_all_variables())
                    train_step.run(feed_dict={self.x: batch})

                self.logger.info('%.2f s, Timestep: %d, Period: %d, finished cnn training', timeit.default_timer() - self.tic, t, p)
                fea = sess.run(features, feed_dict={self.x: self.images}) # Get the new feature representation
                A_clu, asymA = self.calA_cluster_based(fea, C) # Update A based on the new feature representation
                ts = t
                Np = np.ceil(self.eta * self.Nc)
                p = p + 1

        labels = self.get_labels(C)
        labels_maped = bestMap(self.gnd, labels)
        AC = float(np.count_nonzero(self.gnd == labels_maped)) / np.size(self.gnd, 0) # Evaluate AC: accuracy
        MIhat = MutualInfo(self.gnd, labels_maped) # Evaluate MIhat: nomalized mutual information
        self.logger.info('%.2f s, AC: %f, MIhat: %f', timeit.default_timer() -self.tic, AC, MIhat)
