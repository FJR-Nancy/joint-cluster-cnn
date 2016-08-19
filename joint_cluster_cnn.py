import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, batch_norm, fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import fetch_mldata
import math
from bestMap import *
from MutualInfo import *
import timeit
import logging
from PIL import Image
import os.path

class joint_cluster_cnn():

    Ks = 20  # the number of nearest neighbours of a sample
    Kc = 5  # the number of nearest clusters of a cluster
    a = 1.0
    l = 1.0  # lambda
    #alpha = 0
    eta = 0.9
    epochs = 1 #20
    batch_size = 100
    gamma_tr = 2.0 # weight of positive pairs in weighted triplet loss.
    margin = 0.2 # margin for weighted triplet loss
    num_nsampling = 20 # number of negative samples for each positive pairs to construct triplet.

    def __init__(self, dataset, RC=False, testmode=False, updateCNN=True, eta=0.9):
        self.sess = tf.InteractiveSession()

        self.dataset = dataset
        self.RC = RC
        self.updateCNN = updateCNN
        self.testmode = testmode
        self.eta = eta
        self.tic = timeit.default_timer()

        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename='/logfile/logfile.log',
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
        if 'mnist-test' in dataset:
            mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
            self.images = mnist.test.images
            self.gnd = mnist.test.labels
            image_size1 = 28
            image_size2 = 28
            self.K = 10
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 1])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 1])
            self.logger.info('%.2f s, Finished extracting MNIST-test dataset', timeit.default_timer() - self.tic)
        elif 'mnist-full' in dataset:
            mnist = input_data.read_data_sets('MNIST_data', dtype=tf.uint8, one_hot=False)
            images = np.concatenate((mnist.test.images, mnist.validation.images), axis=0)
            gnd = np.concatenate((mnist.test.labels, mnist.validation.labels), axis=0)
            self.images = np.concatenate((mnist.train.images, images), axis=0)
            self.gnd = np.concatenate((mnist.train.labels, gnd), axis=0)
            image_size1 = 28
            image_size2 = 28
            self.K = 10
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 1])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 1])
            self.logger.info('%.2f s, Finished extracting MNIST-full dataset', timeit.default_timer() - self.tic)
        elif 'usps' in dataset:
            usps = fetch_mldata("USPS")
            self.images = usps.data
            self.gnd = usps.target.astype(np.int) - 1
            image_size1 = 16
            image_size2 = 16
            self.K = 10
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 1])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 1])
            self.logger.info('%.2f s, Finished extracting USPS dataset', timeit.default_timer() - self.tic)
        elif 'coil20' in dataset:
            image_size1 = 128
            image_size2 = 128
            self.images = np.zeros((20*72, image_size1 * image_size2), np.uint8)
            self.gnd = np.zeros(20*72, np.uint8)
            path = '../dataset/coil-20-proc/obj'
            for i in range(20):
                for j in range(72):
                    img_name = path + str(i+1) + '__' + str(j) + '.png'
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[i*72+j, :] = np.reshape(img_data, (1, image_size1 * image_size2))
                    self.gnd[i*72+j] = i
            self.K = 20
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 1])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 1])
            self.logger.info('%.2f s, Finished extracting COIL20 dataset', timeit.default_timer() - self.tic)
        elif 'coil100' in dataset:
            image_size1 = 128
            image_size2 = 128
            self.images = np.zeros((100*72, image_size1 * image_size2 * 3), np.uint8)
            self.gnd = np.zeros(100*72, np.uint8)
            path = '../dataset/coil-100/obj'
            for i in range(100):
                for j in range(72):
                    img_name = path + str(i+1) + '__' + str(j*5) + '.png'
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[i*72+j, :] = np.reshape(img_data, (1, image_size1 * image_size2 * 3))
                    self.gnd[i*72+j] = i
            self.K = 100
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2 * 3])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 3])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 3])
            self.logger.info('%.2f s, Finished extracting COIL100 dataset', timeit.default_timer() - self.tic)
        elif 'umist' in dataset:
            image_size1 = 112
            image_size2 = 92
            self.images = np.zeros((575, image_size1 * image_size2), np.uint8)
            self.gnd = np.zeros(575, np.uint8)
            path = '../dataset/umist/'
            n = 0
            for i in range(20):
                j = 0
                while True:
                    img_name = path + chr(97 + i) + str(j + 1) + '.pgm' # ord('a')=97
                    if not os.path.isfile(img_name):
                        break
                    img = Image.open(img_name)
                    img.load()
                    img_data = np.asarray(img, dtype=np.uint8)
                    self.images[n, :] = np.reshape(img_data, (1, image_size1 * image_size2))
                    self.gnd[n] = i
                    j += 1
                    n += 1
            self.K = 20
            self.Ns = np.size(self.images, 0)
            self.x = tf.placeholder(tf.float32, shape=[None, image_size1 * image_size2])
            self.x_image = tf.reshape(self.x, [self.Ns, image_size1, image_size2, 1])
            self.x_image_batch = tf.reshape(self.x, [self.batch_size, image_size1, image_size2, 1])
            self.logger.info('%.2f s, Finished extracting UMist dataset', timeit.default_timer() - self.tic)
        self.num_batch = self.Ns / self.batch_size

    def calA_cluster_based(self, fea, C):

        # Calculate Dis
        self.logger.info('%.2f s, Begin to fit neighbour graph', timeit.default_timer() - self.tic)
        neigh = NearestNeighbors(n_neighbors=self.Ks+1, n_jobs=-1).fit(fea)
        self.logger.info('%.2f s, Finished fitting, begin to calculate Dis', timeit.default_timer() - self.tic)
        Dis = neigh.kneighbors_graph(X=fea, mode='distance').power(2)
        self.logger.info('%.2f s, Finished the calculation of Dis, begin to calculate W', timeit.default_timer() - self.tic)

        # Calculate W
        sig2 = Dis.sum()/(self.Ks * self.Ns) * self.a
        if sum(Dis.data==0) == self.Ns:
            W_scr = (-Dis * (1 / sig2)).expm1()
            I = csr_matrix((np.ones(self.Ks * self.Ns), W_scr.nonzero()))
            W = (W_scr+I).toarray()
        else:
            indexDis = Dis.toarray().argsort(axis=1)[:, ::-1][:, 0: self.Ks]
            sortedDis = np.sort(Dis.toarray(), axis=1)[:, ::-1][:, 0: self.Ks]
            XI = np.transpose(np.tile(range(self.Ns), (self.Ks, 1)))
            W = csr_matrix((np.exp(-sortedDis.flatten() * (1 / sig2)), (XI.flatten(), indexDis.flatten())),
                           shape=(self.Ns, self.Ns)).toarray()
        self.logger.info('%.2f s, Finished the calculation of W, begin to calculate A', timeit.default_timer() - self.tic)

        # Calculate A
        asymA = np.zeros((self.Nc, self.Nc))
        A = np.zeros((self.Nc, self.Nc))
        for i in range(self.Nc):
            self.logger.info('%.2f s, Calculating A..., i: %d', timeit.default_timer() - self.tic, i)
            for j in range(i):
                if np.size(C[j], 0) != 0:
                    asymA[i, j] = np.dot(np.sum(W[C[i], :][:, C[j]], 0), np.sum(W[C[j], :][:, C[i]], 1)) / math.pow(
                    np.size(C[j], 0), 2)  # A(Ci -> Cj)
                if np.size(C[i], 0) != 0:
                    asymA[j, i] = np.dot(np.sum(W[C[j], :][:, C[i]], 0), np.sum(W[C[i], :][:, C[j]], 1)) / math.pow(
                    np.size(C[i], 0), 2)  # A(Cj -> Ci)
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
            if loss < minLoss and i != indexA[i, 0]:
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
        asymA[0: Nc, minIndex1] = (asymA[0: Nc, minIndex1] * len1 + asymA[0: Nc, minIndex2] * len2) / (len1 + len2)
        asymA[minIndex1, minIndex1] = 0
        A[minIndex1, :] = asymA[minIndex1, :] + asymA[:, minIndex1]
        A[:, minIndex1] = A[minIndex1, :]

        # Replace the second cluster to be merged with the last cluster of the cluster array
        if (minIndex2 != Nc):
            C[minIndex2] = C[-1]
            asymA[0: Nc, minIndex2] = asymA[0: Nc, Nc - 1]
            asymA[minIndex2, 0: Nc] = asymA[Nc - 1, 0: Nc]
            asymA[minIndex2, minIndex2] = 0
            A[0: Nc, minIndex2] = A[0: Nc, Nc - 1]
            A[minIndex2, 0: Nc] = A[Nc - 1, 0: Nc]

        # Remove the last cluster
        C.pop()
        asymA = asymA[0 : Nc - 1, 0 : Nc - 1]
        A = A[0 : Nc - 1, 0 : Nc - 1]

        return A, asymA, C

    def model(self, data):
        # CNN
        if 'mnist' in self.dataset: # 28 * 28
            net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 24 * 24
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 12 * 12
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 8 * 8
        elif 'usps' in self.dataset: # 16 * 16
            net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 12 * 12
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 6 * 6
        elif 'coil' in self.dataset: # 128 * 128
            net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 124 * 124
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 62 * 62
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 58 * 58
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 29 * 29
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 25 * 25
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 13 * 13
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 9 * 9
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 5 * 5
        elif 'umist' in self.dataset: # 112 * 92
            net = convolution2d(data, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 108 * 88
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 54 * 44
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 50 * 40
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 25 * 20
            net = convolution2d(net, num_outputs=50, kernel_size=(5, 5), stride=(1, 1), padding='VALID',
                                normalizer_fn=batch_norm, activation_fn=tf.nn.relu)  # 21 * 16
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')  # 10 * 8

        net = fully_connected(net, num_outputs=160, activation_fn=tf.nn.relu)
        net = tf.nn.l2_normalize(net, 3)

        # print net.get_shape()
        net_shape = net.get_shape().as_list()
        net_reshape = tf.reshape(net, [net_shape[0], net_shape[1] * net_shape[2] * net_shape[3]])

        return net_reshape

    def get_C(self, labels):
        num_sam = np.size(labels)
        labels_from_one = np.zeros(num_sam, np.int)

        idx_sorted = labels.argsort()
        labels_sorted = labels[idx_sorted]
        nclusters = 0
        label = -1
        for i in range(num_sam):
            if labels_sorted[i] != label:
                label = labels_sorted[i]
                nclusters += 1
            labels_from_one[idx_sorted[i]] = nclusters

        C = []
        for i in range(nclusters):
            C.append([])
        for i in range(num_sam):
            C[labels_from_one[i] - 1].append(i)
        return C

    def get_labels(self, C):
        # Generate sample labels

        labels = np.zeros(self.Ns, np.int)
        for i in range(np.size(C)):
            labels[C[i]] = i

        return labels

    def evaluation(self, C):
        labels = self.get_labels(C)

        # Evaluate AC: accuracy
        if np.size(np.unique(self.gnd))>=np.size(np.unique(labels)):
            labels = bestMap(self.gnd, labels)
            AC = float(np.count_nonzero(self.gnd == labels)) / np.size(self.gnd, 0)
        else:
            AC = float('nan')

        # Evaluate MIhat: nomalized mutual information
        MIhat = MutualInfo(self.gnd, labels)
        return AC, MIhat

    def get_triplet(self, labels_batch):
        num_sam = self.batch_size
        C_batch = self.get_C(labels_batch)
        nclusters = np.size(C_batch)

        if nclusters <= self.num_nsampling:
            num_neg_sampling = nclusters - 1
        else:
            num_neg_sampling = self.num_nsampling

        num_triplet = 0
        for i in range(nclusters):
            num_Ci = np.size(C_batch[i])
            num_triplet += num_Ci * (num_Ci - 1) * num_neg_sampling / 2

        if num_triplet == 0:
            return

        anc = np.zeros(num_triplet, np.int)
        pos = np.zeros(num_triplet, np.int)
        neg = np.zeros(num_triplet, np.int)

        id_triplet = 0
        for i in range(nclusters):
            if np.size(C_batch[i]) > 1:
                for m in C_batch[i]:
                    for n in C_batch[i][m + 1:]:
                        is_choosed = np.zeros(num_sam)
                        while id_triplet == 0 or id_triplet % num_neg_sampling != 0:
                            id_s = np.random.randint(num_sam)
                            if is_choosed[id_s] == 0 and labels_batch[id_s] != i:
                                anc[id_triplet] = m
                                pos[id_triplet] = n
                                neg[id_triplet] = id_s
                                is_choosed[id_s] = 1
                                id_triplet += 1

        return anc, pos, neg

    def train(self, fea, updateCNN):
        if not self.testmode:
            self.Nc = max(int(math.ceil(self.Ns / 50 )), self.K)
        else:
            self.Nc = self.K * 2
        self.logger.info('%.2f s, Nc is %d...', timeit.default_timer() - self.tic, self.Nc)

        # Initialization with K-means
        self.logger.info('%.2f s, Begin K means...', timeit.default_timer() - self.tic)
        indexClusters = MiniBatchKMeans(self.Nc, max_iter=100, max_no_improvement=10, tol=0.0001).fit_predict(fea) #optimize?????
        self.logger.info('%.2f s, K means is completed', timeit.default_timer() - self.tic)

        C = self.get_C(indexClusters)
        A_clu, asymA = self.calA_cluster_based(fea, C)

        ts = 0
        t = 0
        p = 0
        Np = np.ceil(self.eta * self.Nc)

        while self.Nc > self.K:
            self.logger.info('%.2f s, Timestep: %d, cluster...', timeit.default_timer() - self.tic, t)
            t += 1
            A_clu, asymA, C = self.merge_cluster(A_clu, asymA, C)
            self.Nc -= 1

            if updateCNN and t == ts + Np:
                labels = self.get_labels(C)
                for e in range(self.epochs):
                    self.logger.info('%.2f s, Period: %d, epoch: %d, training...', timeit.default_timer() - self.tic, p, e)
                    for i in range(self.num_batch):
                        self.logger.info('%.2f s, Period: %d, epoch: %d, batch: %d, training...', timeit.default_timer() - self.tic, p, e, i)
                        fea_batch = self.model(self.x_image_batch)
                        labels_batch = labels[self.batch_size * i: self.batch_size * (i + 1)]

                        # get trplets
                        anc_idx, pos_idx, neg_idx = self.get_triplet(labels_batch)
                        anc = tf.gather(fea_batch, anc_idx)
                        pos = tf.gather(fea_batch, pos_idx)
                        neg = tf.gather(fea_batch, neg_idx)

                        # triplet loss fuction
                        d_pos = tf.reduce_sum(tf.square(anc - pos), 1)
                        d_neg = tf.reduce_sum(tf.square(anc - neg), 1)
                        loss = tf.maximum(0., self.margin + self.gamma_tr * d_pos - d_neg)
                        loss = tf.reduce_mean(loss)

                        # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
                        global_step = tf.Variable(0)
                        # Decay once per epoch, using an exponential schedule starting at 0.01.
                        learning_rate = tf.train.exponential_decay(
                            0.01,  # Base learning rate.
                            global_step * self.batch_size,  # Current index into the dataset.
                            self.num_batch,  # Decay step.
                            0.99995,  # Decay rate.
                            staircase=True)
                        # learning_rate = learning_rate * torch.pow(1 + opt.gamma_lr * iter, - opt.power_lr) #inverse learning rate policy

                        train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)
                        batch = self.images[self.batch_size * i: self.batch_size * (i + 1), :]
                        self.sess.run(tf.initialize_all_variables())
                        train_step.run(feed_dict={self.x: batch})

                self.logger.info('%.2f s, Timestep: %d, Period: %d, finished cnn training', timeit.default_timer() - self.tic, t, p)

                # extract features, get the new feature representation
                num_fea = fea_batch.get_shape().as_list()[1]
                fea = np.zeros((self.Ns, num_fea))
                for i in range(self.num_batch):
                    batch = self.images[self.batch_size * i: self.batch_size * (i + 1), :]
                    fea[self.batch_size * i: self.batch_size * (i + 1), :] = self.sess.run(fea_batch, feed_dict={self.x: batch})
                    self.logger.info('%.2f s, Period: %d, epoch: %d, batch: %d, feature extracting...', timeit.default_timer() - self.tic, p, e, i)
                self.logger.info('%.2f s, Timestep: %d, Period: %d, finished extraction of feature', timeit.default_timer() - self.tic, t, p)

                # Update A based on the new feature representation
                A_clu, asymA = self.calA_cluster_based(fea, C)
                self.logger.info('%.2f s, Timestep: %d, Period: %d, finished recalculation of A based on the new feature representation', timeit.default_timer() - self.tic, t, p)
                ts = t
                Np = np.ceil(self.eta * self.Nc)
                p += 1

        return C, fea

    def run(self):
        C, fea = self.train(self.images, self.updateCNN)
        AC, MIhat = self.evaluation(C)
        self.logger.info('%.2f s, AC: %f, MIhat: %f', timeit.default_timer() - self.tic, AC, MIhat)

        if self.RC:
            self.logger.info('%.2f s, begin to re-run clustering', timeit.default_timer() - self.tic)
            C, fea = self.train(fea, updateCNN = False)
            AC, MIhat = self.evaluation(C)
            self.logger.info('%.2f s, AC: %f, MIhat: %f, after re-running clustering', timeit.default_timer() - self.tic, AC, MIhat)

