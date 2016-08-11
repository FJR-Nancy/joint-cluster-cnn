from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.cluster import KMeans
from bestMap import *
from MutualInfo import *

def cluster(fea, gnd, K, imageSize, feature):
    '''
    The implementation of agglomerative clustering algorithim.
        fea - Each row is a sample
        gnd - The label of the sample
        K - The target clustering number
        imageSize - The size of the original image
        feature - The representation of image would be used
    '''
    
    Ks = 20 # the number of nearest neighbours of a sample
    Kc = 5 # the number of nearest clusters of a cluster
    a = 1.0
    l = 1.0 #lambda
    alpha = 0
    
    Ns = np.size(fea, 0)
    
    if 'no' not in feature:
        for i in range(Ns):
            image_i = np.reshape(fea[i,:], imageSize, imageSize)
            if 'lbp' in feature:
                fea[i,:] = local_binary_pattern(image_i, 8, 1, method='default')
            elif 'hog' in feature:
                fea[i,:] = hog(image_i, 8, 1, method='default')
    
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
    indexDis = Dis.argsort(axis = 1)[:, 0 : Ks + 1]
    sortedDis = np.sort(Dis, axis = 1)[:, 0 : Ks + 1]

    sig2 = np.mean(np.mean(sortedDis)) * a
    XI = np.transpose(np.tile(range(Ns), (Ks + 1, 1)))
    W = coo_matrix((np.exp(-sortedDis.flatten()*(1 / sig2)), (XI.flatten(), indexDis.flatten())), shape = (Ns, Ns)).toarray()
    
    # Calculate A
    asymA = np.zeros((Nc,Nc))
    A = np.zeros((Nc,Nc))
    for j in range(Nc):
        for i in range(j - 1):
            asymA[j, i] = np.dot(np.sum(W[C[i], :][ :, C[j]], 0), np.sum(W[C[j], :][ :, C[i]], 1)) / math.pow(np.size(C[i], 0), 2)
            asymA[i, j] = np.dot(np.sum(W[C[j], :][ :, C[i]], 0), np.sum(W[C[i], :][ :, C[j]], 1)) / math.pow(np.size(C[j], 0), 2)
            A[i, j] = asymA[i, j] + asymA[j, i]
            A[j, i] = A[i, j]

    while Nc > K:
    
        # Find two clusters with the smallest loss
        np.fill_diagonal(A, - float('Inf'))
        indexA = A.argsort(axis = 1)[:, ::-1][:, 0: Kc]
        sortedA = np.sort(A, axis = 1)[:, ::-1][:, 0: Kc]

        minLoss = float('Inf')
        for i in range(Nc):
            loss = - (1 + l ) * sortedA[i, 0] + sum(sortedA[i, 1: Kc]) * l / (Kc - 1)
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
        asymA[minIndex1, 0 : Nc] = asymA[minIndex1, 0 : Nc] + asymA[minIndex2, 0 : Nc]
        len1 = np.size(cluster1, 0)
        len2 = np.size(cluster2, 0)
        asymA[0 : Nc, minIndex1] = asymA[0 : Nc, minIndex1] * (1 + alpha) * math.pow(len1, 2) / math.pow(len1 + len2, 2)\
                + asymA[0 : Nc, minIndex2] *(1 + alpha) * math.pow(len2, 2) / math.pow(len1 + len2, 2)

        A[minIndex1,:] = asymA[minIndex1,:] + asymA[:, minIndex1]
        A[:, minIndex1] = A[minIndex1,:]

        # replace the second cluster to be merged with the last cluster of the cluster array
        if (minIndex2 != Nc):
            C[minIndex2] = C[-1]
            asymA[0 : Nc, minIndex2] = asymA[0 : Nc, Nc - 1]
            asymA[minIndex2, 0 : Nc] = asymA[Nc - 1, 0 : Nc]
            A[0 : Nc, minIndex2] = A[0 : Nc, Nc - 1]
            A[minIndex2, 0 : Nc] = A[Nc - 1, 0 : Nc]

        # remove the last cluster
        C.pop()
        asymA[0 : Nc, Nc - 1] = 0
        asymA[Nc - 1, 0 : Nc] = 0
        A[0 : Nc, Nc - 1] = 0
        A[Nc - 1, 0 : Nc] = 0

        Nc = Nc - 1
    
    # generate sample labels
    labels = np.ones((Ns, 1))
    for i in range(np.size(C)):
        labels[C[i]] = i

    labels = bestMap(gnd, labels)
    #labels = bestMap(labels, gnd)
    # evaluate AC: accuracy
    AC =  float(np.count_nonzero(gnd == labels)) / np.size(gnd, 0)
    # evaluate MIhat: nomalized mutual information
    MIhat = MutualInfo(gnd, labels)
    
    print AC
    print MIhat

    return labels