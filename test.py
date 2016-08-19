# from test_cluster import *
#
# test_cluster('mnist-test', 'no')


from joint_cluster_cnn import *

#joint_cluster_cnn('usps', RC = True, testmode = False, updateCNN = True, eta = 0.9).run()
#joint_cluster_cnn('mnist-test', RC = True, testmode = False, updateCNN = True, eta = 0.9).run()
#joint_cluster_cnn('mnist-full', RC = True, testmode = True, updateCNN = True, eta = 0.2).run()
#joint_cluster_cnn('coil20', RC = True, testmode = False, updateCNN = True, eta = 0.2).run()
#joint_cluster_cnn('coil100', RC = True, testmode = False, updateCNN = True, eta = 0.2).run()
joint_cluster_cnn('umist', RC = True, testmode = True, updateCNN = True, eta = 0.2).run()
