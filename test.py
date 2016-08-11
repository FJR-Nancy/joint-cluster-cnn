# from test_cluster import *
#
# test_cluster('mnist-test', 'no')


from joint_cluster_cnn import *

joint_cluster_cnn('usps', testmode = False).train()
#joint_cluster_cnn('mnist').train()
