# joint-cluster-cnn

An unofficial implementation of JULE (Joint Unsupervised Learning of Deep Representations and Image Clusters) in TensorFlow.

# File Structure
Main class:
	joint_cluster_cnn.py

Test file:
	test.py

# Result
|Dataset | MNIST-test | USPS ｜ COIL-20 | COIL-100 | UMist | 
|---|---|---|---|---|---| 
|JC-NL | 0.849 | 0.883 ｜ 0.904 | 0 | 0.750 |
|JC-SF | 0.861/0.876 | 0.875/0.858 | 0.967/1.000 | 0.978 | 0.840/0.880 | 
|JC-RC | **0.900**/0.915 | **0.944**/0.913 | **0.983**/1.000 | **0.985** | 0.849/0.877 | 
|OURS-NL | 0.841 | 0.911 | 0.933 | 0.957 | 0.767 | 
|OURS-SF | 0.859 | 0.901 | 0.956 | 0.964 | **0.945** | 
|OURS-RC | 0.888 | 0.921 | 0.964 | 0.915 | 0.941 |
