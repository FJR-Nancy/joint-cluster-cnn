import numpy as np
from munkres import Munkres

def bestMap(L1,L2):
# bestmap: permute labels of L2 to match L1 as good as possible
#    [newL2] = bestMap(L1,L2);

	L1 = np.array(L1)
	L1 = L1.flatten()

	L2 = np.array(L2)
	L2 = L2.flatten()

	# if size(L1) ~= size(L2)
	#     error('size(L1) must == size(L2)')

	Label1 = np.unique(L1)
	nClass1 = np.size(Label1, 0)
	Label2 = np.unique(L2)
	nClass2 = np.size(Label2, 0)

	nClass = max(nClass1,nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		for j in range(nClass2):
			G[i,j] = np.count_nonzero(np.logical_and(L1 == Label1[i], L2 == Label2[j]))

	m = Munkres()
	c = m.compute(-G)
	#[c,t] = hungarian(-G)
	newL2 = np.zeros(L2.size)
	for i in range(nClass2):
	    #newL2[L2 == Label2[i]] = Label1[c[i][1]]
		newL2[L2 == Label2[c[i][1]]] = Label1[c[i][0]]
	
	return newL2
