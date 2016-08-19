import numpy as np

def MutualInfo(L1,L2):
    # mutual information

    L1 = np.array(L1)
    L1 = L1.flatten()

    L2 = np.array(L2)
    L2 = L2.flatten()

    if np.size(L1) != np.size(L2):
        print('size(L1) must == size(L2)')
    
    Label1 = np.unique(L1)
    nClass = np.size(Label1, 0)
    Label2 = np.unique(L2)
    nClass2 = np.size(Label2, 0)
    
    if nClass2 < nClass:
         # smooth
         L1 = np.concatenate((L1, Label1), 0)
         L2 = np.concatenate((L2, Label1), 0)
    elif nClass2 > nClass:
         # smooth
         L1 = np.concatenate((L1, Label2), 0)
         L2 = np.concatenate((L2, Label2), 0)
    
    G = np.zeros((nClass, nClass))
    for i in range (nClass):
        for j in range(nClass):
            G[i,j] = sum(np.logical_and(L1 == Label1[i], L2 == Label2[j]))
        
    sumG = sum(sum(G))
    
    P1 = np.sum(G,1)
    P1 = P1/sumG
    P2 = np.sum(G,0)
    P2 = P2/sumG
    if sum(P1==0) > 0 or sum(P2==0) > 0:
        # smooth
        print('Smooth fail!')
    else:
        H1 = sum(-P1 * np.log2(P1))
        H2 = sum(-P2 * np.log2(P2))
        P12 = G/sumG
        PPP = P12 /np.tile(P2,(nClass,1)) /np.tile(P1,(nClass, 1)).transpose()
        PPP[abs(PPP) < 1e-12] = 1
        MI = sum(P12.flatten() * np.log2(PPP.flatten()))
        MIhat = MI / max(H1,H2)
        #MIhat = real(MIhat)

    return MIhat


