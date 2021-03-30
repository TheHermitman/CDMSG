import os
import warnings

import torch
import numpy as np


def save_nmi(file_path, result):
    with open(file_path, mode='a') as f:
        f.write('NMI: ' + str(result) + '\n')
        f.close()
    return

def getPredictCAM(F, threshold):
    one = torch.from_numpy(np.array([1]))
    zero = torch.from_numpy(np.array([0]))
    PCAM = torch.where(F >= threshold, one, zero)
    return PCAM


def getRCAM_and_CN(num_node, file, rcam_file):
    if os.path.exists(rcam_file):
        # RCAM = np.loadtxt(racm_file)
        RCAM = torch.from_numpy(np.loadtxt(rcam_file))
    else:
        communities = []
        with open(file, 'r') as file:
            line = file.readline()
            while line:
                communities.append([int(node) for node in line.split('	')])
                line = file.readline()
            file.close()

        RCAM = torch.zeros(num_node, len(communities)) # 输入图中节点数量和图形成的社区数量，返回？
        # print(m.shape)
        for i, c in enumerate(communities):
            for n in c:
                RCAM[n, i] = 1
        np.savetxt(rcam_file, RCAM)
    # print(m)
    return RCAM, int(RCAM.shape[- 1])


def overlapping_nmi(X, Y):
    # X:RCAM
    # Y:PCAM
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix") # 二元矩阵
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forgot to transpose the F matrix") # 忘了转置F矩阵
    X = X.T
    Y = Y.T

    def cmp(x, y):
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)

    def H_uncond(X):
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)
