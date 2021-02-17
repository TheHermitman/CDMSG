# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import igraph as ig
from evaluate import *
import sys
import time


markov_time = 1
base_path = ".\\data\\" + str(sys.argv[1])
gml_file_path = base_path + '\\network.gml'
community_file_path = base_path + '\\community_v2.txt'
rcam_file = base_path + '\\RCAM.txt'
g = ig.Graph.Read_GML(gml_file_path)
num_node = len(g.vs)
adj_m = g.get_adjacency(type=2, eids=False)
adj = []
for row in adj_m:
    for item in row:
        adj.append(item)
adj = np.mat(np.array(adj).reshape(num_node, num_node))
RCAM, num_community = getRCAM_and_CN(num_node, community_file_path, rcam_file)
num_hidden = 128
threshold = np.sqrt(-np.log(1 - (2 * num_node / (num_node * (num_node - 1)))))
num_epoch = 40
lr = 0.01
dropout_on = True
weight_decay_on = True
nmi_result_file = base_path + '\\NMI' + '_MT_' + str(markov_time) + '' + '_THRESHOLD_' + str(threshold).replace('.','_') + '_hardmax' + '.txt'


def m_pow(m, t):
    m_p = torch.eye(len(m))
    for i in range(t):
        m_p = torch.mm(m_p, m)
    return m_p


def pre_loss_func():
    t = markov_time
    A = torch.from_numpy(adj).float()  # 邻接矩阵
    # np.savetxt('A.txt', A.detach().numpy())
    m = torch.sum(A) / 2
    d = torch.sum(A, 1)
    pai = d / (2 * m.item())
    L = torch.diag(pai)  # pai的对角阵
    D = torch.diag(d).float()
    D_ = torch.inverse(D)
    M = torch.mm(D_, A)
    firstmd = torch.mm(L, m_pow(M, t))
    return A, m, d, pai, L, D, D_, M, firstmd


A, m, d, pai, L, D, D_, M, firstmd = pre_loss_func()


def norm(adj):
    adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    degree = np.diag(np.power(degree, -0.5))
    adj_n = degree.dot(adj).dot(degree)
    adj_n = (torch.from_numpy(adj_n)).float()
    return adj_n


class GCNLayer(nn.Module):
    def __init__(self, infeature_size, outfeature_size):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(infeature_size, outfeature_size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(norm(adj), support)
        if active:
            #output = F.relu(output)
            output = F.relu(output) + 0.0001
        return output


class GCN(nn.Module):
    def __init__(self, input_size, output_size, num_community):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_size, output_size)
        self.dropout = torch.nn.Dropout(0.5)
        self.gcn2 = GCNLayer(output_size, num_community)

    def forward(self, features, adj):
        h = self.gcn1(features, adj)
        do_h = self.dropout(h)
        o = self.gcn2(do_h, adj)
        return o


def loss_func(H, pai):
    # A = torch.from_numpy(adj).float()  # 邻接矩阵
    # # np.savetxt('A.txt', A.detach().numpy())
    # m = torch.sum(A) / 2
    # d = torch.sum(A, 1)
    # pai = d / (2 * m.item())
    # L = torch.diag(pai)  # pai的对角阵
    # D = torch.diag(d).float()
    # D_ = torch.inverse(D)
    # M = torch.mm(D_, A)
    # firstmd = torch.mm(L, m_pow(M, t))
    pai = pai.reshape(1, len(pai))
    secondmd = torch.mm(pai.T, pai)
    med = firstmd - secondmd
    R = torch.mm(torch.mm(H.t(), med), H)
    loss = torch.trace(R)
    return -loss


def hardmax(H):
    return (H.T / H.sum(axis=1)).T


def run():
    model = GCN(num_node, num_hidden, num_community)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    adj_torch = torch.from_numpy(adj).float()
    for epoch in range(num_epoch):
        H = model(adj_torch, adj_torch)
        # H = F.softmax(H, dim=1)
        H = hardmax(H)
        loss = loss_func(H, pai)
        optimizer.zero_grad()
        loss.backward()
        PCAM = getPredictCAM(H, threshold)
        nmi_score = overlapping_nmi(RCAM.float(), PCAM.float())
        print('Epoch:{}  Loss:{}  NMI:{}'.format(str(epoch), str(loss.item()), str(nmi_score.item())))
        # 保存结果
        optimizer.step()
    save_nmi(nmi_result_file, nmi_score.item())
    return


if __name__ == '__main__':
    run()
