import os
import sys
import numpy as np
from ge import LINE
import networkx as nx
import igraph as ig
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from ge.classify import read_node_label, Classifier

sys.path.append('..\\..')
from evaluate import *

sys.setrecursionlimit(1000000)
basepath = "..\\..\\data\\" + str(sys.argv[1]) + "\\"
times = 0


def initiEmbeddings():
    if not os.path.exists(basepath + 'line_embeddings.emb'):
        G = nx.read_gml(basepath + "network.gml", label=None)
        model = LINE(G, embedding_size=128, order='second')
        model.train(batch_size=1024, epochs=150, verbose=2)
        embeddings = model.get_embeddings()
        # print (embeddings)
        t = []
        for key in embeddings:
            t.append(embeddings[key])
            #print (embeddings[key])
        np.savetxt(basepath + 'line_embeddings.emb', np.array(t))

    # Emneddings = np.loadtxt(basepath + 'line_embeddings.emb')
    Embeddings = np.loadtxt(basepath + 'line_embeddings.emb')
    return Embeddings


def distEclud(vecA, vecB):
    i = 0
    if len(vecA) != len(vecB):
        return
    sum = 0
    while i < len(vecA):
        sum = sum + (float(vecA[i]) - float(vecB[i])) ** 2
        i = i + 1
    return sum ** 0.5


def getClusterCenter(initual, g, k, Embeddings):
    if initual:
        while True:
            centers = np.random.randint(0, len(g.vs), k)
            if len(set(centers)) == len(centers):
                break
        return Embeddings[centers]
    clusters = {}
    for v in g.vs:
        if v['cluster'] not in clusters:
            clusters[v['cluster']] = []
            clusters[v['cluster']].append(v.index)
        else:
            clusters[v['cluster']].append(v.index)
    centers = []
    for c in clusters:
        new_center = np.mean(Embeddings[clusters[c]], 0)
        centers.append(new_center)
    return np.array(centers)


def kmeans(state, k, g, Embeddings, upperCenters, times):
    times = times + 1
    #print (times)
    if state:
        upperCenters = getClusterCenter(True, g, k, Embeddings)
        np.savetxt('centerInitual.txt', upperCenters)
    for v in g.vs:
        C2D = []
        for center in upperCenters:
            C2D.append(distEclud(Embeddings[v.index], center))
        v['cluster'] = C2D.index(min(C2D))
    new_centers = getClusterCenter(False, g, k, Embeddings)
    if (upperCenters == new_centers).all():
        return
    return kmeans(False, k, g, Embeddings, new_centers, times)


def main():
    community_file_path = basepath + '\\community_v2.txt'
    rcam_file = basepath + '\\RCAM.txt'
    nmi_result_file = basepath + '\\NMI' + '_LINE.txt'
    g = ig.Graph.Read_GML(basepath + "network.gml")
    RCAM, k = getRCAM_and_CN(len(g.vs), community_file_path, rcam_file)

    print ('Initialize embedding starts.')
    Embeddings = initiEmbeddings()
    print ('Initialize embedding ends.')
    print ('K-Means cluster starts.')
    kmeans(True, k, g, Embeddings, None, times)
    print ('K-Means cluster ends.')
    print ('NMI calculation starts.')
    s = [v['cluster'] for v in g.vs]
    LINE_CAM = torch.zeros(len(s), k)
    for node_index, item in enumerate(s):
        LINE_CAM[node_index, item] = 1
    np.savetxt(basepath + 'LINE_CAM.txt', LINE_CAM.detach().numpy())
    nmi_score = overlapping_nmi(RCAM.float(), LINE_CAM.float())
    save_nmi(nmi_result_file, nmi_score.item())
    print ('NMI calculation ends.')
    print('NMI:', nmi_score.item())


if __name__ == '__main__':
    main()
