import os
import sys
import numpy as np
import networkx as nx
import igraph as ig

sys.setrecursionlimit(1000000)
sys.path.append('..\\..')
from evaluate import *

basepath = "..\\..\\data\\" + str(sys.argv[1]) + "\\"


def networkx_getnetwork():
    G = nx.Graph()
    f = open(basepath + "network.dat", "r")
    data = f.readlines()
    f.close()
    number_of_nodes = int(data[len(data) - 1].split("\t")[0])
    for i in range(number_of_nodes):
        G.add_node(i)
    for string in data:
        firstnode = int(string.split("\t")[0])
        secodnode = int(string.split("\t")[1].split("\n")[0])
        firstnode = firstnode - 1
        secodnode = secodnode - 1
        G.add_edge(firstnode, secodnode)
    return G


def initiEmbeddings():
    if not os.path.exists(basepath + 'n2v_embeddings_rc.emb'):
        if not os.path.exists(basepath + 'embeddings.emb'):
            EMBEDDING_FILENAME = basepath + 'embeddings.emb'
            try:
                graph = nx.read_gml(basepath + "network.gml", label=None)
            except nx.exception.NetworkXError:
                graph = networkx_getnetwork()

            node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            model.wv.save_word2vec_format(EMBEDDING_FILENAME)

        with open(basepath + 'embeddings.emb') as f1:
            nodeandvec = {}
            _ = f1.readline()
            line = f1.readline()
            while line:
                words = line.split(" ")
                node = words[0]
                nodeandvec[int(node)] = np.array([float(num) for num in words[1:]])
                line = f1.readline()
                # print(str(node))
        f1.close()
        Embeddings = []
        for i in range(len(nodeandvec)):
            Embeddings.append(nodeandvec[i])
        np.savetxt(basepath + 'n2v_embeddings_rc.emb', np.array(Embeddings))
        return np.array(Embeddings)
    else:
        Emneddings = np.loadtxt(basepath + 'n2v_embeddings_rc.emb')
        return Emneddings


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
    print(times)
    times = times + 1
    if state:
        upperCenters = getClusterCenter(True, g, k, Embeddings)
    for v in g.vs:
        C2D = []
        for center in upperCenters:
            C2D.append(distEclud(Embeddings[v.index], center))
        v['cluster'] = C2D.index(min(C2D))
    new_centers = getClusterCenter(False, g, k, Embeddings)
    #np.savetxt()
    if (upperCenters == new_centers).all():
        return
    return kmeans(False, k, g, Embeddings, new_centers, times)


def main():
    community_file_path = basepath + '\\community_v2.txt'
    rcam_file = basepath + '\\RCAM.txt'
    nmi_result_file = basepath + '\\NMI' + '_N2V.txt'
    g = ig.Graph.Read_GML(basepath + "network.gml")
    RCAM, k = getRCAM_and_CN(len(g.vs), community_file_path, rcam_file)

    print ('Initialize embedding starts.')
    Embeddings = initiEmbeddings()
    print ('Initialize embedding ends.')
    print ('K-Means cluster starts.')
    kmeans(True, k, g, Embeddings, None, 0)
    print ('K-Means cluster ends.')
    print ('NMI calculation starts.')
    s = [v['cluster'] for v in g.vs]
    N2V_CAM = torch.zeros(len(s), k)
    for node_index, item in enumerate(s):
        N2V_CAM[node_index, item] = 1
    np.savetxt(basepath + 'N2V_CAM.txt', N2V_CAM.detach().numpy())
    nmi_score = overlapping_nmi(RCAM.float(), N2V_CAM.float())
    save_nmi(nmi_result_file, nmi_score.item())
    print ('NMI calculation ends.')
    print('NMI:', nmi_score.item())


if __name__ == '__main__':
    main()
