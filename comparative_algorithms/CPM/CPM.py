from collections import defaultdict
import networkx as nx
import sys

sys.path.append('..\\..')
from evaluate import *

# basepath = "..\\..\\data\\" + str(sys.argv[1]) + "\\"
basepath = "D:\work\SocialWork\CDMSG\data2\youtube\\"
G = nx.read_gml(basepath + "network.gml", label='id')


# g = Graph.Read_GML(basepath + "network.gml")


class CPM():

    def __init__(self, G, k=4):
        self._G = G
        self._k = k

    def execute(self):
        # find all cliques which size > k，找到所有大小大于k的团体
        cliques = list(nx.find_cliques(G))
        vid_cid = defaultdict(lambda: set())
        for i, c in enumerate(cliques):
            if len(c) < self._k:
                continue
            for v in c:
                vid_cid[v].add(i)

        # build clique neighbor, 建立派系邻居
        clique_neighbor = defaultdict(lambda: set())
        remained = set()
        for i, c1 in enumerate(cliques):
            # if i % 100 == 0:
            # print i
            if len(c1) < self._k:
                continue
            remained.add(i)
            s1 = set(c1)
            candidate_neighbors = set()
            for v in c1:
                candidate_neighbors.update(vid_cid[v])
            candidate_neighbors.remove(i)
            for j in candidate_neighbors:
                c2 = cliques[j]
                if len(c2) < self._k:
                    continue
                if j < i:
                    continue
                s2 = set(c2)
                if len(s1 & s2) >= min(len(s1), len(s2)) - 1:
                    clique_neighbor[i].add(j)
                    clique_neighbor[j].add(i)

                    # depth first search clique neighbors for communities，社区的深度优先搜索集团邻居
        communities = []
        for i, c in enumerate(cliques):
            if i in remained and len(c) >= self._k:
                # print 'remained cliques', len(remained)
                communities.append(set(c))
                neighbors = list(clique_neighbor[i])
                while len(neighbors) != 0:
                    n = neighbors.pop()
                    if n in remained:
                        # if len(remained) % 100 == 0:
                        # print 'remained cliques', len(remained)
                        communities[len(communities) - 1].update(cliques[n])
                        remained.remove(n)
                        for nn in clique_neighbor[n]:
                            if nn in remained:
                                neighbors.append(nn)
        return communities




if __name__ == '__main__':
    """
    community_file_path = basepath + '\\community_v2.txt'
    rcam_file = basepath + '\\RCAM.txt'
    nmi_result_file = basepath + '\\NMI' + '_CPM.txt'
    num_nodes = len(G.nodes)

    RCAM, k = getRCAM_and_CN(num_nodes, community_file_path, rcam_file)
    """
    # print('Network',str(sys.argv[1]),' CPM runs.')
    print('Network youtube CPM runs.')
    algorithm = CPM(G, k=4)
    communities = algorithm.execute()
    num_nodes = len(G.nodes)
    k = len(communities)
    CPM_CAM = torch.zeros(num_nodes, k)

    for c in communities:
        for n in c:
            CPM_CAM[n, communities.index(c)] = 1

    np.savetxt(basepath + 'CPM_CAM.txt', CPM_CAM.detach().numpy())

    np.savetxt(basepath + 'communities_CPM',communities)

    """
    nmi_score = overlapping_nmi(RCAM.float(), CPM_CAM.float())
    print(str(sys.argv[1]),'Ends.')
    print(str(sys.argv[1]), 'NMI', str(nmi_score.item()))
    save_nmi(nmi_result_file, nmi_score.item())
    """

    # print(communities)
    # for community in communities:
    #     print (community)
