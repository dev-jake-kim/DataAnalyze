import heapq
from typing import List
import numpy as np
from scipy import cluster
from graphnize.makeset import MERGE_THRESHOLD


class Cluster:
    def __init__(self, idx):
        self.component = idx
        self.parent = self
        self.size = 1

    def find_set(self):
        if self.is_root():
            return self
        self.parent = self.parent.find_set()
        return self.parent

    def is_root(self):
        return self.parent == self


    def union(self, other):
        if self.find_set() == other.find_set():
            return

        root1 = self.find_set()
        root2 = other.find_set()
        root2.parent = root1
        root1.size += root2.size

MERGE_THRESHOLD=8

def make_clusters(demands:np.ndarray, dtw_dist:np.ndarray, cluster_hierarchy:List[int]) -> np.ndarray:
    # 초기 클러스터 생성
    num_nodes = demands.shape[0]
    clusters = [Cluster(idx) for idx in range(num_nodes)]
    snap_shots = []
    hq = []
    for i in range(dtw_dist.shape[0]):
        for j in range(i + 1, dtw_dist.shape[1]):
            heapq.heappush(hq, (dtw_dist[i, j], i, j))
    print(f"heap size: {len(hq)}")
    hierarchy_pos = 0

    while hq and hierarchy_pos < len(cluster_hierarchy):
        dist, u, v = heapq.heappop(hq)
        root_u = clusters[u].find_set()
        root_v = clusters[v].find_set()

        if root_u != root_v:
            if clusters[u].find_set().size + clusters[v].find_set().size > MERGE_THRESHOLD:
                continue
            root_u.union(root_v)

        # 현재 클러스터 수 계산
        current_clusters = set()
        for cluster in clusters:
            current_clusters.add(cluster.find_set().component)

        if len(current_clusters) <= cluster_hierarchy[hierarchy_pos]:
            # print(f"Reached {cluster_hierarchy[hierarchy_pos]} clusters.")
            hierarchy_pos += 1
            cluster_dict = {}
            for c in clusters:
                root = c.find_set().component
                if root not in cluster_dict:
                    cluster_dict[root] = []
                cluster_dict[root].append(c.component)
            for key, value in cluster_dict.items():
                snap_shots.append(value)
                

    print(f"Final number of clusters: {len(snap_shots)}")
    #print(f"Clusters: {snap_shots}")

    return make_assign_matrix(num_nodes, snap_shots)

def make_assign_matrix(num_nodes: int, clusters: List[List[int]]) -> np.ndarray:
    assign = np.zeros((num_nodes, len(clusters)), dtype=int)
    for c_idx, cluster in enumerate(clusters):
        for node in cluster:
            assign[node, c_idx] = 1
    return assign
