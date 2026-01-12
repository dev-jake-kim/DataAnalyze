import copy
from pathlib import Path
import numpy as np
import json
import heapq

cluster_hierarchy = [32, 16]
MERGE_THRESHOLD = 8
result = set()


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


def make_clusters():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'

    # 디렉토리 생성 (없을 경우)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    demands = np.load(data_dir / 'demands_array.npy').T
    dtw_dist = np.load(data_dir / 'dtw_similarity.npy')

    print(f"Loaded demands shape: {demands.shape}")
    print(f"Loaded DTW distance shape: {dtw_dist.shape}")

    # 초기 클러스터 생성
    clusters = [Cluster(idx) for idx in range(demands.shape[0])]
    hq = []
    for i in range(dtw_dist.shape[0]):
        for j in range(i + 1, dtw_dist.shape[1]):
            heapq.heappush(hq, (dtw_dist[i, j], i, j))
    print(f"heap size: {len(hq)}")
    hierarchy_pos = 0

    for_cnt = 0
    # 클러스터 병합
    while hq:
        for_cnt += 1
        dist, i, j = heapq.heappop(hq)
        active_sizes = [c.size for c in clusters if c.is_root()]
        min_size = min(active_sizes)
        if min_size * MERGE_THRESHOLD <= clusters[i].find_set().size + clusters[j].find_set().size:
            continue
        # print(f"min_size: {min_size}, merging clusters {i} and {j} size: {clusters[i].size}, {clusters[j].size} with distance: {dist}")
        clusters[i].union(clusters[j])

        dict_clusters = {}
        for cluster in clusters:
            root = cluster.find_set().component
            if root not in dict_clusters:
                dict_clusters[root] = []
            dict_clusters[root].append(cluster.component)
        #print(f"Number of clusters: {len(dict_clusters)}")
        if len(dict_clusters) <= cluster_hierarchy[hierarchy_pos]:
            hierarchy_pos += 1
            print(f"totla cluster: {len(dict_clusters)}")
            cnt = 0
            for cluster_key in dict_clusters.keys():
                mean_demands = np.mean(np.sum(demands[dict_clusters[cluster_key]], axis=0))
                print(f"cluster {cluster_key}: size {len(dict_clusters[cluster_key])}, mean demand: {mean_demands}", end='; ')
                cnt += 1
                if cnt % 3 == 0:
                    print()
            print()
            for cls in dict_clusters.values():
                result.add(tuple(cls))

            if hierarchy_pos >= len(cluster_hierarchy):
                print(f"pos: {hierarchy_pos}, len: {len(cluster_hierarchy)}")
                break
    return result
            
