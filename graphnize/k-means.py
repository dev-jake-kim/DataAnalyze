from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm  # 진행률 표시
import matplotlib
matplotlib.use("Agg")  # 서버/헤드리스 환경에서도 저장 가능
import matplotlib.pyplot as plt

from outjson import OutJson

# 하이퍼파라미터
K = 200           # 초기 클러스터 개수
R = 2500          # 클러스터 중심으로부터 허용 반경 (이상은 제거)
POW = 183*2        # 클러스터 최소 데이터 수 (이하 클러스터 제거)
MAX_NODE_DIST = 8000  # 노드 간 연결 최대 거리(이상은 미연결)
MAX_ITER = 100    # K-means 반복 횟수
SEED = 42         # 재현성
NUM_OF_DAYS = 182

TRUST_RANGE = 3.03
CONNECTION_THRESHOLD = 0.1


def _init_centroids_kpp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """간단한 k-means++ 초기화."""
    n = X.shape[0]
    # 첫 중심 무작위 선택
    centroids = np.empty((k, 2), dtype=float)
    i0 = rng.integers(0, n)
    centroids[0] = X[i0]
    # 나머지 선택
    closest_sq = np.sum((X - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = closest_sq / closest_sq.sum() if closest_sq.sum() > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centroids[i] = X[idx]
        d2 = np.sum((X - centroids[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, d2)
    return centroids


def kmeans_fit(X: np.ndarray, k: int, max_iter: int = MAX_ITER, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    간단한 K-means 학습
    - 입력: X (N,2), k
    - 출력: (centroids(K,2), labels(N,))
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)  # 안전 장치
    if k <= 0:
        return np.zeros((0, 2)), np.full(n, -1, dtype=int)

    # 초기 중심 (k-means++)
    centroids = _init_centroids_kpp(X, k, rng)

    labels = np.full(n, -1, dtype=int)
    for _ in tqdm(range(max_iter), desc="K-means iter", unit="it"):
        # 할당 단계
        dists = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))  # (N,K)
        new_labels = np.argmin(dists, axis=1)

        # 수렴 검사
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 업데이트 단계 (빈 클러스터 처리 포함)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # 빈 클러스터: 임의 재배치
                centroids[j] = X[rng.integers(0, n)]
            else:
                centroids[j] = X[mask].mean(axis=0)

    return centroids, labels


def filter_by_radius(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    반경 기준으로 포인트 필터링.
    - 반환: kept_mask(N,), dists(N,)
    """
    # 각 포인트의 현재 중심까지 거리
    dists = np.linalg.norm(X - centroids[labels], axis=1)
    kept_mask = dists < radius  # R 이상은 제거
    return kept_mask, dists


def drop_small_clusters(labels: np.ndarray, kept_mask: np.ndarray, min_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    작은 클러스터 제거. 반환: cluster_keep(K,), labels_after_small_drop(N,) (작은 클러스터는 -1)
    """
    valid = labels.copy()
    valid[~kept_mask] = -1
    # 남아있는 포인트 기준 클러스터 크기 집계
    k = labels.max() + 1 if labels.size else 0
    cluster_sizes = np.zeros(k, dtype=int)
    for j in range(k):
        cluster_sizes[j] = np.sum(valid == j)
    cluster_keep = cluster_sizes > min_points  # 이하 제거

    # 작은 클러스터 소속 포인트 => -1
    for j in range(k):
        if not cluster_keep[j]:
            valid[valid == j] = -1
    return cluster_keep, valid


def reassign_removed(X: np.ndarray, centroids: np.ndarray, valid_labels: np.ndarray, radius: float) -> np.ndarray:
    """
    제거된 포인트(-1 라벨)를 남은 클러스터 중 가장 가까운 곳에 재할당.
    단, 모든 클러스터와 거리 >= R 이면 그대로 -1 유지.
    """
    if centroids.shape[0] == 0:
        return valid_labels
    removed_mask = valid_labels == -1
    if not np.any(removed_mask):
        return valid_labels
    Xr = X[removed_mask]
    d = np.sqrt(((Xr[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))  # (Nr, K')
    nearest = np.argmin(d, axis=1)
    nearest_dist = d[np.arange(d.shape[0]), nearest]
    accept = nearest_dist < radius
    # 적용
    new_labels = valid_labels.copy()
    target_idx = np.where(removed_mask)[0]
    new_labels[target_idx[accept]] = nearest[accept]
    return new_labels


def compress_labels(labels: np.ndarray, keep_clusters: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    남은 클러스터 인덱스를 0..M-1로 압축. 반환: new_labels(N,), old_to_new(dict)
    """
    old_to_new = {}
    new_id = 0
    for j, keep in enumerate(keep_clusters):
        if keep:
            old_to_new[j] = new_id
            new_id += 1
    new_labels = labels.copy()
    for old, new in old_to_new.items():
        new_labels[labels == old] = new
    # 제거된(-1)은 그대로 유지
    return new_labels, old_to_new


def recompute_centroids(X: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    현재 라벨 기준으로 중심 재계산.
    """
    if num_clusters == 0:
        return np.zeros((0, 2))
    C = np.zeros((num_clusters, 2), dtype=float)
    for j in range(num_clusters):
        pts = X[labels == j]
        if len(pts) > 0:
            C[j] = pts.mean(axis=0)
    return C


def build_edges(centroids: np.ndarray, max_dist: float) -> List[Tuple[int, int, float]]:
    """
    중심 간 거리 < max_dist 인 쌍을 양방향 엣지로 생성. 가중치 w = max(0, 1 - d/max_dist)
    """
    m = centroids.shape[0]
    edges: List[Tuple[int, int, float]] = []
    if m == 0:
        return edges
    for i in tqdm(range(m), desc="Build edges", unit="node"):
        for j in range(i + 1, m):
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d < max_dist:
                w = max(0.0, 1.0 - d / max_dist)
                edges.append((i, j, w))
                edges.append((j, i, w))
    return edges


def build_edges_mahalanobis(node_arrays: List[np.ndarray],
                            C_arr: List[Tuple[Tuple[float, float], np.ndarray]],
                            trust_range: float,
                            conn_threshold: float) -> List[Tuple[int, int, float]]:
    """
    Mahalanobis 거리 기반으로 directed edges 생성.
    - 각 노드 i에 대해 (mean, cov)이 C_arr에 있음.
    - 노드 A(i)와 노드 B(j)에 대해 B의 데이터 중 A의 분포에서 Mahalanobis 거리 <= trust_range 인 점의 비율이
      conn_threshold 이상이면 A -> B 엣지를 생성.
    - 엣지 가중치(weight)는 (조건을 만족하는) 점들의 Mahalanobis 거리 평균으로 설정.

    반환: List of (i, j, weight)
    """
    m = len(C_arr)
    edges: List[Tuple[int, int, float]] = []
    if m == 0:
        return edges

    eps = 1e-6
    for i in range(m):
        mu_i, cov_i = C_arr[i]
        mu_i = np.asarray(mu_i, dtype=float)
        cov_i = np.asarray(cov_i, dtype=float)
        # 정규화: 빈 또는 영 공분산이면 역행렬 계산 불가 -> 작은 정규화 항 추가
        try:
            cov_inv = np.linalg.pinv(cov_i + eps * np.eye(2))
        except Exception:
            cov_inv = np.linalg.pinv(np.eye(2) * eps)

        for j in range(m):
            if i == j:
                continue
            pts_B = node_arrays[j]
            if pts_B is None or pts_B.size == 0:
                continue
            # Mahalanobis 거리 계산: sqrt((x-mu)^T cov_inv (x-mu))
            diffs = pts_B - mu_i
            # compute squared Mahalanobis distances
            # use einsum for speed
            d2 = np.einsum('...i,ij,...j', diffs, cov_inv, diffs)
            # numerical safety
            d2 = np.maximum(d2, 0.0)
            dists = np.sqrt(d2)
            # 비율 계산
            in_mask = dists <= trust_range
            count_in = int(np.count_nonzero(in_mask))
            total_b = pts_B.shape[0]
            proportion = count_in / total_b
            if proportion >= conn_threshold and count_in > 0:
                weight = float(np.mean(dists[in_mask]))
                edges.append((i, j, weight))
    return edges


def build_demands(df: pd.DataFrame, labels: np.ndarray, num_nodes: int) -> Tuple[pd.Timestamp, pd.Timestamp, List[List[int]]]:
    """
    각 시간(hour) x 노드별 수요 행렬 생성. 라벨 -1 은 제외.
    """
    # 시간 경계
    minHour = df['call_date'].min()
    maxHour = df['call_date'].max()
    total_hours = int((maxHour - minHour).total_seconds() // 3600) + 1
    demands = np.zeros((total_hours, num_nodes), dtype=int)

    # 시간 인덱스 계산 벡터화
    t_idx = ((df['call_date'] - minHour).dt.total_seconds() // 3600).astype(int).to_numpy()
    valid_mask = (labels != -1) & (t_idx >= 0) & (t_idx < total_hours)
    rows = t_idx[valid_mask]
    cols = labels[valid_mask]
    # 다중 인덱싱 집계
    np.add.at(demands, (rows, cols), 1)

    return minHour, maxHour, demands.tolist()


def plot_kmeans_partitions(
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        counts: np.ndarray,
        save_dir: Path,
        edges: List[Tuple[int, int, float]],
        sample_cap: int = 200_000) -> None:
    """
    간단한 K-means 파티션 플롯:
    - 모든 포인트는 클러스터 색으로 산점도로 그리되 투명도 낮게 표시
    - 노드 번호(label)는 edges에 연결된 노드만 표시
    - edges에 있는 연결만 선으로 그림 (directed 표시 안함, 단순 선)
    - 막대그래프는 그대로 유지
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 유효 포인트만
    mask = labels != -1
    Xv = X[mask]
    Lv = labels[mask]

    # 샘플링
    n = Xv.shape[0]
    if n > sample_cap:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, size=sample_cap, replace=False)
        Xp = Xv[idx]
        Lp = Lv[idx]
    else:
        Xp, Lp = Xv, Lv

    num_nodes = centroids.shape[0]

    # 연결된 노드 집합 추출
    connected = set()
    for (a, b, _) in edges:
        connected.add(int(a))
        connected.add(int(b))

    # If no connected nodes found, fall back to showing all nodes (helpful for debugging)
    show_all_nodes_if_none = False
    if len(connected) == 0:
        show_all_nodes_if_none = True
        # do not modify `connected`, just use flag to show all labels in that case

    # 산점도 + 노드 번호 및 엣지 그리기
    if num_nodes > 0 and Xp.shape[0] > 0:
        plt.figure(figsize=(10, 8), facecolor='white')
        cmap = plt.get_cmap('tab20')
        # 모든 클러스터 포인트를 희미하게 그림
        for c in range(num_nodes):
            m = Lp == c
            if np.any(m):
                plt.scatter(Xp[m, 0], Xp[m, 1], s=3, alpha=0.25, color=cmap(c % 20), linewidths=0)

        # 엣지 그리기 (centroid 좌표 사용)
        for (i, j, w) in edges:
            if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes:
                continue
            p1 = centroids[int(i)]
            p2 = centroids[int(j)]
            # 선: 회색, 두께는 가중치에 따라 약간 조절
            try:
                lw = max(0.5, 2.0 - float(w) / (max(1.0, float(w)) + 1e-9))
            except Exception:
                lw = 1.0
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', alpha=0.7, linewidth=lw, zorder=1)

        # 모든 centroid는 작은 마커로 먼저 표시(항상 보이도록)
        if num_nodes > 0:
            centroids_arr = np.asarray(centroids)
            plt.scatter(centroids_arr[:, 0], centroids_arr[:, 1], s=20, c='none', edgecolors='black', linewidths=0.8, zorder=2)

        # 연결된 노드만 번호 표시 (없을 경우 모든 노드 표시)
        for c in range(num_nodes):
            cx, cy = centroids[c]
            if show_all_nodes_if_none or (c in connected):
                # 강조해서 표시
                plt.scatter([cx], [cy], c='black', s=80, zorder=3)
                plt.text(cx, cy, str(c), fontsize=10, ha='center', va='center', color='white', fontweight='bold',
                         bbox=dict(facecolor='black', alpha=0.85, edgecolor='none', pad=1), zorder=4)
            else:
                # 비연결 노드는 작은 회색 점으로 표시(라벨 없음)
                plt.scatter([cx], [cy], c='gray', s=10, zorder=2)

        # If we showed all nodes because none were connected, add a small note
        if show_all_nodes_if_none:
            plt.text(0.01, 0.01, 'Note: no edges found => showing all node ids', transform=plt.gca().transAxes,
                     fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        plt.title('K-means Partitions (connected nodes)')
        plt.xlabel('xpos')
        plt.ylabel('ypos')
        plt.tight_layout()
        plt.savefig(save_dir / f'kmeans_partitions_connected.png', dpi=600)
        plt.close()

    # 막대그래프(클러스터별 데이터 수)
    if counts.size > 0:
        order = np.argsort(-counts)
        plt.figure(figsize=(12, 5), facecolor='white')
        plt.bar(np.arange(len(order)), counts[order], color='steelblue')
        plt.xlabel('Cluster (sorted by size)')
        plt.ylabel('#Points')
        plt.title('Points per Cluster')
        plt.tight_layout()
        plt.savefig(save_dir / 'cluster_counts.png', dpi=600)
        plt.close()

def plot_demnands_histogram(demand_T: np.ndarray, save_dir: Path):
    # 값별 개수 집계
    values, counts = np.unique(demand_T, return_counts=True)
    plt.figure(figsize=(12, 6), facecolor='white')
    plt.bar(values, counts, color='orange')
    plt.yscale('log')
    plt.xlabel('Demand Value')
    plt.ylabel('Count (log scale)')
    plt.title('Demand Value Histogram (log scale)')
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'demand_histogram.png', dpi=600)
    plt.close()

def node_arrays_and_covariances(X: np.ndarray, compressed_labels: np.ndarray
                                ) -> Tuple[List[np.ndarray], List[Tuple[Tuple[float, float], np.ndarray]]]:
    """
    반환:
      - node_arrays: 각 노드 j에 대해 할당된 점들의 np.ndarray (N_j, 2)
      - C_arr: 각 노드에 대해 ((xmean, ymean), cov_matrix(2x2)) 리스트
    빈 노드는 빈 배열과 2x2 영행렬을 반환.
    """
    # 유효 라벨 존재 확인
    if compressed_labels.size == 0 or not np.any(compressed_labels != -1):
        return [], []

    num_nodes = int(compressed_labels.max()) + 1
    node_arrays: List[np.ndarray] = []
    C_arr: List[Tuple[Tuple[float, float], np.ndarray]] = []

    for j in range(num_nodes):
        pts = X[compressed_labels == j]
        node_arrays.append(pts)

        if pts.shape[0] == 0:
            mean = (0.0, 0.0)
            cov = np.zeros((2, 2), dtype=float)
        else:
            mean_vals = pts.mean(axis=0)
            mean = (float(mean_vals[0]), float(mean_vals[1]))
            if pts.shape[0] >= 2:
                cov = np.cov(pts, rowvar=False)
                # np.cov의 결과가 (2,2)인지 보장
                cov = cov.reshape(2, 2)
            else:
                cov = np.zeros((2, 2), dtype=float)

        C_arr.append((mean, cov))

    return node_arrays, C_arr

if __name__ == "__main__":
    # 스크립트 기준 경로 고정
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'

    # 1) 데이터 로드
    csv_path = data_dir / 'essential_columns.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    # 필수 컬럼 확인 (xpos, ypos, call_date)
    for col in ['xpos', 'ypos', 'call_date']:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼 누락: {col}")
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H')

    # 2) 좌표 행렬
    X = df[['xpos', 'ypos']].to_numpy(dtype=float)
    n = X.shape[0]
    if n == 0:
        # 빈 입력 방어
        outjson = OutJson(
            minHour=pd.Timestamp.now(),
            maxHour=pd.Timestamp.now(),
            total_nodes=0,
            edges=[],
            demands=[],
        )
        outjson.save_json(data_dir / 'gwn_data.json')
        raise SystemExit(0)

    # 3) K-means 학습
    centroids, labels = kmeans_fit(X, K)

    # 4) 반경 기준 제거
    kept_mask, _ = filter_by_radius(X, centroids, labels, R)

    # 5) 작은 클러스터 제거 (반경 필터 후 남은 포인트 기준)
    cluster_keep, labels_after_small = drop_small_clusters(labels, kept_mask, POW)

    # 6) 제거된 포인트를 남은 클러스터에 재할당 (반경 내만)
    # 남은 클러스터 중심만 사용
    kept_indices = np.where(cluster_keep)[0]
    if kept_indices.size > 0:
        kept_centroids = centroids[kept_indices]
    else:
        kept_centroids = np.zeros((0, 2))
    reassigned = reassign_removed(X, kept_centroids, labels_after_small, R)

    # 재할당된 포인트에만 kept_indices 매핑 적용
    final_labels = reassigned.copy()
    if kept_indices.size > 0:
        newly_assigned_mask = (labels_after_small == -1) & (final_labels != -1)
        final_labels[newly_assigned_mask] = kept_indices[final_labels[newly_assigned_mask]]

    # 7) 실제로 포인트가 배정된 클러스터만 keep 대상으로 선정 후 라벨 압축 + 중심 재계산
    k_total = centroids.shape[0]
    final_keep = np.zeros(k_total, dtype=bool)
    used_ids = np.unique(final_labels[final_labels != -1])
    final_keep[used_ids] = True
    compressed_labels, old2new = compress_labels(final_labels, final_keep)
    num_nodes = len(old2new)

    # 중심 재계산 (재할당 반영)
    final_centroids = recompute_centroids(X, compressed_labels, num_nodes)

    node_arrays, C_arr = node_arrays_and_covariances(X, compressed_labels)

    # 8) 엣지 구성 (거리 < MAX_NODE_DIST)
    # Mahalanobis 기반 엣지 생성: node_arrays, C_arr 사용
    edges = build_edges_mahalanobis(node_arrays, C_arr, TRUST_RANGE, CONNECTION_THRESHOLD)

    # Debug/log: show edges summary to help explain missing labels/lines in plot
    print(f"DEBUG: edges count = {len(edges)}")
    if len(edges) > 0:
        print("DEBUG: sample edges (up to 20):")
        for e in edges[:20]:
            print(" ", e)
        connected_nodes = set()
        for a, b, _ in edges:
            connected_nodes.add(int(a)); connected_nodes.add(int(b))
        print(f"DEBUG: connected node count = {len(connected_nodes)} -> {sorted(list(connected_nodes))[:50]}")
    else:
        print(f"DEBUG: No edges found. TRUST_RANGE={TRUST_RANGE}, CONNECTION_THRESHOLD={CONNECTION_THRESHOLD}")

    # 9) 수요 행렬 구성
    minHour, maxHour, demands = build_demands(df, compressed_labels, num_nodes)
    sum_demands = sum(sum(row) for row in demands)
    print(f'Sum of demands: {sum_demands}')
    print(f'dropped points: {n - sum_demands} / {n} ({(n - sum_demands) / n:.2%})')

    # 10) OutJson 저장
    outjson = OutJson(
        minHour=minHour,
        maxHour=maxHour,
        total_nodes=num_nodes,
        edges=edges,
        demands=demands,
    )
    outjson.save_json(data_dir / 'gwn_data.json')

    # # 11) 시각화 저장
    plot_kmeans_partitions(
        X,
        compressed_labels,
        final_centroids,
        np.array([np.sum(compressed_labels == j) for j in range(num_nodes)]),
        img_dir,
        edges,
    )


    # 간단 로그
    print(f"Total rows: {n}")
    print(f"Initial K: {K} -> final nodes: {num_nodes}")
    print(f"Edges: {len(edges)}")
    print("Saved to", (data_dir / 'gwn_data.json').as_posix())