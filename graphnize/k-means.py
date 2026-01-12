from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm  # 진행률 표시
import matplotlib
from scipy.spatial import ConvexHull
from scipy.signal import find_peaks  # 추가됨

from graphnize.makeset import make_clusters

matplotlib.use("Agg")  # 서버/헤드리스 환경에서도 저장 가능
import matplotlib.pyplot as plt
# from tslearn.metrics import cdist_dtw
from numba import njit, prange
from outjson import OutJson

# 하이퍼파라미터
K = 200       # 초기 클러스터 개수
R = 2500          # 클러스터 중심으로부터 허용 반경 (이상은 제거)
POW = 183*10       # 클러스터 최소 데이터 수 (이하 클러스터 제거)
MAX_NODE_DIST = 8000  # 노드 간 연결 최대 거리(이상은 미연결)
MAX_ITER = 100    # K-means 반복 횟수
SEED = 42         # 재현성
NUM_OF_DAYS = 182

CONNECTION_THRESHOLD = 1.8
CLUSTER_CONNECTION_THRESHOLD = 2.5


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


def kmeans_fit(
        X: np.ndarray, k: int, max_iter: int = MAX_ITER,
        seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """
    간단한 K-means 학습
    - 입력: X (N,2), k
    - 출력: (centroids(K,2), labels(N,))
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k >= n:
        raise Exception("k는 n보다 커야함")

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

# 8) centroid 시각화 함수
def visualize_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, save_path: Path, n: int = 9) -> float:
    """클러스터별 면적 합계가 전체 면적에서 차지하는 비율을 반환합니다."""
    plt.figure(figsize=(10, 10))

    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    # 1. 전체 면적 계산 (격자 설정 범위 기준)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    total_area = (x_max - x_min) * (y_max - y_min)

    # 2. 클러스터별 면적 계산 (방법 B)
    total_cluster_area = 0.0

    # 클러스터별 데이터 산점도 및 면적 합산
    for i, lbl in enumerate(unique_labels):
        cluster_mask = (labels == lbl)
        cluster_points = X[cluster_mask]

        # 산점도 그리기
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(i), alpha=0.6)

        # 면적 계산 (label -1은 노이즈로 간주하여 제외하거나, 포함하려면 조건 수정)
        if lbl != -1 and len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                total_cluster_area += hull.volume  # 2D에서 hull.volume은 면적을 의미함
            except:
                # 점들이 일직선상에 있는 등 ConvexHull을 만들 수 없는 경우 예외 처리
                pass

    # 3. 중심점 표시
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, color='black', marker='x', label='Centroids')

    # 4. 격자 설정 및 그래프 꾸미기
    x_edges = np.linspace(x_min, x_max, n + 1)
    y_edges = np.linspace(y_min, y_max, n + 1)
    plt.xticks(x_edges)
    plt.yticks(y_edges)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 퍼센트 계산
    area_percentage = (total_cluster_area / total_area) * 100 if total_area > 0 else 0.0

    plt.title(f'K-means Clustering (Occupied Area: {area_percentage:.2f}%)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return float(area_percentage)

#각 centroid의 point개수 시각화 함수(막대 그래프)
def visualize_allocation(labels: np.ndarray, num_nodes: int, save_path: Path) -> None:
    plt.figure(figsize=(30, 10))

    allocation_counts = [np.sum(labels == i) for i in range(num_nodes)]
    #정렬
    allocation_counts.sort(reverse=True)

    bars = plt.bar(range(num_nodes), allocation_counts, color='skyblue')
    plt.bar_label(bars, padding=3)
    plt.xlabel('Centroid Index')
    plt.ylabel('Number of Points Allocated')
    plt.title('Point Allocation per Centroid')
    plt.grid(axis='y')
    plt.savefig(save_path)
    plt.close()


@njit(fastmath=True)
def dtw_distance(s1, s2):
    """두 시계열 사이의 DTW 거리를 계산 (L2 norm 기반)"""
    n, m = len(s1), len(s2)
    # DP 테이블 초기화 (메모리 절약을 위해 현재와 이전 열만 사용할 수도 있지만, 이해를 위해 전체 테이블 사용)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s1[i - 1] - s2[j - 1]) ** 2
            # 대각선, 위, 왼쪽 중 최소값 선택
            last_min = min(dtw_matrix[i - 1, j],  # insertion
                           dtw_matrix[i, j - 1],  # deletion
                           dtw_matrix[i - 1, j - 1])  # match
            dtw_matrix[i, j] = cost + last_min

    return np.sqrt(dtw_matrix[n, m])


@njit(parallel=True)
def compute_dtw_matrix(X):
    """
    (nodes, timesteps) 형태의 2차원 배열을 받아
    (nodes, nodes) 형태의 거리 행렬을 반환
    """
    n_nodes = X.shape[0]
    dist_matrix = np.zeros((n_nodes, n_nodes))

    # parallel=True와 prange를 사용하여 멀티코어 병렬 처리
    for i in prange(n_nodes):
        for j in range(i + 1, n_nodes):
            d = dtw_distance(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  # 대칭 행렬이므로 복사

    return dist_matrix

def visualize_dtw(dtw_matrix: np.ndarray, save_path: Path) -> None:
    """DTW 유사도 행렬 시각화 및 저장."""
    plt.figure(figsize=(10, 8))
    plt.imshow(dtw_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='DTW Distance')
    plt.title('DTW Similarity Matrix')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.savefig(save_path)
    plt.close()

def five_summation(arr):
    sorted_indices = np.argsort(arr)
    n = len(arr)

    # 2. 5점 요약에 해당하는 순수 인덱스 계산
    # (0%, 25%, 50%, 75%, 100% 위치의 인덱스 추출)
    idx_min = sorted_indices[0]
    idx_q1 = sorted_indices[int(np.percentile(np.arange(n), 25))]
    idx_q2 = sorted_indices[int(np.percentile(np.arange(n), 50))]
    idx_q3 = sorted_indices[int(np.percentile(np.arange(n), 75))]
    idx_max = sorted_indices[-1]

    # 결과 출력
    summary_labels = ["최솟값(Min)", "제1사분위(Q1)", "중앙값(Q2)", "제3사분위(Q3)", "최댓값(Max)"]
    indices = [idx_min, idx_q1, idx_q2, idx_q3, idx_max]

    print(f"{'구분':<10} | {'인덱스':<6} | {'값':<5}")
    print("-" * 25)
    for label, idx in zip(summary_labels, indices):
        print(f"{label:<10} | {idx:<8} | {arr[idx]:<5}")

def demand_fft(demands_array: np.ndarray, img_dir: Path):
    demands_array = demands_array[:, :1]
    T, B = demands_array.shape
    time_step_hours = 1.0

    for b in range(B):
        x = demands_array[:, b]
        x = x - np.mean(x)

        fft_vals = np.fft.rfft(x)
        magnitude = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(T, d=time_step_hours)
        magnitude[0] = 0.0

        # --- 수정된 부분: 인접한 값이 아닌 진짜 '봉우리'를 찾음 ---
        # distance: 피크 사이의 최소 간격 (여기서는 인덱스 거리)
        # 24시간 근처에서 여러 개가 잡히지 않도록 적절한 간격을 줍니다.
        peaks, properties = find_peaks(magnitude, distance=5) 
        
        # 찾은 피크들 중 magnitude가 큰 순서대로 상위 5개 추출
        peak_magnitudes = magnitude[peaks]
        top_n_indices = peaks[np.argsort(peak_magnitudes)[-5:][::-1]]
        # ---------------------------------------------------

        plt.figure(figsize=(12, 6))
        plt.plot(freqs, magnitude, label="FFT Magnitude", color='royalblue', alpha=0.7)

        for i, idx in enumerate(top_n_indices):
            f = freqs[idx]
            mag = magnitude[idx]
            period = 1/f if f != 0 else float('inf')
            
            plt.axvline(f, color='red', linestyle="--", alpha=0.3)
            plt.text(f, mag, f'({period:.1f}h)', 
                     color='red', fontsize=9, verticalalignment='bottom', horizontalalignment='center')

        plt.xlabel("Frequency (1/hour)")
        plt.ylabel("Magnitude")
        plt.xlim(0, 0.1)
        plt.title(f"Demand FFT - Distinct Top Peaks")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = img_dir / f"demand_fft_top_peaks_{b}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def assign_matrix(num_nodes: int, clusters: List[List[int]]) -> np.ndarray:
    assign = np.zeros((num_nodes, len(clusters)), dtype=int)
    for c_idx, cluster in enumerate(clusters):
        for node in cluster:
            assign[node, c_idx] = 1
    return assign


if __name__ == "__main__":
    # 스크립트 기준 경로 고정
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'

    # 1) 데이터 로드
    csv_path = data_dir / 'extraction.csv'
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
        raise Exception("n == 0")

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
        raise Exception("유효한 cluster가 없음")
    reassigned = reassign_removed(X, kept_centroids, labels_after_small, R)

    print(reassigned.shape)

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

    print(f'Final number of nodes: {num_nodes}')
    print(f'Dropped points: {np.sum(compressed_labels == -1)} / {n} ({np.sum(compressed_labels == -1) / n:.2%})')

    #centroid의 지도상 위치 표시
    coverage = visualize_clusters(X, compressed_labels, final_centroids, img_dir / 'kmeans_result.png')
    #각 centroid의 point를 그래프로 표시
    visualize_allocation(compressed_labels, num_nodes, img_dir / 'kmeans_allocation.png')

    # # 9) 수요 행렬 구성
    minHour, maxHour, demands = build_demands(df, compressed_labels, num_nodes)
    print(f'Demand time range: {minHour} to {maxHour}')
    print(f'demands shape: {len(demands)} hours x {num_nodes} nodes')

    demands_array = np.array(demands) #4368 * num_nodes
    demand_fft(demands_array, img_dir)

    mean_demand = np.mean(demands_array.reshape(-1, 7 * 24,num_nodes), axis=0)
    print(f'Mean weekly demand shape: {mean_demand.shape}') # (7*24=168, num_nodes)
    sum_demand = np.sum(mean_demand, axis=0)
    five_summation(sum_demand)
    np.save(data_dir / 'demands_array.npy', demands_array)


    #파일이 있으면 불러오기
    sim_matrix = compute_dtw_matrix(mean_demand.T)
    #dtw 행렬 시각화
    visualize_dtw(sim_matrix, img_dir / 'dtw_similarity.png')
    np.save(data_dir / 'dtw_similarity.npy', sim_matrix)

    edges = []
    n_nodes = sim_matrix.shape[0]
    for u_idx in range(n_nodes):
        for v_idx in range(u_idx + 1, n_nodes):  # 자기 자신 제외, 상삼각 행렬만 순회
            sim = sim_matrix[u_idx, v_idx]
            if sim <= CONNECTION_THRESHOLD:
                edges.append([u_idx, v_idx, float(sim)])
                edges.append([v_idx, u_idx, float(sim)])
    print(f'Number of edges: {len(edges)}')

    clusters = list(make_clusters())
    print(len(clusters))
    num_clusters = len(clusters)
    assigns = assign_matrix(num_nodes, clusters)
    np.save(data_dir / 'assign_matrix.npy', assigns)
    cluster_demands = np.matmul(demands_array, assigns)
    np.save(data_dir / 'cluster_demands.npy', cluster_demands)

    #dtw
    cluster_mean_demand = np.mean(cluster_demands.reshape(-1, 7 * 24, num_clusters), axis=0)
    cluster_sim_matrix = compute_dtw_matrix(cluster_mean_demand.T)
    visualize_dtw(cluster_sim_matrix, img_dir / 'cluster_dtw_similarity.png')
    np.save(data_dir / 'cluster_dtw_similarity.npy', cluster_sim_matrix)

    cluster_edges = []
    for u_idx in range(num_clusters):
        for v_idx in range(u_idx + 1, num_clusters):  # 자기 자신 제외, 상삼각 행렬만 순회
            sim = cluster_sim_matrix[u_idx, v_idx]
            if sim <= CLUSTER_CONNECTION_THRESHOLD:
                cluster_edges.append([u_idx, v_idx, float(sim)])
                cluster_edges.append([v_idx, u_idx, float(sim)])
    print(f'Number of cluster edges: {len(cluster_edges)}')

    # 10) OutJson 저장
    outjson = OutJson(
        minHour=minHour,
        maxHour=maxHour,
        total_nodes=num_nodes,
        edges=edges,
        demands=demands,
        coverage=coverage,
        assignment_adj=assigns.tolist(),
        clusters_demand=cluster_demands.tolist(),
        clusters_edge=cluster_edges,
    )
    outjson.save_json(data_dir / 'gwn_data.json')