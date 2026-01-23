from numba import njit, prange
import numpy as np

@njit(fastmath=True)
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
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

@njit(parallel=True, fastmath=True)
def compute_dtw_matrix(X): #N, T -> N, N
    n_nodes = X.shape[0]
    dist_matrix = np.zeros((n_nodes, n_nodes))

    # parallel=True와 prange를 사용하여 멀티코어 병렬 처리
    for i in prange(n_nodes):
        for j in range(i + 1, n_nodes):
            d = dtw_distance(X[i], X[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d  # 대칭 행렬이므로 복사

    return dist_matrix