import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ------------------------
# 설정값
# ------------------------
N = 200           # 포인터(ptr) 개수
K = 50            # 각 이동 단계에서 선택할 최근접 K개
r = 1000.0        # 최종 할당 반경 r
SEED = 42         # 재현을 위한 난수 시드
EPSILON = 1e-6    # 수렴 허용오차(포인터 최대 이동량)
MAX_ITERS = 100   # 외부 반복의 최대 횟수


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """2차원 점 a와 b(형상 (2,)) 사이의 유클리드 거리를 반환합니다."""
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def select_k_nearest(points: np.ndarray, center: np.ndarray, k: int) -> np.ndarray:
    """`points` 중에서 중심점 `center`에 가장 가까운 k개의 인덱스를 반환합니다.

    - points: 형상 (M, 2)
    - center: 형상 (2,)
    - k: int
    """
    if points.size == 0:
        return np.array([], dtype=int)
    k = max(0, min(k, len(points)))
    if k == 0:
        return np.array([], dtype=int)
    # 안전한 계산을 위해 float로 변환하여 제곱 거리 계산
    diff = points.astype(float) - center.astype(float)
    d2 = np.sum(diff * diff, axis=1)
    # 최근접 k개를 argpartition으로 선택
    idx = np.argpartition(d2, kth=k - 1)[:k]
    # 선택된 인덱스를 실제 거리 순으로 정렬
    idx = idx[np.argsort(d2[idx])]
    return idx


def move_ptrs_until_converged(ptrs: np.ndarray, data: np.ndarray, k: int, eps: float, max_iters: int) -> np.ndarray:
    """외부 반복마다 데이터 사본으로부터 최근접 K개를 사용하여 포인터들을 이동시키고, 수렴할 때까지 반복합니다.

    각 외부 반복에서 수행:
      - new_data = data의 복사본
      - 각 포인터에 대해 순서대로:
          * new_data에서 최근접 K개 선택
          * 해당 포인트들의 (가중=1) 평균 위치로 이동(정수 반올림)
          * 선택된 K개는 new_data에서 제거
    포인터들의 최대 이동량이 eps보다 작아지거나 max_iters에 도달하면 중단합니다.
    """
    ptrs = ptrs.copy()

    outer_bar = tqdm(range(max_iters), desc="Outer iters", unit="iter")
    for _ in outer_bar:
        old_ptrs = ptrs.copy()
        new_data = data.copy()

        for i in tqdm(range(len(ptrs)), desc="Move ptrs", unit="ptr", leave=False):
            if len(new_data) == 0:
                break
            k_i = min(k, len(new_data))
            idx = select_k_nearest(new_data, ptrs[i], k_i)
            if len(idx) == 0:
                continue
            chosen = new_data[idx]
            # 가중치가 모두 1인 평균을 계산하고 정수로 반올림하여 좌표를 이동
            mean_pos = np.rint(chosen.astype(float).mean(axis=0)).astype(int)
            ptrs[i] = mean_pos
            # 방금 사용한 K개는 new_data에서 제거
            keep_mask = np.ones(len(new_data), dtype=bool)
            keep_mask[idx] = False
            new_data = new_data[keep_mask]

        # 이번 반복에서의 포인터 최대 이동량을 계산하여 수렴 여부 확인
        max_shift = np.max(np.linalg.norm(ptrs.astype(float) - old_ptrs.astype(float), axis=1)) if len(ptrs) else 0.0
        outer_bar.set_postfix(max_shift=f"{max_shift:.3g}")
        if max_shift < eps:
            break

    outer_bar.close()
    return ptrs


def assign_points_to_ptrs(ptrs: np.ndarray, data: np.ndarray, radius: float):
    """각 데이터 포인트를 가장 가까운 포인터에 반경 `radius` 이내인 경우에만 할당합니다.

    반환값:
      - assignments: 각 포인터에 할당된 데이터 인덱스 리스트의 리스트
      - distances_per_ptr: 포인터별로 각 할당 데이터까지의 거리 배열 리스트
    """
    M = len(data)
    P = len(ptrs)
    assignments = [[] for _ in range(P)]
    distances_per_ptr = [list() for _ in range(P)]

    if P == 0 or M == 0:
        return assignments, [np.array(d) for d in distances_per_ptr]

    for j in tqdm(range(M), desc="Assign points", unit="pt"):
        p = data[j]
        # 정의한 distance() 함수를 사용하여 모든 포인터와의 거리를 계산
        dists = np.array([distance(ptrs[i], p) for i in range(P)], dtype=float)
        i_min = int(np.argmin(dists))
        d_min = float(dists[i_min])
        if d_min <= radius:
            assignments[i_min].append(j)
            distances_per_ptr[i_min].append(d_min)

    distances_per_ptr = [np.array(d, dtype=float) if len(d) else np.array([], dtype=float) for d in distances_per_ptr]
    return assignments, distances_per_ptr


def five_number_summary(values: np.ndarray):
    """1차원 배열에 대한 다섯 수 요약을 (최솟값, 1사분위수, 중앙값, 3사분위수, 최댓값) 순서로 반환합니다.
    값이 비어 있으면 NaN을 반환합니다.
    """
    if values.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    quantiles = np.percentile(values, [0, 25, 50, 75, 100])
    return tuple(float(x) for x in quantiles)


if __name__ == "__main__":
    # 데이터 로드 (xpos, ypos만 사용)
    csv_path = Path('../data') / 'essential_columns.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    if not {'xpos', 'ypos'}.issubset(df.columns):
        raise ValueError("CSV에는 'xpos'와 'ypos' 컬럼이 포함되어 있어야 합니다.")

    # 정수 좌표로 numpy 배열 구성
    points = df[['xpos', 'ypos']].to_numpy(dtype=np.int64)

    # 데이터의 경계값
    maxX = int(df['xpos'].max())
    maxY = int(df['ypos'].max())
    minX = int(df['xpos'].min())
    minY = int(df['ypos'].min())
    print(f"Max X: {maxX}, Max Y: {maxY}")

    # 경계 내에서 정수 좌표로 포인터 초기화
    rng = np.random.default_rng(SEED)
    xs = rng.integers(minX, maxX + 1, size=N, endpoint=False)
    ys = rng.integers(minY, maxY + 1, size=N, endpoint=False)
    ptrs = np.column_stack([xs, ys]).astype(np.int64)

    # 수렴할 때까지 포인터 이동 반복
    ptrs = move_ptrs_until_converged(ptrs, points, K, EPSILON, MAX_ITERS)

    # 반경 r 이내로 최종 할당 수행
    assignments, distances_per_ptr = assign_points_to_ptrs(ptrs, points, r)

    # 포인터별 할당된 데이터 개수
    counts = np.array([len(a) for a in assignments], dtype=int)

    # 포인터별 데이터 개수에 대한 다섯 수 요약
    five_num = five_number_summary(counts.astype(float))

    # 포인터별 거리 평균/표준편차 계산
    mean_std_per_ptr = []
    for dists in distances_per_ptr:
        if dists.size == 0:
            mean_std_per_ptr.append((np.nan, np.nan))
        else:
            mean_std_per_ptr.append((float(np.mean(dists)), float(np.std(dists, ddof=0))))

    # 결과 출력(요약)
    print("\n=== Results ===")
    print(f"Total ptrs: {len(ptrs)}")
    print(f"Assignment radius r: {r}")
    print("Assigned counts per ptr (first 20):", counts[:20].tolist())
    print("Five-number summary of assigned counts across ptrs:")
    print(f"min={five_num[0]:.3f}, Q1={five_num[1]:.3f}, median={five_num[2]:.3f}, Q3={five_num[3]:.3f}, max={five_num[4]:.3f}")

    # 포인터별 거리 통계 미리보기(최대 20개)
    preview = [
        {
            'ptr_index': i,
            'mean_dist': (None if np.isnan(ms[0]) else round(ms[0], 6)),
            'std_dist': (None if np.isnan(ms[1]) else round(ms[1], 6)),
            'count': int(counts[i])
        }
        for i, ms in enumerate(mean_std_per_ptr[:20])
    ]
    print("Mean/Std of distances per ptr (preview up to 20):")
    print(preview)

    # 포인터 좌표 및 할당 개수 저장(선택)
    out_df = pd.DataFrame({
        'ptr_x': ptrs[:, 0],
        'ptr_y': ptrs[:, 1],
        'assigned_count': counts
    })
    out_path = Path('../data') / 'ptrs_summary.csv'
    try:
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"Saved ptrs summary to {out_path.resolve()}")
    except Exception as e:
        print(f"Warning: failed to save ptrs summary: {e}")

