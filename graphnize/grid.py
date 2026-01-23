from pathlib import Path
import pandas as pd
import numpy as np
import pulp
from scipy import cluster

from DMVST.gridnize import GRID_SIZE
from graphnize.utils.clustering import make_clusters
from graphnize.utils.compute import compute_dtw_matrix
from graphnize.utils.dataframe import *
from graphnize.utils.grid_utils import *
from graphnize.utils.statics import five_num
from graphnize.utils.visualize import visualize_dtw
from outjson import OutJson
from visualize.grid2hitmap import gridhitmap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GRID_SIZE = 9500
CONNECTION_THRESHOLD = 4
CLUSTER_CONNECTION_THRESHOLD = 15

N = 25
A = 7000 // GRID_SIZE
B = A

def get_patch(grid: np.ndarray, a: int, b: int, x: int):
    N, M = grid.shape
    
    # 1. 전처리: 2D 누적 합을 이용해 모든 가능한 패치 점수 계산
    S = np.zeros((N + 1, M + 1), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            S[i+1, j+1] = grid[i, j] + S[i, j+1] + S[i+1, j] - S[i, j]
            
    def get_sum(r, c):
        return int(S[r+a, c+b] - S[r, c+b] - S[r+a, c] + S[r, c])

    # 2. 최적화 문제 정의
    prob = pulp.LpProblem("Patch_Optimization", pulp.LpMaximize)
    
    # 3. 변수 생성: 각 위치 (r, c)에 패치를 놓을지 여부 (이진 변수)
    choices = pulp.LpVariable.dicts("Choice", (range(N-a+1), range(M-b+1)), cat='Binary')
    
    # 4. 목적 함수: 전체 점수 합계 최대화
    prob += pulp.lpSum([choices[r][c] * get_sum(r, c) 
                        for r in range(N-a+1) for c in range(M-b+1)])
    
    # 5. 제약 조건 1: 패치 개수 x개 선택
    prob += pulp.lpSum([choices[r][c] for r in range(N-a+1) for c in range(M-b+1)]) == x
    
    # 6. 제약 조건 2: 겹침 방지 (중요!)
    # 각 격자 칸 (i, j)는 최대 하나의 패치에 의해서만 덮여야 함
    for i in range(N):
        for j in range(M):
            # 칸 (i, j)를 덮을 수 있는 패치의 왼쪽 상단 시작점 (r, c) 범위:
            # i-a+1 <= r <= i  AND  j-b+1 <= c <= j
            applicable_patches = [
                choices[r][c]
                for r in range(max(0, i-a+1), min(i+1, N-a+1))
                for c in range(max(0, j-b+1), min(j+1, M-b+1))
            ]
            if applicable_patches:
                prob += pulp.lpSum(applicable_patches) <= 1

    # 7. 솔버 실행 (PULP_CBC_CMD는 기본 내장 솔버)
    # msg=0은 로그 출력을 끕니다.
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 8. 결과 추출
    best_patches = []
    for r in range(N-a+1):
        for c in range(M-b+1):
            if pulp.value(choices[r][c]) == 1:
                best_patches.append((r, c))
                
    total_score = int(pulp.value(prob.objective))
    
    return np.array(best_patches), total_score

def visualize_patches(grid, patches, a, b, days, save_path=None):
    """
    grid: 전체 데이터 격자 (sum_grid)
    patches: get_patch 함수에서 반환된 (r, c) 좌표 리스트
    a, b: 패치의 높이(세로), 너비(가로)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 배경 Heatmap 그리기
    im = ax.imshow(grid, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax, label='Value')

    # 2D 누적합을 다시 구하지 않고 각 패치의 점수를 직접 계산하여 표기
    for r, c in patches:
        # 패치 영역 추출 및 합계 계산
        patch_score = grid[r:r+a, c:c+b].sum() / days
        
        # 2. 패치 테두리 그리기 (Rectangle)
        # matplotlib의 Rectangle은 (x, y)가 (column, row) 순서임에 주의
        rect = mpatches.Rectangle((c - 0.5, r - 0.5), b, a, 
                                  linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # 3. 패치 중앙에 점수 텍스트 적기
        # 가독성을 위해 배경색이 어두우면 흰색, 밝으면 검은색으로 조정 가능
        ax.text(c + b/2 - 0.5, r + a/2 - 0.5, f'{int(patch_score)}',
                color='white', fontweight='bold', ha='center', va='center',
                fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    ax.set_title(f'Selected {len(patches)} Patches (Size: {a}x{b})', fontsize=15)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # 스크립트 기준 경로 고정
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'
    output_dir = base_dir.parent / 'output'

    csv_path = data_dir / 'extraction.csv'

    df = data_load(data_dir, 'extraction.csv', essential_cols=['call_date', 'xpos', 'ypos'])
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H')

    df['xpos'] = zero_based_regulization(df, 'xpos')
    df['ypos'] = zero_based_regulization(df, 'ypos')
    print(f'max xpos: {df["xpos"].max()}, max ypos: {df["ypos"].max()}')

    data = df2np(df)

    grid = create_grid(data, GRID_SIZE)
    np.save(output_dir / f'grid({GRID_SIZE}).npy', grid)
    print(f'Grid shape: {grid.shape}')
    sum_grid = grid.sum(axis=0)

    gridhitmap(sum_grid, save_path=img_dir / 'grid_heatmap.png')

    patches, total_score = get_patch(sum_grid, A, B, N)
    coverage = A*B*N/(grid.shape[1] * grid.shape[2])
    
    print(f'Coverage: {coverage:.4f}')
    print(f'Total Score of Selected Patches: {total_score}/{sum_grid.sum()}, {total_score/sum_grid.sum():.4f}')
    visualize_patches(sum_grid, patches, A, B, grid.shape[0] // 24, save_path=img_dir / f'selected_patches({GRID_SIZE}).png')

    demands = []
    for r, c in patches:
        demand = grid[:, r:r+A, c:c+B].sum(axis=(1,2))
        demands.append(demand)
    demands = np.array(demands)

    print(f'Demands shape: {demands.shape}')
    np.save(output_dir / f'demands({GRID_SIZE}){N}_{A}x{B}.npy', demands)

    weekly_demands = demands.reshape(demands.shape[0], -1, 7*24).mean(axis=1)
    print(f'Weekly demands shape: {weekly_demands.shape}')
    #np.save(output_dir / f'weekly_demands({GRID_SIZE})_{N}_{A}x{B}.npy', weekly_demands)

    dtw_distances = compute_dtw_matrix(weekly_demands) if not (output_dir / f'dtw_distances({GRID_SIZE}){N}_{A}x{B}.npy').exists() else np.load(output_dir / f'dtw_distances({GRID_SIZE}){N}_{A}x{B}.npy')
    np.save(output_dir / f'dtw_distances({GRID_SIZE}){N}_{A}x{B}.npy', dtw_distances)

    visualize_dtw(dtw_distances, save_path=img_dir / f'dtw_distance_distribution({GRID_SIZE})_{N}_{A}x{B}.png')

    num_nodes = demands.shape[0]

    #통계 출력(5점요약)
    dtw_distances_arr = np.sort(dtw_distances.copy().flatten())[num_nodes:]  # 자기 자신과의 거리는 제외    
    five_num(dtw_distances_arr, 'Node DTW Distance Statistics')

    #edge 생성
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if dtw_distances[i, j] > CONNECTION_THRESHOLD:
                continue
            edges.append((i, j, dtw_distances[i, j]))
            edges.append((j, i, dtw_distances[i, j]))
    print(f'Number of edges: {len(edges)}')

    assign_matrix = make_clusters(demands, dtw_distances, [8])
    print(f'Assign matrix shape: {assign_matrix.shape}')
    np.save(output_dir / f'assign_matrix({GRID_SIZE})_{N}_{A}x{B}.npy', assign_matrix)

    cluster_demands = np.matmul(assign_matrix.T, demands) # T, C
    #np.save(output_dir / f'cluster_demands_{N}_{A}x{B}.npy', cluster_demands)
   # print(f'Cluster demands shape: {cluster_demands.shape}')

    weekly_cluster_demands = cluster_demands.reshape(cluster_demands.shape[0], -1, 7*24).mean(axis=1)
    #np.save(output_dir / f'weekly_cluster_demands_{N}_{A}x{B}.npy', weekly_cluster_demands)
    
    cluster_dtw_distances = compute_dtw_matrix(weekly_cluster_demands) if not (output_dir / f'cluster_dtw_distances({GRID_SIZE})_{N}_{A}x{B}.npy').exists() else np.load(output_dir / f'cluster_dtw_distances({GRID_SIZE})_{N}_{A}x{B}.npy')
    np.save(output_dir / f'cluster_dtw_distances({GRID_SIZE})_{N}_{A}x{B}.npy', cluster_dtw_distances)

    cluster_edges = []
    num_clusters = cluster_demands.shape[0]
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            if cluster_dtw_distances[i, j] > CLUSTER_CONNECTION_THRESHOLD:
                continue
            cluster_edges.append((i, j, cluster_dtw_distances[i, j]))
            cluster_edges.append((j, i, cluster_dtw_distances[i, j]))
    print(f'Number of cluster edges: {len(cluster_edges)}')
    cluster_dtw_distances_arr = np.sort(cluster_dtw_distances.copy().flatten())[num_clusters:]  # 자기 자신과의 거리는 제외    
    five_num(cluster_dtw_distances_arr, 'Cluster DTW Distance Statistics')
    # print(weekly_demands.shape[1])

    OutJson(
        minHour=df['call_date'].min(),
        maxHour=df['call_date'].max(),
        self_loop_w=1.0,
        total_nodes=len(patches),
        edges=edges,
        demands=demands.T.tolist(),
        assignment_matrix=assign_matrix.tolist(),
        clusters_demand=cluster_demands.T.tolist(),
        clusters_edge=cluster_edges,
        dropped_demand=int(sum_grid.sum() - total_score) / demands.shape[1],
    ).save_json(output_dir / 'gwn_data.json')

    