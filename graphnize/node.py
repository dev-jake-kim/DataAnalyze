import heapq

import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from outjson import OutJson
from visualize.grid2hitmap import gridhitmap

#격자 크기
GRID_X = 1000
GRID_Y = 1000

#
MIN_DATA_PER_NODE = 1000
MAX_NODE_SIZE = 5

MAX_DIST = 1000

def get_max(df):
    max_xpos = df['xpos'].max()
    max_ypos = df['ypos'].max()
    return max_xpos, max_ypos

def make_grid(df):
    max_xpos, max_ypos = get_max(df)
    grid = [[[] for _ in range(max_xpos//GRID_X+1)] for _ in range(max_ypos//GRID_X+1)]
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        xpos = row['xpos']
        ypos = row['ypos']
        grid[ypos//GRID_Y][xpos//GRID_X].append(row)
    return grid

def generate_polyominoes(n):
    """
    n개의 연결된 블록(폴리오미노) 모양을 모두 생성 (중복 회전/대칭 제외)
    각 모양은 (0,0) 기준 상대좌표 집합으로 표현
    """
    from collections import deque
    def normalize(cells):
        # 좌상단 기준 정렬
        min_x = min(x for x, y in cells)
        min_y = min(y for x, y in cells)
        return tuple(sorted((x - min_x, y - min_y) for x, y in cells))

    def rotations_and_reflections(cells):
        # 90도 회전, 대칭 포함 모든 변형
        result = set()
        for k in range(4):
            rotated = [(x, y) for x, y in cells]
            for _ in range(k):
                rotated = [(-y, x) for x, y in rotated]
            for reflect in [False, True]:
                if reflect:
                    reflected = [(-x, y) for x, y in rotated]
                else:
                    reflected = rotated
                result.add(normalize(reflected))
        return result

    seen = set()
    queue = deque()
    queue.append(((0, 0),))
    results = set()
    while queue:
        cells = queue.popleft()
        if len(cells) == n:
            norm = normalize(cells)
            if norm not in seen:
                seen.update(rotations_and_reflections(cells))
                results.add(norm)
            continue
        for x, y in cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) not in cells:
                    new_cells = tuple(sorted(cells + ((nx, ny),)))
                    norm = normalize(new_cells)
                    if norm not in seen:
                        queue.append(new_cells)
    return reversed(list(results))

# 폴리오미노 모양 미리 생성 (1~MAX_NODE_SIZE)
POLYOMINOES = {n: generate_polyominoes(n) for n in range(1, MAX_NODE_SIZE+1)}

def plot_complete_nodes(node_map, complete_node):
    """
    complete_node의 각 노드를 색 다르게, 각 블록에 인덱스(노드 번호) 표시
    """
    h, w = node_map.shape
    # 노드별 색상 배열 생성
    cmap = plt.get_cmap('tab20')
    color_map = np.ones((h, w, 3), dtype=float)
    index_map = np.full((h, w), -1, dtype=int)
    for idx, node in enumerate(complete_node):
        color = cmap(idx % 20)[:3]  # RGB
        for y, x in node:
            color_map[y, x] = color
            index_map[y, x] = idx
    fig, ax = plt.subplots(figsize=(w/2, h/2), facecolor = 'white')
    ax.imshow(color_map, interpolation='none', origin='upper')
    # 인덱스 텍스트 표시
    for y in range(h):
        for x in range(w):
            if index_map[y, x] >= 0:
                ax.text(x, y, str(index_map[y, x]), va='center', ha='center', fontsize=6, color='black', weight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Complete Nodes Visualization')
    plt.tight_layout()
    plt.savefig(Path('../imgs/complete_nodes.png'))
    plt.show()
    plt.close()

def node_center(node, grid):
    """
    node: ((y1,x1), (y2,x2), ...)
    grid: 2차원 리스트, 각 칸에 row 객체 리스트
    -> 해당 노드의 모든 row의 평균 xpos, ypos 반환
    """
    xpos_list = []
    ypos_list = []
    for y, x in node:
        for row in grid[y][x]:
            xpos_list.append(row['xpos'])
            ypos_list.append(row['ypos'])
    if xpos_list:
        return (np.mean(xpos_list), np.mean(ypos_list))
    else:
        return (0, 0)

def distance(c1, c2):
    return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

if __name__ == "__main__":
    csv_path = Path('../data') / 'essential_columns.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H')
    max_xpos, max_ypos = get_max(df)
    print(f"Max xpos: {max_xpos}, Max ypos: {max_ypos}")

    grid = make_grid(df)

    # gridhitmap(grid, logscale=True, title="Grid Hitmap(log)", save_path=Path('../imgs/grid_hitmap.png'))
    # gridhitmap(grid, logscale=False, title="Grid Hitmap(linear)", save_path=Path('../imgs/grid_hitmap_linear.png'))

    grid_x_size = len(grid[0])
    grid_y_size = len(grid)
    print(f"Grid size: {grid_x_size} x {grid_y_size}")

    ground = np.zeros((grid_y_size, grid_x_size), dtype=int)
    for i in range(grid_y_size):
        for j in range(grid_x_size):
            ground[i][j] = len(grid[i][j])

    complete_node = []
    for block_count in range(1, MAX_NODE_SIZE+1):
        polyominoes = POLYOMINOES[block_count]
        for shape in polyominoes:
            # shape: ((0,0),(0,1),...) 상대좌표
            for i in range(grid_y_size):
                for j in range(grid_x_size):
                    coords = []
                    valid = True
                    for dx, dy in shape:
                        x, y = j+dx, i+dy
                        if 0 <= x < grid_x_size and 0 <= y < grid_y_size:
                            coords.append((y, x))
                        else:
                            valid = False
                            break
                    if not valid:
                        continue
                    # 음수(-1) 포함 여부 확인
                    has_negative = any(ground[y][x] < 0 for y, x in coords)
                    block_sum = sum(ground[y][x] for y, x in coords)
                    if not has_negative and block_sum >= MIN_DATA_PER_NODE:
                        for y, x in coords:
                            ground[y][x] = -1
                        complete_node.append(tuple(coords))
    node_sizes = [0]*MAX_NODE_SIZE
    for node in complete_node:
        node_sizes[len(node)-1] += 1
    print(f"Complete nodes: {len(complete_node)}, Sizes: {node_sizes}")

    # complete_node를 2차원 배열에 매핑
    node_map = np.zeros_like(ground)
    for node in complete_node:
        for y, x in node:
            node_map[y][x] = 1
    # 시각화
    plot_complete_nodes(node_map, complete_node)

    # --- 노드 중심좌표 계산 ---
    node_centers = [node_center(node, grid) for node in complete_node]

    # --- 엣지 생성 ---
    edges = []
    n = len(node_centers)
    for i in range(n):
        for j in range(i+1, n):
            dist = 1 - distance(node_centers[i], node_centers[j])/MAX_DIST
            if dist < MAX_DIST:
                edges.append((i, j, dist))
                edges.append((j, i, dist))

    # --- demands 생성 ---
    minHour = df['call_date'].min()
    maxHour = df['call_date'].max()
    total_hours = int((maxHour - minHour).total_seconds() // 3600) + 1
    demands = np.zeros((total_hours, n), dtype=int)
    for node_idx, node in enumerate(complete_node):
        for y, x in node:
            for row in grid[y][x]:
                t = int((row['call_date'] - minHour).total_seconds() // 3600)
                if 0 <= t < total_hours:
                    demands[t][node_idx] += 1
    demands_list = demands.tolist()

    # --- OutJson 생성 및 저장 ---
    outjson = OutJson(
        minHour=minHour,
        maxHour=maxHour,
        total_nodes=n,
        edges=edges,
        demands=demands_list
    )
    outjson.save_json(Path('../data/gwn_data.json'))
