"""
그래프 생성 스크립트
1. 전처리된 데이터에서 시계열 demand grid 생성
2. 최적 패치 선택 (get_patch)
3. 각 노드의 토지 이용 구성 및 위치 계산
4. JSON 형식으로 저장
"""
from __future__ import annotations

import re
import sys
import json
import re
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.stats import f
from shapely import bounds
from tqdm import tqdm
from shapely.geometry import Point
import pulp

# extract_frame.py의 전처리 함수들 import
sys.path.append(str(Path(__file__).parent / 'control_origin_data'))
from extract_frame import remove_other_region, eleminate_duplicates, crop_filter

# 좌표 범위 설정
US_RANGE_X = (128_950739, 129_500739)
US_RANGE_Y = (35_756099, 35_305729)

GRID_SIZE = 950  # 100m
TARGET_CRS = 'EPSG:5174'

# 패치 설정
N_PATCHES = 50  # 선택할 패치 개수
PATCH_SIZE = 7  # 7x7 그리드 (700m x 700m)

CELLS_PER_PATCH = 1  # 패치당 셀 개수 (get_patch_cell용)

# 한국 공휴일 설정 (2024-2025)
# 형식: 'YYYY-MM-DD'
KOREAN_HOLIDAYS = [
    # 2024년 공휴일
    '2024-01-01',  # 신정
    '2024-02-09',  # 설날 연휴
    '2024-02-10',  # 설날
    '2024-02-11',  # 설날 연휴
    '2024-02-12',  # 대체공휴일
    '2024-03-01',  # 삼일절
    '2024-04-10',  # 국회의원선거
    '2024-05-05',  # 어린이날
    '2024-05-06',  # 대체공휴일
    '2024-05-15',  # 석가탄신일
    '2024-06-06',  # 현충일
    '2024-08-15',  # 광복절
    '2024-09-16',  # 추석 연휴
    '2024-09-17',  # 추석
    '2024-09-18',  # 추석 연휴
    '2024-10-03',  # 개천절
    '2024-10-09',  # 한글날
    '2024-12-25',  # 성탄절
    # 2025년 공휴일
    '2025-01-01',  # 신정
    '2025-01-28',  # 설날 연휴
    '2025-01-29',  # 설날
    '2025-01-30',  # 설날 연휴
    '2025-03-01',  # 삼일절
    '2025-03-03',  # 대체공휴일
    '2025-05-05',  # 어린이날
    '2025-05-06',  # 석가탄신일
    '2025-06-06',  # 현충일
    '2025-08-15',  # 광복절
    '2025-10-05',  # 추석 연휴
    '2025-10-06',  # 추석
    '2025-10-07',  # 추석 연휴
    '2025-10-08',  # 대체공휴일
    '2025-10-03',  # 개천절
    '2025-10-09',  # 한글날
    '2025-12-25',  # 성탄절
]


def load_and_preprocess_data(csv_path: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    데이터 로드 및 전처리 (시계열 유지)
    """
    print("="*60)
    print("Loading and preprocessing data with timestamps")
    print("="*60)
    
    # 원본 데이터 로드
    print("\n[1/4] Loading CSV...")
    df = pd.read_csv(csv_path, encoding='cp949')
    print(f"  Initial rows: {len(df):,}")
    
    # 필수 컬럼 확인
    required_cols = ['xpos', 'ypos', 'clientid', 'call_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns missing: {required_cols}")
    
    # 시간 변환 (전처리 전에 수행)
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    
    # 전처리 적용
    print("\n[2/4] Applying remove_other_region...")
    df = remove_other_region(df, x_col='xpos', y_col='ypos', 
                             x_range=US_RANGE_X, y_range=US_RANGE_Y)
    
    print("\n[3/4] Applying eleminate_duplicates...")
    df = eleminate_duplicates(df, client_col='clientid', 
                             time_col='call_date', window_seconds=600)
    
    print("\n[4/4] Applying crop_filter...")
    df = crop_filter(df, threshold_rate=0.9, step_rate=0.001)
    
    print(f"\nFinal preprocessed data: {len(df):,} rows")
    
    # GeoDataFrame 생성 (위치 정보용)
    df_geo = df[['xpos', 'ypos']].copy()
    df_geo['lon'] = df_geo['xpos'] / 1_000_000
    df_geo['lat'] = df_geo['ypos'] / 1_000_000
    
    gdf = gpd.GeoDataFrame(
        df_geo,
        geometry=gpd.points_from_xy(df_geo['lon'], df_geo['lat']),
        crs='EPSG:4326'
    )
    gdf = gdf.to_crs(TARGET_CRS)
    
    # dest 컬럼이 있는지 확인하고 유지
    if 'dest_xpos' in df.columns and 'dest_ypos' in df.columns:
        print(f"  Destination columns preserved")
    #df.to_csv(csv_path.parent / 'preprocessed_data.csv', index=False, encoding='cp949')
    return df, gdf


def create_temporal_grid(df: pd.DataFrame, gdf: gpd.GeoDataFrame, 
                        bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    시계열 demand grid 생성 (T, H, W)
    T: 시간 스텝 (시간 단위)
    H, W: 공간 그리드
    """
    print("\n" + "="*60)
    print("Creating temporal demand grid")
    print("="*60)
    
    min_x, max_x, min_y, max_y = bounds
    
    # Grid 크기 계산
    width = max_x - min_x
    height = max_y - min_y
    n_cols = int(np.ceil(width / GRID_SIZE))
    n_rows = int(np.ceil(height / GRID_SIZE))
    
    # 시간 범위 확인
    df_with_time = df.copy()
    df_with_time['hour'] = df_with_time['call_date'].dt.floor('h')
    
    unique_hours = sorted(df_with_time['hour'].unique())
    n_timesteps = len(unique_hours)
    
    print(f"\nTemporal grid configuration:")
    print(f"  Spatial: {n_rows} rows x {n_cols} columns")
    print(f"  Temporal: {n_timesteps} time steps (hours)")
    print(f"  Time range: {unique_hours[0]} to {unique_hours[-1]}")
    print(f"  Total shape: ({n_timesteps}, {n_rows}, {n_cols})")
    
    # 시계열 grid 초기화
    temporal_grid = np.zeros((n_timesteps, n_rows, n_cols), dtype=np.int32)
    
    # 좌표 계산
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    
    cols = ((x_coords - min_x) / GRID_SIZE).astype(int)
    rows = ((max_y - y_coords) / GRID_SIZE).astype(int)
    
    # 시간별로 데이터 분류
    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}
    df_with_time['time_idx'] = df_with_time['hour'].map(hour_to_idx)
    
    # 각 데이터 포인트를 시계열 grid에 매핑
    print("\nMapping demands to temporal grid...")
    for idx in tqdm(range(len(df_with_time)), desc="Processing records"):
        time_idx = df_with_time.iloc[idx]['time_idx']
        row, col = rows[idx], cols[idx]
        
        if 0 <= row < n_rows and 0 <= col < n_cols and pd.notna(time_idx):
            temporal_grid[int(time_idx), row, col] += 1
    
    print(f"\nTemporal grid statistics:")
    print(f"  Total demands: {temporal_grid.sum():,}")
    print(f"  Non-zero cells (space-time): {np.count_nonzero(temporal_grid):,}")
    
    return temporal_grid, unique_hours


def get_patch(grid: np.ndarray, a: int, b: int, x: int,
              bounds: tuple[float, float, float, float]) -> tuple[list[list[dict]], int]:
    """
    최적 패치 선택 및 구성 셀 정보 반환
    Returns: (List[List[Dict]], total_score)
    patches[i] = [{'x': grid_col, 'y': grid_row, 'lat': lat, 'lon': lon}, ...]
    """
    N, M = grid.shape
    
    # 2D 누적 합
    S = np.zeros((N + 1, M + 1), dtype=np.int64)
    for i in range(N):
        for j in range(M):
            S[i+1, j+1] = grid[i, j] + S[i, j+1] + S[i+1, j] - S[i, j]
            
    def get_sum(r, c):
        return int(S[r+a, c+b] - S[r, c+b] - S[r+a, c] + S[r, c])

    # 최적화 문제
    prob = pulp.LpProblem("Patch_Optimization", pulp.LpMaximize)
    choices = pulp.LpVariable.dicts("Choice", (range(N-a+1), range(M-b+1)), cat='Binary')
    
    # 목적 함수
    prob += pulp.lpSum([choices[r][c] * get_sum(r, c) 
                        for r in range(N-a+1) for c in range(M-b+1)])
    
    # 제약 조건 1: 패치 개수
    prob += pulp.lpSum([choices[r][c] for r in range(N-a+1) for c in range(M-b+1)]) == x
    
    # 제약 조건 2: 겹침 방지
    for i in range(N):
        for j in range(M):
            applicable_patches = [
                choices[r][c]
                for r in range(max(0, i-a+1), min(i+1, N-a+1))
                for c in range(max(0, j-b+1), min(j+1, M-b+1))
            ]
            if applicable_patches:
                prob += pulp.lpSum(applicable_patches) <= 1

    # 솔버 실행
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 결과 추출 - 사각형 좌상단 좌표
    best_patches_tl = []
    for r in range(N-a+1):
        for c in range(M-b+1):
            if pulp.value(choices[r][c]) == 1:
                best_patches_tl.append((r, c))
                
    total_score = int(pulp.value(prob.objective))
    
    # 각 패치를 구성하는 셀 정보 생성
    print("  Generating patch cell details...")
    patches = [[] for _ in range(len(best_patches_tl))]
    
    min_x, max_x, min_y, max_y = bounds
    
    # 1. 모든 패치의 모든 셀 좌표 수집
    all_cells = [] # (row, col, patch_idx)
    
    for idx, (r_start, c_start) in enumerate(best_patches_tl):
        for i in range(a):
            for j in range(b):
                r = r_start + i
                c = c_start + j
                if r < N and c < M:
                    all_cells.append((r, c, idx))
    
    if not all_cells:
         return [], total_score

    # 2. 좌표 변환을 위한 일괄 처리
    rows = np.array([item[0] for item in all_cells])
    cols = np.array([item[1] for item in all_cells])
    indices = [item[2] for item in all_cells]
    
    # Grid 중심 좌표 (EPSG:5174)
    xs = min_x + (cols + 0.5) * GRID_SIZE
    ys = max_y - (rows + 0.5) * GRID_SIZE
    
    # WGS84 변환
    points_df = gpd.GeoDataFrame(
        {'geometry': gpd.points_from_xy(xs, ys)},
        crs=TARGET_CRS
    ).to_crs('EPSG:4326')
    
    lons = points_df.geometry.x.values
    lats = points_df.geometry.y.values
    
    # 3. 결과 구조화
    for k, idx in enumerate(indices):
        cell_info = {
            'x': int(cols[k]),
            'y': int(rows[k]),
            'lat': float(lats[k]),
            'lon': float(lons[k])
        }
        patches[idx].append(cell_info)
    
    return patches, total_score


def map_landuse_to_grid(gdf: gpd.GeoDataFrame, grid_shape: tuple[int, int], 
                        bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Grid의 각 셀에 해당하는 용도(ATRB_SE)를 매핑
    """
    print("  Mapping land use to grid...")
    
    min_x, max_x, min_y, max_y = bounds
    print(f"  Grid bounds: x({min_x}, {max_x}), y({min_y}, {max_y})")
    n_rows, n_cols = grid_shape
    
    landuse_grid = np.zeros((n_rows, n_cols), dtype=np.int8)
    
    def atrb_to_num(atrb: str) -> int:
        if atrb.startswith('UQA'):
            try:
                return int(atrb[3])
            except (ValueError, IndexError):
                print(f"Warning: Unexpected ATRB_SE format: {atrb}")
                return 0
        return 0
    
    sindex = gdf.sindex
    
    for row_idx in tqdm(range(n_rows), desc="  Mapping rows"):
        for col_idx in range(n_cols):
            center_x = min_x + (col_idx + 0.5) * GRID_SIZE
            center_y = max_y - (row_idx + 0.5) * GRID_SIZE
            
            point = Point(center_x, center_y)
            
            possible_matches_idx = list(sindex.intersection(point.bounds))
            if possible_matches_idx:
                possible_matches = gdf.iloc[possible_matches_idx]
                containing = possible_matches[possible_matches.contains(point)]
                
                if len(containing) > 0:
                    atrb = containing.iloc[0]['ATRB_SE']
                    landuse_grid[row_idx, col_idx] = atrb_to_num(atrb)
    
    print(f"  Classified: {np.count_nonzero(landuse_grid)}/{landuse_grid.size} cells")
    return landuse_grid


def compute_node_composition(landuse_grid: np.ndarray, patches: list[list[dict]],
                             poi_gdf: gpd.GeoDataFrame = None,
                             bounds: tuple[float, float, float, float] = None) -> list[dict]:
    """
    각 노드(패치)의 토지 이용 구성 및 POI 계산
    patches: List[List[{'x': col, 'y': row, ...}]]
    poi_gdf: POI 정보를 담은 GeoDataFrame (옵션)
    bounds: 그리드 범위 (min_x, max_x, min_y, max_y) (POI 계산 시 필수)
    """
    print("\n" + "="*60)
    print("Computing node composition (Land Use + POI)")
    print("="*60)
    
    # POI 매핑 준비
    poi_grid_map = defaultdict(list)
    if poi_gdf is not None and bounds is not None:
        print("  Mapping POIs to grid...")
        min_x, max_x, min_y, max_y = bounds
        
        # 좌표 추출
        poi_x = poi_gdf.geometry.x.values
        poi_y = poi_gdf.geometry.y.values
        categories = poi_gdf['개방서비스명'].values
        
        # Grid 좌표 계산
        poi_cols = ((poi_x - min_x) / GRID_SIZE).astype(int)
        poi_rows = ((max_y - poi_y) / GRID_SIZE).astype(int)
        
        # 유효 범위 필터링
        n_rows, n_cols = landuse_grid.shape
        valid_mask = (poi_rows >= 0) & (poi_rows < n_rows) & \
                     (poi_cols >= 0) & (poi_cols < n_cols)
        
        valid_rows = poi_rows[valid_mask]
        valid_cols = poi_cols[valid_mask]
        valid_cats = categories[valid_mask]
        
        for r, c, cat in zip(valid_rows, valid_cols, valid_cats):
            poi_grid_map[(r, c)].append(cat)
            
        print(f"  Mapped {len(valid_rows):,} POIs to grid")
    
    node_compositions = []
    for idx, patch_cells in enumerate(tqdm(patches, desc="Computing compositions")):
        # 패치를 구성하는 모든 셀의 값 가져오기
        composition = Counter()
        poi = Counter()
        
        for cell in patch_cells:
            r, c = cell['y'], cell['x']
            
            # 1. Land Use Counting
            try:
                val = landuse_grid[r, c]
                if val > 0:
                    composition[f'UQA{val}'] += 1
                elif val == 0:
                    composition['Unclassified'] += 1
            except IndexError:
                pass
            
            # 2. POI Counting
            if (r, c) in poi_grid_map:
                for cat in poi_grid_map[(r, c)]:
                    if pd.notna(cat):
                        poi[cat] += 1
        
        node_compositions.append({
            'land_use': dict(composition),
            'poi': dict(poi)
        })
    
    return node_compositions


def create_integrated_nodes(patches: list[list[dict]], 
                           grid_size: int, node_compositions: list[dict]) -> list[dict]:
    """
    노드 정보 통합: node_id, lat, lon, composition
    lat, lon은 패치 구성 셀들의 평균 좌표 사용
    """
    print("\n" + "="*60)
    print("Creating integrated node information")
    print("="*60)
    
    nodes = []
    
    for idx, patch_cells in enumerate(tqdm(patches, desc="Creating nodes")):
        if not patch_cells:
            continue
            
        # 중심 좌표 계산 (평균)
        lats = [c['lat'] for c in patch_cells]
        lons = [c['lon'] for c in patch_cells]
        
        avg_lat = sum(lats) / len(lats)
        avg_lon = sum(lons) / len(lons)
        
        cell_coords = [[c['y'], c['x']] for c in patch_cells]
        
        # 통합 노드 정보
        nodes.append({
            'node_id': idx,
            'lat': float(avg_lat),
            'lon': float(avg_lon),
            'composition': node_compositions[idx],
            'cells': cell_coords,
            # Size: 셀 개수 * 단위 면적
            'size': len(patch_cells) * (grid_size/1000)**2  # km² 단위
        })
    
    return nodes


def create_coord_to_node_mapping(patches: list[list[dict]]) -> dict:
    """
    좌표(grid row, col)를 노드 인덱스로 매핑하는 딕셔너리 생성
    Returns: {(row, col): node_idx}
    """
    coord_to_node = {}
    
    for node_idx, patch_cells in enumerate(patches):
        for cell in patch_cells:
            coord_to_node[(cell['y'], cell['x'])] = node_idx
    
    return coord_to_node


def extract_node_demands(temporal_grid: np.ndarray, patches: list[list[dict]]) -> np.ndarray:
    """
    각 노드(패치)의 시계열 수요 추출
    Returns: (T, N) 배열
    """
    print("\n" + "="*60)
    print("Extracting node demands")
    print("="*60)
    
    n_timesteps = temporal_grid.shape[0]
    n_nodes = len(patches)
    
    demands = np.zeros((n_timesteps, n_nodes), dtype=np.int32)
    
    for idx, patch_cells in enumerate(tqdm(patches, desc="Extracting demands")):
        if not patch_cells:
            continue
            
        rows = [c['y'] for c in patch_cells]
        cols = [c['x'] for c in patch_cells]
        
        # 각 시간 스텝에서 패치 영역의 수요 합계
        # temporal_grid[:, rows, cols] -> (T, N_cells)
        demands[:, idx] = temporal_grid[:, rows, cols].sum(axis=1)
    
    print(f"\nDemand extraction complete:")
    print(f"  Shape: {demands.shape} (T={n_timesteps}, N={n_nodes})")
    print(f"  Total demand: {demands.sum():,}")
    print(f"  Average demand per node per hour: {demands.mean():.2f}")
    
    return demands


def extract_od_flows(df: pd.DataFrame, gdf: gpd.GeoDataFrame,
                    unique_hours: list, coord_to_node: dict,
                    bounds: tuple[float, float, float, float]) -> list[list[dict]]:
    """
    각 시간 스텝에 대한 OD 흐름 추출
    Returns: List[List[Dict]] - 각 timestep별 OD 흐름 리스트
    """
    print("\n" + "="*60)
    print("Extracting OD flows")
    print("="*60)
    
    min_x, max_x, min_y, max_y = bounds
    
    # dest 컬럼이 있는지 확인
    if 'dest_xpos' not in df.columns or 'dest_ypos' not in df.columns:
        print("  No destination columns found, skipping OD extraction")
        return [[] for _ in range(len(unique_hours))]
    
    # 결측치 제거
    df_with_dest = df.dropna(subset=['dest_xpos', 'dest_ypos']).copy()
    print(f"  Records with valid destinations: {len(df_with_dest):,} / {len(df):,}")
    
    if len(df_with_dest) == 0:
        return [[] for _ in range(len(unique_hours))]
    
    # 시간 인덱스 추가
    df_with_dest['hour'] = df_with_dest['call_date'].dt.floor('h')
    hour_to_idx = {hour: idx for idx, hour in enumerate(unique_hours)}
    df_with_dest['time_idx'] = df_with_dest['hour'].map(hour_to_idx)
    
    # 출발지 좌표를 EPSG:5174로 변환 (이미 gdf에 있음)
    origin_coords = gdf.loc[df_with_dest.index]
    origin_x = origin_coords.geometry.x.values
    origin_y = origin_coords.geometry.y.values
    
    # 도착지 좌표를 EPSG:5174로 변환
    df_with_dest['dest_lon'] = df_with_dest['dest_xpos'] / 1_000_000
    df_with_dest['dest_lat'] = df_with_dest['dest_ypos'] / 1_000_000
    
    gdf_dest = gpd.GeoDataFrame(
        df_with_dest[['dest_lon', 'dest_lat']],
        geometry=gpd.points_from_xy(df_with_dest['dest_lon'], df_with_dest['dest_lat']),
        crs='EPSG:4326'
    )
    gdf_dest = gdf_dest.to_crs(TARGET_CRS)
    
    dest_x = gdf_dest.geometry.x.values
    dest_y = gdf_dest.geometry.y.values
    
    # Grid 좌표 계산
    origin_cols = ((origin_x - min_x) / GRID_SIZE).astype(int)
    origin_rows = ((max_y - origin_y) / GRID_SIZE).astype(int)
    dest_cols = ((dest_x - min_x) / GRID_SIZE).astype(int)
    dest_rows = ((max_y - dest_y) / GRID_SIZE).astype(int)
    
    # 각 레코드에 대해 출발/도착 노드 매핑
    df_with_dest['origin_node'] = -1
    df_with_dest['dest_node'] = -1
    
    for idx in range(len(df_with_dest)):
        origin_coord = (origin_rows[idx], origin_cols[idx])
        dest_coord = (dest_rows[idx], dest_cols[idx])
        
        df_with_dest.iloc[idx, df_with_dest.columns.get_loc('origin_node')] = coord_to_node.get(origin_coord, -1)
        df_with_dest.iloc[idx, df_with_dest.columns.get_loc('dest_node')] = coord_to_node.get(dest_coord, -1)
    
    # 유효한 OD만 필터링 (둘 다 노드에 속함)
    valid_od = df_with_dest[(df_with_dest['origin_node'] >= 0) & (df_with_dest['dest_node'] >= 0)]
    print(f"  Valid OD pairs (both in selected nodes): {len(valid_od):,}")
    
    # 시간별 OD 흐름 계산
    od_flows = [[] for _ in range(len(unique_hours))]
    
    if len(valid_od) > 0:
        # 시간별, OD pair별 그룹화하여 카운트
        grouped = valid_od.groupby(['time_idx', 'origin_node', 'dest_node']).size().reset_index(name='cnt')
        
        print("  Computing OD flows by timestep...")
        for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="  Processing OD pairs"):
            time_idx = int(row['time_idx'])
            u = int(row['origin_node'])
            v = int(row['dest_node'])
            cnt = int(row['cnt'])
            
            if pd.notna(time_idx) and 0 <= time_idx < len(unique_hours):
                od_flows[time_idx].append({'u': u, 'v': v, 'cnt': cnt})
    
    # 통계 출력
    total_od_records = sum(len(od_list) for od_list in od_flows)
    total_od_count = sum(od['cnt'] for od_list in od_flows for od in od_list)
    non_empty_timesteps = sum(1 for od_list in od_flows if len(od_list) > 0)
    
    print(f"\nOD flow statistics:")
    print(f"  Timesteps with OD data: {non_empty_timesteps} / {len(unique_hours)}")
    print(f"  Unique OD pairs (across all time): {total_od_records:,}")
    print(f"  Total OD trips: {total_od_count:,}")
    
    return od_flows

def get_patch_cell(grid: np.ndarray, n_patches: int, n_cells_per_patch: int,
                             bounds: tuple[float, float, float, float]) -> tuple[list[list[dict]], int]:
    """
    영역 확장(Region Growing) 방식의 패치 생성
    1. 방문하지 않은 가장 높은 수요의 셀을 시드(Seed)로 선정
    2. 해당 패치에 인접한 셀 중 가장 수요가 높은 셀을 추가 (반복)
    3. 지정된 n_cells_per_patch에 도달하면 다음 패치 생성
    """
    print("\n" + "="*60)
    print(f"Generating patches using Region Growing (Size: {n_cells_per_patch} cells)")
    print("="*60)

    N, M = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    patches = []
    total_score = 0
    
    min_x, max_x, min_y, max_y = bounds

    # 4방향 탐색 (상, 하, 좌, 우) - 대각선 포함시 방향 추가 필요
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in tqdm(range(n_patches), desc="Growing regions"):
        # 1. 시드 찾기: 방문하지 않은 곳 중 수요가 가장 높은 곳 (Greedy)
        # 마스킹된 그리드 생성
        masked_grid = np.where(visited, -1, grid)
        if masked_grid.max() == -1: # 더 이상 가용한 셀이 없음
            break
            
        seed_flat_idx = np.argmax(masked_grid)
        seed_r, seed_c = divmod(seed_flat_idx, M)
        
        current_patch_indices = [(seed_r, seed_c)]
        visited[seed_r, seed_c] = True
        current_patch_score = grid[seed_r, seed_c]
        
        # 2. 영역 확장
        # 인접 후보군 관리 (좌표 중복 방지 위해 set 사용)
        candidates = set()
        
        def add_neighbors(r, c):
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < M and not visited[nr, nc]:
                    candidates.add((nr, nc))

        add_neighbors(seed_r, seed_c)

        while len(current_patch_indices) < n_cells_per_patch and candidates:
            # 후보군 중 가장 수요가 높은 셀 선택
            best_candidate = None
            max_val = -1
            
            for r, c in candidates:
                if grid[r, c] > max_val:
                    max_val = grid[r, c]
                    best_candidate = (r, c)
            
            if best_candidate is None: # 후보가 더 이상 없을 경우 (고립된 지역)
                break
                
            # 선택된 셀을 패치에 추가
            br, bc = best_candidate
            visited[br, bc] = True # 방문 처리
            current_patch_indices.append((br, bc))
            current_patch_score += max_val
            
            # 후보군 업데이트 (선택된 노드는 제거, 그 노드의 이웃 추가)
            candidates.remove(best_candidate)
            add_neighbors(br, bc)
            
            # 이미 방문한 노드가 candidates에 다시 들어가지 않도록 필터링 (add_neighbors에서 처리됨)
            # 하지만 candidates 내에 있던 다른 노드가 이번 루프에서 visited가 될 일은 없으므로 안전

        total_score += current_patch_score

        # 3. 좌표 변환 및 결과 포맷팅 (기존 로직과 호환성 유지)
        patch_cells = []
        for r, c in current_patch_indices:
            # Grid 중심 좌표 (EPSG:5174)
            cx = min_x + (c + 0.5) * GRID_SIZE
            cy = max_y - (r + 0.5) * GRID_SIZE
            
            # 좌표 변환을 위해 임시 GeoDataFrame 사용 (성능 최적화를 위해 루프 밖에서 일괄 처리도 가능하나 가독성 유지)
            # 여기서는 간단히 단일 점 변환 로직 사용 (성능 민감시 기존 코드처럼 일괄 변환 권장)
            pt_df = gpd.GeoDataFrame({'geometry': [Point(cx, cy)]}, crs=TARGET_CRS).to_crs('EPSG:4326')
            lon = pt_df.geometry.x.iloc[0]
            lat = pt_df.geometry.y.iloc[0]
            
            patch_cells.append({
                'x': int(c), # col
                'y': int(r), # row
                'lat': float(lat),
                'lon': float(lon)
            })
        
        patches.append(patch_cells)

    return patches, total_score


def create_temporal_features(unique_hours: list) -> list[dict]:
    """
    각 시간 스텝에 대한 temporal feature 생성
    day: 요일 (Mon, Tue, Wed, Thu, Fri, Sat, Sun)
    time: 시간 (0-23)
    holiday: 한국 공휴일 여부 (True/False)
    """
    print("\n" + "="*60)
    print("Creating temporal features")
    print("="*60)
    
    # 공휴일 set으로 변환 (빠른 검색)
    holiday_set = set(KOREAN_HOLIDAYS)
    
    # 요일 매핑
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    temporal_features = []
    
    for hour in tqdm(unique_hours, desc="Processing timestamps"):
        date_str = hour.strftime('%Y-%m-%d')
        
        feature = {
            'day': day_names[hour.weekday()],  # 0=Monday, 6=Sunday
            'time': hour.hour,  # 0-23
            'holiday': date_str in holiday_set
        }
        temporal_features.append(feature)
    
    # 통계 출력
    n_holidays = sum(1 for f in temporal_features if f['holiday'])
    print(f"\nTemporal features statistics:")
    print(f"  Total timesteps: {len(temporal_features)}")
    print(f"  Holiday timesteps: {n_holidays} ({n_holidays/len(temporal_features)*100:.2f}%)")
    
    # 요일별 통계
    day_counts = {day: sum(1 for f in temporal_features if f['day'] == day) for day in day_names}
    print(f"  Day distribution:")
    for day, count in day_counts.items():
        print(f"    {day}: {count} timesteps")
    
    return temporal_features


def save_graph_json(output_path: Path, nodes: list[dict], demands: np.ndarray,
                    temporal_features: list[dict], od_flows: list[list[dict]]):
    """
    그래프 데이터를 새로운 JSON 형식으로 저장 (OD 정보 포함)
    """
    print("\n" + "="*60)
    print("Saving graph to JSON")
    print("="*60)
    
    # x 배열 생성: 각 timestep에 대한 demand + temporal features + OD
    x = []
    for t in range(len(demands)):
        x_t = {
            'demand': demands[t].tolist(),  # N개 노드의 수요
            'day': temporal_features[t]['day'],
            'time': temporal_features[t]['time'],
            'holiday': temporal_features[t]['holiday'],
            'OD': od_flows[t]  # OD 흐름 정보
        }
        x.append(x_t)
    
    data = {
        'nodes': nodes,  # 통합된 노드 정보
        'x': x  # 시계열 데이터 + temporal features + OD
    }
    
    json_str = json.dumps(data, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    def compact_list(match):
        return match.group(0).replace('\n', '').replace(' ', '')
    
    compact_json = re.sub(r'\[[\s\d,]+\]', compact_list, json_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(compact_json)
    print(f"\nJSON saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nData structure:")
    print(f"  nodes: {len(nodes)} nodes")
    print(f"  x: {len(x)} timesteps")
    print(f"    - Each timestep has: demand (list of {len(demands[0])}), day, time, holiday, OD (list)")


def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'
    
    origin_csv = data_dir / 'origin_data.csv'
    shp_file = data_dir / 'UPIS_C_UQ111.shp'
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "GRAPH GENERATION FROM GRID" + " "*16 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Step 1: 데이터 전처리
    df, gdf = load_and_preprocess_data(origin_csv) #df: demand정보의 dataframe, gdf: demand정보의 geodataframe
    
    # 좌표 범위 계산
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())
    
    # Step 2: 시계열 demand grid 생성
    temporal_grid, unique_hours = create_temporal_grid(df, gdf, bounds) # temporal_grid: (T,H,W)크기의 numpy ndarray, unique_hours: 시간 스텝 리스트
    
    np.save(output_dir / 'temporal_grid.npy', temporal_grid)
    
    # Step 3: 총 수요 grid (시간 합산)
    sum_grid = temporal_grid.sum(axis=0)
    
    # Step 4: 최적 패치 선택
    print("\n" + "="*60)
    print("Selecting optimal patches")
    print("="*60)
    
    # 수정됨: bounds 전달, 반환값은 (List[List[Dict]], score)
    # 만약에 기존 방식을 사용할 것이면 아래 주석 해제
    # patches, total_score = get_patch(sum_grid, PATCH_SIZE, PATCH_SIZE, N_PATCHES, bounds)
    # 영역 확장 방식 사용
    patches, total_score = get_patch_cell(sum_grid, N_PATCHES, CELLS_PER_PATCH, bounds)
    print(f'patches sample: {patches[0][:3] if patches and patches[0] else "No patches"}')
    
    # 패치 구성 셀 개수로 커버리지 계산
    total_cells = sum(len(p) for p in patches)
    coverage = total_cells / (sum_grid.shape[0] * sum_grid.shape[1])
    
    print(f"\nPatch selection results:")
    print(f"  Selected patches: {len(patches)}")
    print(f"  Total cells in patches: {total_cells}")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Total score: {total_score:,} / {sum_grid.sum():,} ({total_score/sum_grid.sum()*100:.2f}%)")
    
    # Step 5: 토지 이용 정보 로드 및 구성 계산
    print("\nLoading shapefile and creating land use grid...")
    
    # landuse_grid가 없으면 생성
    landuse_grid_path = output_dir / 'landuse_grid.npy'
   
    print(f"  Creating landuse_grid...")
    # Shapefile 로드
    gdf_shp = gpd.read_file(shp_file)
    if gdf_shp.crs != TARGET_CRS:
        gdf_shp = gdf_shp.to_crs(TARGET_CRS)
    
    # Land use grid 생성
    landuse_grid = map_landuse_to_grid(
        gdf_shp, 
        sum_grid.shape, 
        bounds
    )
    print(f'  landuse_grid shape: {landuse_grid.shape}')
    np.save(landuse_grid_path, landuse_grid)
    print(f"  Saved landuse_grid to {landuse_grid_path}")

    # POI 데이터 로드 (Main에 추가)
    print("\nLoading POI data...")
    poi_df = pd.read_csv(data_dir / 'poi_data.csv', encoding='utf-8')
    poi_df = poi_df.dropna(subset=['좌표정보x(epsg5174)', '좌표정보y(epsg5174)'])
    poi_gdf = gpd.GeoDataFrame(
        poi_df, 
        geometry=gpd.points_from_xy(poi_df['좌표정보x(epsg5174)'], poi_df['좌표정보y(epsg5174)']), 
        crs="EPSG:5174"
    )
    print(f"  Loaded {len(poi_gdf):,} POIs")

    node_compositions = compute_node_composition(landuse_grid, patches, poi_gdf, bounds)
    
    # Step 6: 노드 정보 통합 (node_id, lat, lon, composition)
    nodes = create_integrated_nodes(patches, GRID_SIZE, node_compositions)
    
    # Step 7: 노드별 시계열 수요 추출
    demands = extract_node_demands(temporal_grid, patches) # demands: 노드별 시계열 수요 리스트
    
    # Step 8: 좌표-노드 매핑 생성
    print("\n" + "="*60)
    print("Creating coordinate-to-node mapping")
    print("="*60)
    coord_to_node = create_coord_to_node_mapping(patches)
    print(f"  Mapped grid cells: {len(coord_to_node):,}")
    
    # Step 9: OD 흐름 추출
    od_flows = extract_od_flows(df, gdf, unique_hours, coord_to_node, bounds)
    
    # Step 10: Temporal features 생성 (day, time, holiday)
    temporal_features = create_temporal_features(unique_hours)
    
    # Step 11: JSON 저장
    save_graph_json(
        output_dir / 'graph_data.json',
        nodes=nodes,
        demands=demands,
        temporal_features=temporal_features,
        od_flows=od_flows
    )
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*18 + "GRAPH GENERATION COMPLETE!" + " "*13 + "║")
    print("╚" + "="*58 + "╝\n")
    
    print(f"Output saved to: {output_dir / 'graph_data.json'}")


if __name__ == '__main__':
    main()
