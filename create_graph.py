"""
그래프 생성 스크립트
1. 전처리된 데이터에서 시계열 demand grid 생성
2. 최적 패치 선택 (get_patch)
3. 각 노드의 토지 이용 구성 및 위치 계산
4. JSON 형식으로 저장
"""
from __future__ import annotations

import sys
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point
import pulp

# extract_frame.py의 전처리 함수들 import
sys.path.append(str(Path(__file__).parent / 'control_origin_data'))
from extract_frame import remove_other_region, eleminate_duplicates, crop_filter

# 좌표 범위 설정
US_RANGE_X = (128_950739, 129_500739)
US_RANGE_Y = (35_756099, 35_305729)

GRID_SIZE = 100  # 100m
TARGET_CRS = 'EPSG:5174'

# 패치 설정
N_PATCHES = 50  # 선택할 패치 개수
PATCH_SIZE = 7  # 7x7 그리드 (700m x 700m)

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
    df = crop_filter(df, threshold_rate=0.85, step_rate=0.001)
    
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
    df_with_time['hour'] = df_with_time['call_date'].dt.floor('H')
    
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


def get_patch(grid: np.ndarray, a: int, b: int, x: int) -> tuple[np.ndarray, int]:
    """
    최적 패치 선택 (grid.py의 get_patch 함수)
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
    
    # 결과 추출
    best_patches = []
    for r in range(N-a+1):
        for c in range(M-b+1):
            if pulp.value(choices[r][c]) == 1:
                best_patches.append((r, c))
                
    total_score = int(pulp.value(prob.objective))
    
    return np.array(best_patches), total_score


def map_landuse_to_grid(gdf: gpd.GeoDataFrame, grid_shape: tuple[int, int], 
                        bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Grid의 각 셀에 해당하는 용도(ATRB_SE)를 매핑
    """
    print("  Mapping land use to grid...")
    
    min_x, max_x, min_y, max_y = bounds
    n_rows, n_cols = grid_shape
    
    landuse_grid = np.zeros((n_rows, n_cols), dtype=np.int8)
    
    def atrb_to_num(atrb: str) -> int:
        if atrb.startswith('UQA'):
            try:
                return int(atrb[3])
            except (ValueError, IndexError):
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


def compute_node_composition(landuse_grid: np.ndarray, patches: np.ndarray, 
                             patch_size: int) -> list[dict]:
    """
    각 노드(패치)의 토지 이용 구성 계산
    """
    print("\n" + "="*60)
    print("Computing node land use composition")
    print("="*60)
    
    node_compositions = []
    
    for idx, (r, c) in enumerate(tqdm(patches, desc="Computing compositions")):
        patch_landuse = landuse_grid[r:r+patch_size, c:c+patch_size]
        
        # 각 카테고리별 셀 개수 (비율로 변환하지 않고 개수 유지)
        composition = {}
        
        for category in range(1, 6):
            count = np.sum(patch_landuse == category)
            if count > 0:
                composition[f'UQA{category}'] = int(count)
        
        # 미분류
        unclassified = np.sum(patch_landuse == 0)
        if unclassified > 0:
            composition['Unclassified'] = int(unclassified)
        
        node_compositions.append(composition)
    
    return node_compositions


def create_integrated_nodes(patches: np.ndarray, patch_size: int, bounds: tuple,
                           grid_size: int, node_compositions: list[dict]) -> list[dict]:
    """
    노드 정보 통합: node_id, lat, lon, composition
    """
    print("\n" + "="*60)
    print("Creating integrated node information")
    print("="*60)
    
    min_x, max_x, min_y, max_y = bounds
    nodes = []
    
    for idx, (r, c) in enumerate(tqdm(patches, desc="Creating nodes")):
        # 패치 중심 좌표 (미터)
        center_x = min_x + (c + patch_size / 2) * grid_size
        center_y = max_y - (r + patch_size / 2) * grid_size
        
        # 위도/경도로 변환
        point = gpd.GeoDataFrame(
            {'geometry': [Point(center_x, center_y)]},
            crs=TARGET_CRS
        )
        point_wgs84 = point.to_crs('EPSG:4326')
        
        lon = point_wgs84.geometry.x.values[0]
        lat = point_wgs84.geometry.y.values[0]
        
        # 통합 노드 정보
        nodes.append({
            'node_id': idx,
            'lat': float(lat),
            'lon': float(lon),
            'composition': node_compositions[idx]
        })
    
    return nodes


def extract_node_demands(temporal_grid: np.ndarray, patches: np.ndarray,
                        patch_size: int) -> np.ndarray:
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
    
    for idx, (r, c) in enumerate(tqdm(patches, desc="Extracting demands")):
        # 각 시간 스텝에서 패치 영역의 수요 합계
        demands[:, idx] = temporal_grid[:, r:r+patch_size, c:c+patch_size].sum(axis=(1, 2))
    
    print(f"\nDemand extraction complete:")
    print(f"  Shape: {demands.shape} (T={n_timesteps}, N={n_nodes})")
    print(f"  Total demand: {demands.sum():,}")
    print(f"  Average demand per node per hour: {demands.mean():.2f}")
    
    return demands


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
                    temporal_features: list[dict]):
    """
    그래프 데이터를 새로운 JSON 형식으로 저장
    """
    print("\n" + "="*60)
    print("Saving graph to JSON")
    print("="*60)
    
    # x 배열 생성: 각 timestep에 대한 demand + temporal features
    x = []
    for t in range(len(demands)):
        x_t = {
            'demand': demands[t].tolist(),  # N개 노드의 수요
            'day': temporal_features[t]['day'],
            'time': temporal_features[t]['time'],
            'holiday': temporal_features[t]['holiday']
        }
        x.append(x_t)
    
    data = {
        'nodes': nodes,  # 통합된 노드 정보
        'x': x  # 시계열 데이터 + temporal features
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nData structure:")
    print(f"  nodes: {len(nodes)} nodes")
    print(f"  x: {len(x)} timesteps")
    print(f"    - Each timestep has: demand (list of {len(demands[0])}), day, time, holiday")


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
    df, gdf = load_and_preprocess_data(origin_csv)
    
    # 좌표 범위 계산
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    bounds = (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max())
    
    # Step 2: 시계열 demand grid 생성
    temporal_grid, unique_hours = create_temporal_grid(df, gdf, bounds)
    
    # Step 3: 총 수요 grid (시간 합산)
    sum_grid = temporal_grid.sum(axis=0)
    
    # Step 4: 최적 패치 선택
    print("\n" + "="*60)
    print("Selecting optimal patches")
    print("="*60)
    
    patches, total_score = get_patch(sum_grid, PATCH_SIZE, PATCH_SIZE, N_PATCHES)
    coverage = (PATCH_SIZE * PATCH_SIZE * N_PATCHES) / (sum_grid.shape[0] * sum_grid.shape[1])
    
    print(f"\nPatch selection results:")
    print(f"  Selected patches: {len(patches)}")
    print(f"  Coverage: {coverage*100:.2f}%")
    print(f"  Total score: {total_score:,} / {sum_grid.sum():,} ({total_score/sum_grid.sum()*100:.2f}%)")
    
    # Step 5: 토지 이용 정보 로드 및 구성 계산
    print("\nLoading shapefile and creating land use grid...")
    
    # landuse_grid가 없으면 생성
    landuse_grid_path = output_dir / 'landuse_grid.npy'
    if landuse_grid_path.exists():
        landuse_grid = np.load(landuse_grid_path)
        print(f"  Loaded existing landuse_grid from {landuse_grid_path}")
    else:
        print(f"  Creating new landuse_grid...")
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
        np.save(landuse_grid_path, landuse_grid)
        print(f"  Saved landuse_grid to {landuse_grid_path}")
    
    node_compositions = compute_node_composition(landuse_grid, patches, PATCH_SIZE)
    
    # Step 6: 노드 정보 통합 (node_id, lat, lon, composition)
    nodes = create_integrated_nodes(patches, PATCH_SIZE, bounds, GRID_SIZE, node_compositions)
    
    # Step 7: 노드별 시계열 수요 추출
    demands = extract_node_demands(temporal_grid, patches, PATCH_SIZE)
    
    # Step 8: Temporal features 생성 (day, time, holiday)
    temporal_features = create_temporal_features(unique_hours)
    
    # Step 9: JSON 저장
    save_graph_json(
        output_dir / 'graph_data.json',
        nodes=nodes,
        demands=demands,
        temporal_features=temporal_features
    )
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*18 + "GRAPH GENERATION COMPLETE!" + " "*13 + "║")
    print("╚" + "="*58 + "╝\n")
    
    print(f"Output saved to: {output_dir / 'graph_data.json'}")


if __name__ == '__main__':
    main()
