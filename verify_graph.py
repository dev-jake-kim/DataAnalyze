"""
검증 스크립트: verify_graph.py
1. 생성된 graph_data.json을 로드합니다.
2. 원본 데이터를 다시 로드하여 배경(Demand Grid)을 재구성합니다.
3. 선택된 셀(Node)들이 수요가 높은 곳에 위치하는지 시각화합니다.
4. POI가 해당 그리드 인덱스 범위 내에 실제로 존재하는지 좌표 계산을 통해 검증합니다.
"""

import json
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

from create_graph import GRID_SIZE as GS

# ==========================================
# 설정 (Main 스크립트와 동일해야 함)
# ==========================================
US_RANGE_X = (128_950739, 129_500739)
US_RANGE_Y = (35_756099, 35_305729)
GRID_SIZE = GS
TARGET_CRS = 'EPSG:5174'

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
GRAPH_JSON_PATH = OUTPUT_DIR / 'graph_data.json'
ORIGIN_CSV_PATH = DATA_DIR / 'origin_data.csv'
POI_CSV_PATH = DATA_DIR / 'poi_data.csv'

# 전처리 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / 'control_origin_data'))
try:
    from extract_frame import remove_other_region, eleminate_duplicates, crop_filter
except ImportError:
    print("경고: 'control_origin_data' 폴더를 찾을 수 없습니다. 간단한 전처리만 수행합니다.")

def load_graph_json(path):
    print(f"Loading graph data from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def recreate_demand_grid(csv_path):
    """검증을 위한 배경 그리드 재생성 (Main 로직의 축소판)"""
    print(f"Recreating demand grid from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='cp949')
    
    # 전처리 (Main과 동일하게 적용해야 위치가 맞음)
    if 'remove_other_region' in globals():
        df = remove_other_region(df, x_col='xpos', y_col='ypos', x_range=US_RANGE_X, y_range=US_RANGE_Y)
        df = eleminate_duplicates(df, client_col='clientid', time_col='call_date', window_seconds=600)
        df = crop_filter(df, threshold_rate=0.9, step_rate=0.001)
    
    # 좌표 변환
    df['lon'] = df['xpos'] / 1_000_000
    df['lat'] = df['ypos'] / 1_000_000
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326'
    ).to_crs(TARGET_CRS)

    # 그리드 생성
    min_x, max_x, min_y, max_y = gdf.geometry.x.min(), gdf.geometry.x.max(), gdf.geometry.y.min(), gdf.geometry.y.max()
    
    # Main 스크립트의 bounds 로직과 일치시키기 위해 전체 데이터 기준 bounds 사용 권장
    # 여기서는 데이터 기반으로 다시 계산
    width = max_x - min_x
    height = max_y - min_y
    n_cols = int(np.ceil(width / GRID_SIZE))
    n_rows = int(np.ceil(height / GRID_SIZE))
    
    grid = np.zeros((n_rows, n_cols), dtype=int)
    
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    
    cols = ((x_coords - min_x) / GRID_SIZE).astype(int)
    rows = ((max_y - y_coords) / GRID_SIZE).astype(int)
    
    for r, c in zip(rows, cols):
        if 0 <= r < n_rows and 0 <= c < n_cols:
            grid[r, c] += 1
            
    return grid, (min_x, max_x, min_y, max_y)

def visualize_verification(grid, nodes, bounds, output_path):
    """히트맵 위에 선택된 셀을 빨간색으로 표시"""
    print(f"Generating visualization -> {output_path}")
    
    plt.figure(figsize=(12, 12))
    
    # 배경: Demand Grid
    masked_grid = np.ma.masked_where(grid == 0, grid)
    plt.imshow(masked_grid, cmap='viridis', interpolation='nearest', origin='upper')
    plt.colorbar(label='Original Demand Count')
    
    ax = plt.gca()
    
    total_cells = 0
    
    # Overlay: JSON에서 로드한 셀 좌표
    for node in nodes:
        # node['cells']는 [[row, col], [row, col]...] 형식이어야 함
        cells = node.get('cells', [])
        node_id = node.get('node_id', '?')
        
        for r, c in cells:
            # matplotlib rect는 (x, y) 기준 -> (col, row)
            # 픽셀 좌표계 보정 (0.5씩 이동하여 그리드 칸에 맞춤)
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=1.5,
                edgecolor='red',
                facecolor='none' # 내부는 비움 (배경 보게)
            )
            ax.add_patch(rect)
            total_cells += 1
            
        # 노드 중심에 ID 표시
        # cx = sum(c for r, c in cells) / len(cells)
        # cy = sum(r for r, c in cells) / len(cells)
        # plt.text(cx, cy, str(node_id), color='white', fontsize=8, ha='center', va='center', fontweight='bold')

    plt.title(f"Verification: Selected Cells (Red) vs Raw Demand\nTotal Nodes: {len(nodes)}, Total Cells: {total_cells}")
    plt.xlabel("Grid Column (X)")
    plt.ylabel("Grid Row (Y)")
    
    plt.savefig(output_path, dpi=300)
    print("  Visualization saved.")
    plt.close()

def verify_poi_logic(nodes, poi_path, bounds):
    """JSON의 POI 카운트와 실제 좌표 기반 공간 쿼리 비교"""
    print("\n" + "="*50)
    print("POI MAPPING VERIFICATION")
    print("="*50)
    
    # POI 데이터 로드
    poi_df = pd.read_csv(poi_path)
    poi_df = poi_df.dropna(subset=['좌표정보x(epsg5174)', '좌표정보y(epsg5174)'])
    poi_gdf = gpd.GeoDataFrame(
        poi_df, 
        geometry=gpd.points_from_xy(poi_df['좌표정보x(epsg5174)'], poi_df['좌표정보y(epsg5174)']), 
        crs="EPSG:5174"
    )
    
    min_x, max_x, min_y, max_y = bounds
    
    # 검증할 샘플 노드 선택 (무작위 3개)
    import random
    sample_nodes = random.sample(nodes, min(3, len(nodes)))
    
    mismatch_count = 0
    
    for node in sample_nodes:
        node_id = node['node_id']
        json_poi_counts = node['composition'].get('poi', {})
        json_total_poi = sum(json_poi_counts.values())
        
        cells = node['cells']
        
        # 공간 쿼리를 통한 실제 POI 개수 계산
        real_spatial_poi_count = 0
        
        print(f"\nChecking Node {node_id} (Cells: {len(cells)})")
        
        for r, c in cells:
            # 그리드 인덱스를 실제 좌표 범위로 변환
            # Col(x): min_x + c*SIZE ~ min_x + (c+1)*SIZE
            # Row(y): max_y - (r+1)*SIZE ~ max_y - r*SIZE (Y축 역전 주의)
            
            cell_min_x = min_x + c * GRID_SIZE
            cell_max_x = min_x + (c + 1) * GRID_SIZE
            cell_min_y = max_y - (r + 1) * GRID_SIZE
            cell_max_y = max_y - r * GRID_SIZE
            
            # 공간 쿼리
            spatial_match = poi_gdf[
                (poi_gdf.geometry.x >= cell_min_x) & (poi_gdf.geometry.x < cell_max_x) &
                (poi_gdf.geometry.y >= cell_min_y) & (poi_gdf.geometry.y < cell_max_y)
            ]
            real_spatial_poi_count += len(spatial_match)
            
        print(f"  - JSON POI Count: {json_total_poi}")
        print(f"  - Real Spatial Count: {real_spatial_poi_count}")
        
        if json_total_poi != real_spatial_poi_count:
            print("  >>> [FAIL] Mismatch detected!")
            mismatch_count += 1
        else:
            print("  >>> [PASS] Counts match exactly.")

    if mismatch_count == 0:
        print("\nRESULT: POI Mapping Logic is CONSISTENT.")
    else:
        print(f"\nRESULT: WARNING! Found {mismatch_count} mismatches.")
        print("Possible causes: Coordinate precision issues, boundary conditions (<= vs <), or data updates.")

def main():
    if not GRAPH_JSON_PATH.exists():
        print(f"Error: {GRAPH_JSON_PATH} not found. Run the main script first.")
        return

    # 1. JSON 로드
    graph_data = load_graph_json(GRAPH_JSON_PATH)
    nodes = graph_data['nodes']
    
    # 2. 배경 그리드 재생성 (시간이 좀 걸림)
    grid, bounds = recreate_demand_grid(ORIGIN_CSV_PATH)
    
    # 3. 시각화 검증
    visualize_verification(grid, nodes, bounds, OUTPUT_DIR / 'verify_patches.png')
    
    # 4. POI 데이터 검증
    verify_poi_logic(nodes, POI_CSV_PATH, bounds)

if __name__ == "__main__":
    main()