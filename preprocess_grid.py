"""
데이터 전처리 스크립트
1. origin_data.csv를 crop하고 100m x 100m grid로 변환
2. shp 파일의 용도(ATRB_SE)를 grid에 매핑
3. 결과를 시각화 및 저장
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pathlib import Path
from tqdm import tqdm

# extract_frame.py의 전처리 함수들 import
sys.path.append(str(Path(__file__).parent / 'control_origin_data'))
from extract_frame import remove_other_region, eleminate_duplicates, crop_filter

# 좌표 범위 설정 (extract_frame.py 참고)
# 원본 데이터는 소수점이 생략된 WGS84 좌표 (예: 127080739 -> 127.080739도)
US_RANGE_X = (128_950739, 129_500739)  # 경도
US_RANGE_Y = (35_756099, 35_305729)  # 위도

GRID_SIZE = 100  # 100m (미터 기반 좌표계에서)

# 울산 지역에 적합한 좌표계: EPSG:5174 (Korean 2000 / Central Belt 2010)
TARGET_CRS = 'EPSG:5174'


def load_and_preprocess_origin_data(csv_path: Path) -> gpd.GeoDataFrame:
    """
    origin_data.csv를 로드하고 전처리 적용
    1. remove_other_region: 지역 범위 밖 데이터 제거
    2. eleminate_duplicates: 중복 제거 (10분 이내 동일 고객)
    3. crop_filter: outlier 제거 (85%까지 축소)
    
    주의: xpos, ypos는 소수점이 생략되어 있음 (예: 127080739 -> 127.080739)
    WGS84 좌표를 미터 기반 좌표계로 변환
    """
    print("="*60)
    print("Step 1: Loading and preprocessing origin_data.csv")
    print("="*60)
    
    # 원본 데이터 로드
    print("\n[1/4] Loading CSV...")
    df = pd.read_csv(csv_path, encoding='cp949')
    print(f"  Initial rows: {len(df):,}")
    
    # 필요한 컬럼 확인 (전처리에 필요한 최소 컬럼)
    required_cols = ['xpos', 'ypos', 'clientid', 'call_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns missing: {required_cols}")
    
    # [1] remove_other_region: 울산 지역 범위로 필터링
    print("\n[2/4] Applying remove_other_region...")
    df = remove_other_region(df, x_col='xpos', y_col='ypos', 
                             x_range=US_RANGE_X, y_range=US_RANGE_Y)
    print(f"  After region filter: {len(df):,} rows")
    
    # [2] eleminate_duplicates: 중복 제거
    print("\n[3/4] Applying eleminate_duplicates...")
    df = eleminate_duplicates(df, client_col='clientid', 
                             time_col='call_date', window_seconds=600)
    print(f"  After duplicate removal: {len(df):,} rows")
    
    # [3] crop_filter: outlier 제거
    print("\n[4/4] Applying crop_filter...")
    df = crop_filter(df, threshold_rate=0.85, step_rate=0.001)
    print(f"  After crop filter: {len(df):,} rows")
    
    # 필요한 컬럼만 선택
    df = df[['xpos', 'ypos']].copy()
    
    print("\n" + "="*60)
    print("Step 2: Converting to GeoDataFrame and CRS transformation")
    print("="*60)
    
    # 정수 좌표를 실제 WGS84 좌표로 변환 (소수점 복원)
    print("\nConverting integer coordinates to WGS84...")
    df['lon'] = df['xpos'] / 1_000_000
    df['lat'] = df['ypos'] / 1_000_000
    
    # GeoDataFrame으로 변환 (WGS84)
    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326'  # WGS84
    )
    
    # 미터 기반 좌표계로 변환 (EPSG:5174)
    print(f"Converting to {TARGET_CRS} (meter-based)...")
    gdf = gdf.to_crs(TARGET_CRS)
    
    print(f"  Final preprocessed data: {len(gdf):,} points")
    
    return gdf


def create_demand_grid(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    택시 수요 데이터를 100m x 100m grid로 변환
    입력 좌표는 미터 기반 좌표계(EPSG:5174)
    Returns: (grid, bounds)
        grid: demand count grid
        bounds: (min_x, max_x, min_y, max_y) in meter-based coordinates
    """
    print("\n" + "="*60)
    print("Step 3: Creating 100m x 100m demand grid")
    print("="*60)
    
    # 미터 좌표 추출
    x_coords = gdf.geometry.x.values
    y_coords = gdf.geometry.y.values
    
    # 범위 계산 (미터 단위)
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    
    # Grid 크기 계산
    width = max_x - min_x
    height = max_y - min_y
    
    n_cols = int(np.ceil(width / GRID_SIZE))
    n_rows = int(np.ceil(height / GRID_SIZE))
    
    print(f"\nGrid configuration:")
    print(f"  Size: {n_rows} rows x {n_cols} columns = {n_rows * n_cols:,} cells")
    print(f"  Width: {width:.0f}m ({width/1000:.1f}km)")
    print(f"  Height: {height:.0f}m ({height/1000:.1f}km)")
    print(f"  Bounds (meter): X=[{min_x:.0f}, {max_x:.0f}], Y=[{min_y:.0f}, {max_y:.0f}]")
    
    # Grid 초기화
    grid = np.zeros((n_rows, n_cols), dtype=np.int32)
    
    # 각 수요를 grid에 매핑 (벡터화)
    print("\nMapping demands to grid cells...")
    cols = ((x_coords - min_x) / GRID_SIZE).astype(int)
    rows = ((max_y - y_coords) / GRID_SIZE).astype(int)  # y는 위에서 아래로
    
    # 범위 체크
    valid_mask = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    rows_valid = rows[valid_mask]
    cols_valid = cols[valid_mask]
    
    # 수요 카운팅
    for r, c in zip(rows_valid, cols_valid):
        grid[r, c] += 1
    
    print(f"\nDemand statistics:")
    print(f"  Total demand count: {grid.sum():,}")
    print(f"  Non-zero cells: {np.count_nonzero(grid):,} / {grid.size:,} ({np.count_nonzero(grid)/grid.size*100:.2f}%)")
    print(f"  Max demand in a cell: {grid.max()}")
    print(f"  Average demand (non-zero): {grid[grid > 0].mean():.2f}")
    
    return grid, (min_x, max_x, min_y, max_y)


def load_shapefile_and_convert_crs(shp_path: Path) -> gpd.GeoDataFrame:
    """
    Shapefile을 로드하고 좌표계 확인
    shp 파일은 EPSG:5174 (미터 기반 좌표계)
    """
    print("\n" + "="*60)
    print("Step 4: Loading shapefile for land use classification")
    print("="*60)
    
    gdf = gpd.read_file(shp_path)
    print(f"\nShapefile information:")
    print(f"  CRS: {gdf.crs}")
    print(f"  Total polygons: {len(gdf)}")
    print(f"  ATRB_SE unique values: {sorted(gdf['ATRB_SE'].unique())}")
    
    # 이미 EPSG:5174이면 그대로 사용, 아니면 변환
    if gdf.crs != TARGET_CRS:
        print(f"\nConverting to {TARGET_CRS}...")
        gdf = gdf.to_crs(TARGET_CRS)
    else:
        print(f"\nAlready in {TARGET_CRS} ✓")
    
    return gdf


def map_landuse_to_grid(gdf: gpd.GeoDataFrame, grid_shape: tuple[int, int], 
                        bounds: tuple[float, float, float, float]) -> np.ndarray:
    """
    Grid의 각 셀에 해당하는 용도(ATRB_SE)를 매핑
    UQA1xx -> 1, UQA2xx -> 2, UQA3xx -> 3, UQA4xx -> 4, UQA5xx -> 5
    좌표는 미터 기반 (EPSG:5174)
    """
    print("\n" + "="*60)
    print("Step 5: Mapping land use to grid cells")
    print("="*60)
    
    min_x, max_x, min_y, max_y = bounds
    n_rows, n_cols = grid_shape
    
    # Land use grid 초기화 (0: 미분류)
    landuse_grid = np.zeros((n_rows, n_cols), dtype=np.int8)
    
    # ATRB_SE를 숫자로 변환하는 함수
    def atrb_to_num(atrb: str) -> int:
        """UQA1xx -> 1, UQA2xx -> 2, ..."""
        if atrb.startswith('UQA'):
            try:
                return int(atrb[3])  # UQA[1]xx
            except (ValueError, IndexError):
                return 0
        return 0
    
    # spatial index 생성으로 성능 개선
    print("\nBuilding spatial index...")
    sindex = gdf.sindex
    
    # 각 grid cell의 중심점을 생성하고 해당하는 polygon 찾기
    total_cells = n_rows * n_cols
    print(f"Mapping {total_cells:,} cells to land use categories...")
    
    with tqdm(total=total_cells, desc="Mapping cells") as pbar:
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                # Cell 중심점 계산 (미터 좌표)
                center_x = min_x + (col_idx + 0.5) * GRID_SIZE
                center_y = max_y - (row_idx + 0.5) * GRID_SIZE
                
                # Point 생성
                point = Point(center_x, center_y)
                
                # spatial index를 사용해 후보 polygon 찾기
                possible_matches_idx = list(sindex.intersection(point.bounds))
                if possible_matches_idx:
                    possible_matches = gdf.iloc[possible_matches_idx]
                    # 실제로 포함하는지 확인
                    containing = possible_matches[possible_matches.contains(point)]
                    
                    if len(containing) > 0:
                        # 첫 번째 매칭되는 polygon의 ATRB_SE 사용
                        atrb = containing.iloc[0]['ATRB_SE']
                        landuse_grid[row_idx, col_idx] = atrb_to_num(atrb)
                
                pbar.update(1)
    
    print(f"\nLand use mapping statistics:")
    print(f"  Classified cells: {np.count_nonzero(landuse_grid):,} / {landuse_grid.size:,} ({np.count_nonzero(landuse_grid)/landuse_grid.size*100:.2f}%)")
    print(f"  Category breakdown:")
    
    # 각 카테고리별 통계
    for i in range(1, 6):
        count = np.sum(landuse_grid == i)
        if count > 0:
            print(f"    Category {i} (UQA{i}xx): {count:,} cells ({count/landuse_grid.size*100:.2f}%)")
    
    return landuse_grid


def visualize_and_save(demand_grid: np.ndarray, landuse_grid: np.ndarray, 
                       output_dir: Path):
    """
    Grid를 시각화하고 저장
    """
    print("\n" + "="*60)
    print("Step 6: Visualization and saving results")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Demand grid 시각화
    print("\n[1/3] Creating demand grid visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(demand_grid, cmap='hot', interpolation='nearest')
    ax.set_title('Taxi Demand Grid (100m x 100m)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    plt.colorbar(im, ax=ax, label='Demand Count')
    
    demand_path = output_dir / 'demand_grid.png'
    plt.savefig(demand_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {demand_path}")
    plt.close()
    
    # 2. Land use grid 시각화
    print("\n[2/3] Creating land use grid visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 색상 맵: 0=white, 1=red, 2=orange, 3=yellow, 4=green, 5=blue
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue']
    cmap = ListedColormap(colors)
    
    im = ax.imshow(landuse_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
    ax.set_title('Land Use Grid (from ATRB_SE)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    
    # Colorbar with labels
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
    cbar.set_label('Land Use Category')
    cbar.ax.set_yticklabels(['Unclassified', 'UQA1xx', 'UQA2xx', 'UQA3xx', 'UQA4xx', 'UQA5xx'])
    
    landuse_path = output_dir / 'landuse_grid.png'
    plt.savefig(landuse_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {landuse_path}")
    plt.close()
    
    # 3. Combined visualization (overlay)
    print("\n[3/3] Creating combined visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Land use as base
    ax.imshow(landuse_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=5, alpha=0.6)
    
    # Demand as overlay (only show non-zero)
    demand_masked = np.ma.masked_where(demand_grid == 0, demand_grid)
    im = ax.imshow(demand_masked, cmap='hot', interpolation='nearest', alpha=0.4)
    
    ax.set_title('Combined: Land Use + Taxi Demand', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Grid Index')
    ax.set_ylabel('Y Grid Index')
    plt.colorbar(im, ax=ax, label='Demand Count')
    
    combined_path = output_dir / 'combined_grid.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {combined_path}")
    plt.close()
    
    # 4. Save grids as numpy arrays
    print("\nSaving numpy arrays...")
    np.save(output_dir / 'demand_grid.npy', demand_grid)
    np.save(output_dir / 'landuse_grid.npy', landuse_grid)
    print(f"  Saved: {output_dir / 'demand_grid.npy'}")
    print(f"  Saved: {output_dir / 'landuse_grid.npy'}")


def main():
    # 경로 설정
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'output'
    
    origin_csv = data_dir / 'origin_data.csv'
    shp_file = data_dir / 'UPIS_C_UQ111.shp'
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "DATA PREPROCESSING & GRID GENERATION" + " "*12 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Step 1-2: Load and preprocess origin data
    gdf_preprocessed = load_and_preprocess_origin_data(origin_csv)
    
    # Step 3: Create demand grid
    demand_grid, bounds = create_demand_grid(gdf_preprocessed)
    
    # Step 4: Load shapefile
    gdf_landuse = load_shapefile_and_convert_crs(shp_file)
    
    # Step 5: Map land use to grid
    landuse_grid = map_landuse_to_grid(gdf_landuse, demand_grid.shape, bounds)
    
    # Step 6: Visualize and save
    visualize_and_save(demand_grid, landuse_grid, output_dir)
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*18 + "PROCESSING COMPLETE!" + " "*19 + "║")
    print("╚" + "="*58 + "╝\n")
    
    print(f"All outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
