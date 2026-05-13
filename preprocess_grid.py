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
from matplotlib.colors import ListedColormap

try:
    import contextily as ctx
except ImportError:
    ctx = None

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


def bounds_from_total_bounds(total_bounds: np.ndarray | list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = total_bounds
    return min_x, max_x, min_y, max_y


def pad_bounds(bounds: tuple[float, float, float, float], pad_ratio: float = 0.03) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    pad_x = max(width * pad_ratio, GRID_SIZE)
    pad_y = max(height * pad_ratio, GRID_SIZE)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def apply_basemap(ax, bounds: tuple[float, float, float, float], crs: str = TARGET_CRS):
    min_x, max_x, min_y, max_y = pad_bounds(bounds)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    if ctx is not None:
        ctx.add_basemap(
            ax,
            crs=crs,
            source=ctx.providers.CartoDB.Positron,
            attribution=False
        )
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_xlabel(f'X ({crs})')
    ax.set_ylabel(f'Y ({crs})')


def xy_dataframe_to_projected_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    xpos/ypos(소수점 생략 WGS84)를 EPSG:5174 GeoDataFrame으로 변환
    """
    df_geo = df[['xpos', 'ypos']].dropna().copy()
    df_geo['lon'] = df_geo['xpos'] / 1_000_000
    df_geo['lat'] = df_geo['ypos'] / 1_000_000

    gdf = gpd.GeoDataFrame(
        df_geo,
        geometry=gpd.points_from_xy(df_geo['lon'], df_geo['lat']),
        crs='EPSG:4326'
    )
    return gdf.to_crs(TARGET_CRS)


def load_and_preprocess_origin_data(csv_path: Path) -> tuple[gpd.GeoDataFrame, dict[str, gpd.GeoDataFrame]]:
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
    df_region = remove_other_region(df, x_col='xpos', y_col='ypos', 
                                    x_range=US_RANGE_X, y_range=US_RANGE_Y)
    print(f"  After region filter: {len(df_region):,} rows")
    
    # [2] eleminate_duplicates: 중복 제거
    print("\n[3/4] Applying eleminate_duplicates...")
    df_dedup = eleminate_duplicates(df_region, client_col='clientid', 
                                    time_col='call_date', window_seconds=600)
    print(f"  After duplicate removal: {len(df_dedup):,} rows")
    
    # [3] crop_filter: outlier 제거
    print("\n[4/4] Applying crop_filter...")
    df_cropped = crop_filter(df_dedup, threshold_rate=0.85, step_rate=0.001)
    print(f"  After crop filter: {len(df_cropped):,} rows")
    
    # 필요한 컬럼만 선택
    df_points = df_cropped[['xpos', 'ypos']].copy()
    
    print("\n" + "="*60)
    print("Step 2: Converting to GeoDataFrame and CRS transformation")
    print("="*60)
    
    # 정수 좌표를 실제 WGS84 좌표로 변환 (소수점 복원)
    print("\nConverting integer coordinates to WGS84...")
    df_points['lon'] = df_points['xpos'] / 1_000_000
    df_points['lat'] = df_points['ypos'] / 1_000_000
    
    # GeoDataFrame으로 변환 (WGS84)
    print("Creating GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points['lon'], df_points['lat']),
        crs='EPSG:4326'  # WGS84
    )
    
    # 미터 기반 좌표계로 변환 (EPSG:5174)
    print(f"Converting to {TARGET_CRS} (meter-based)...")
    gdf = gdf.to_crs(TARGET_CRS)
    
    print(f"  Final preprocessed data: {len(gdf):,} points")

    preprocess_stages = {
        'step1_raw': xy_dataframe_to_projected_gdf(df[['xpos', 'ypos']]),
        'step1_region_filtered': xy_dataframe_to_projected_gdf(df_region[['xpos', 'ypos']]),
        'step1_deduplicated': xy_dataframe_to_projected_gdf(df_dedup[['xpos', 'ypos']]),
        'step1_cropped': xy_dataframe_to_projected_gdf(df_cropped[['xpos', 'ypos']]),
        'step2_projected': gdf.copy(),
    }

    return gdf, preprocess_stages


def visualize_point_stage(gdf: gpd.GeoDataFrame, output_path: Path, title: str,
                          color: str = '#d7301f', alpha: float = 0.10,
                          markersize: float = 0.5):
    """
    GeoDataFrame 포인트 시각화
    """
    if gdf.empty:
        print(f"  Skipping {output_path.name}: empty GeoDataFrame")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    apply_basemap(ax, bounds_from_total_bounds(gdf.total_bounds), crs=TARGET_CRS)
    gdf.plot(ax=ax, color=color, alpha=alpha, markersize=markersize)
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_landuse_polygons(gdf: gpd.GeoDataFrame, output_path: Path):
    """
    shapefile polygon 결과 시각화
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_gdf = gdf.copy()
    plot_gdf['landuse_code'] = plot_gdf['ATRB_SE'].astype(str).str.extract(r'UQA(\d)', expand=False).fillna('0').astype(int)

    colors = ['#f7f7f7', '#e41a1c', '#ff7f00', '#ffd92f', '#4daf4a', '#377eb8']
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 12))
    apply_basemap(ax, bounds_from_total_bounds(plot_gdf.total_bounds), crs=TARGET_CRS)
    plot_gdf.plot(
        ax=ax,
        column='landuse_code',
        cmap=cmap,
        linewidth=0.15,
        edgecolor='black',
        alpha=0.55,
        legend=True
    )
    ax.set_title('Step 4: Land Use Polygons', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_demand_grid_step(demand_grid: np.ndarray,
                               bounds: tuple[float, float, float, float],
                               output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_x, max_x, min_y, max_y = bounds
    extent = (min_x, max_x, min_y, max_y)
    demand_masked = np.ma.masked_where(demand_grid == 0, demand_grid)

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_basemap(ax, bounds, crs=TARGET_CRS)
    im = ax.imshow(
        demand_masked,
        cmap='hot',
        interpolation='nearest',
        origin='upper',
        extent=extent,
        alpha=0.75
    )
    ax.set_title('Step 3: Taxi Demand Grid (100m x 100m)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Demand Count')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_landuse_grid_step(landuse_grid: np.ndarray,
                                bounds: tuple[float, float, float, float],
                                output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue']
    cmap = ListedColormap(colors)
    min_x, max_x, min_y, max_y = bounds
    extent = (min_x, max_x, min_y, max_y)
    landuse_masked = np.ma.masked_where(landuse_grid == 0, landuse_grid)

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_basemap(ax, bounds, crs=TARGET_CRS)
    im = ax.imshow(
        landuse_masked,
        cmap=cmap,
        interpolation='nearest',
        origin='upper',
        vmin=0,
        vmax=5,
        extent=extent,
        alpha=0.60
    )
    ax.set_title('Step 5: Land Use Grid', fontsize=16, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
    cbar.set_label('Land Use Category')
    cbar.ax.set_yticklabels(['Unclassified', 'UQA1xx', 'UQA2xx', 'UQA3xx', 'UQA4xx', 'UQA5xx'])

    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_combined_grid_step(demand_grid: np.ndarray,
                                 landuse_grid: np.ndarray,
                                 bounds: tuple[float, float, float, float],
                                 output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue']
    cmap = ListedColormap(colors)
    min_x, max_x, min_y, max_y = bounds
    extent = (min_x, max_x, min_y, max_y)
    landuse_masked = np.ma.masked_where(landuse_grid == 0, landuse_grid)
    demand_masked = np.ma.masked_where(demand_grid == 0, demand_grid)

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_basemap(ax, bounds, crs=TARGET_CRS)
    ax.imshow(
        landuse_masked,
        cmap=cmap,
        interpolation='nearest',
        origin='upper',
        vmin=0,
        vmax=5,
        extent=extent,
        alpha=0.45
    )
    im = ax.imshow(
        demand_masked,
        cmap='hot',
        interpolation='nearest',
        origin='upper',
        extent=extent,
        alpha=0.35
    )
    ax.set_title('Step 6: Combined Land Use + Taxi Demand', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Demand Count')
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def visualize_preprocess_steps(preprocess_stages: dict[str, gpd.GeoDataFrame],
                               demand_grid: np.ndarray,
                               gdf_landuse: gpd.GeoDataFrame,
                               landuse_grid: np.ndarray,
                               bounds: tuple[float, float, float, float],
                               output_dir: Path):
    """
    각 step 결과를 개별 파일로 저장
    """
    print("\n" + "="*60)
    print("Saving step-by-step visualizations")
    print("="*60)
    if ctx is None:
        print("  contextily is not installed, basemap will be skipped")

    output_dir.mkdir(parents=True, exist_ok=True)

    point_specs = [
        ('step1_raw', 'Step 1-0: Raw Demand Points', 'step1_0_raw_points.png'),
        ('step1_region_filtered', 'Step 1-1: Region Filtered Points', 'step1_1_region_filtered_points.png'),
        ('step1_deduplicated', 'Step 1-2: Deduplicated Points', 'step1_2_deduplicated_points.png'),
        ('step1_cropped', 'Step 1-3: Cropped Points', 'step1_3_cropped_points.png'),
        ('step2_projected', f'Step 2: Points Projected to {TARGET_CRS}', 'step2_projected_points.png'),
    ]

    for key, title, filename in point_specs:
        visualize_point_stage(preprocess_stages[key], output_dir / filename, title)

    visualize_demand_grid_step(demand_grid, bounds, output_dir / 'step3_demand_grid.png')
    visualize_landuse_polygons(gdf_landuse, output_dir / 'step4_landuse_polygons.png')
    visualize_landuse_grid_step(landuse_grid, bounds, output_dir / 'step5_landuse_grid.png')
    visualize_combined_grid_step(demand_grid, landuse_grid, bounds, output_dir / 'step6_combined_grid.png')


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
                       bounds: tuple[float, float, float, float],
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
    demand_path = output_dir / 'demand_grid.png'
    visualize_demand_grid_step(demand_grid, bounds, demand_path)

    # 2. Land use grid 시각화
    print("\n[2/3] Creating land use grid visualization...")
    landuse_path = output_dir / 'landuse_grid.png'
    visualize_landuse_grid_step(landuse_grid, bounds, landuse_path)

    # 3. Combined visualization (overlay)
    print("\n[3/3] Creating combined visualization...")
    combined_path = output_dir / 'combined_grid.png'
    visualize_combined_grid_step(demand_grid, landuse_grid, bounds, combined_path)
    
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
    step_output_dir = output_dir / 'step_visualizations'
    
    origin_csv = data_dir / 'origin_data.csv'
    shp_file = data_dir / 'UPIS_C_UQ111.shp'
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "DATA PREPROCESSING & GRID GENERATION" + " "*12 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Step 1-2: Load and preprocess origin data
    gdf_preprocessed, preprocess_stages = load_and_preprocess_origin_data(origin_csv)
    
    # Step 3: Create demand grid
    demand_grid, bounds = create_demand_grid(gdf_preprocessed)
    
    # Step 4: Load shapefile
    gdf_landuse = load_shapefile_and_convert_crs(shp_file)
    
    # Step 5: Map land use to grid
    landuse_grid = map_landuse_to_grid(gdf_landuse, demand_grid.shape, bounds)

    # Step별 결과 저장
    visualize_preprocess_steps(
        preprocess_stages=preprocess_stages,
        demand_grid=demand_grid,
        gdf_landuse=gdf_landuse,
        landuse_grid=landuse_grid,
        bounds=bounds,
        output_dir=step_output_dir
    )
    
    # Step 6: Visualize and save
    visualize_and_save(demand_grid, landuse_grid, bounds, output_dir)
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*18 + "PROCESSING COMPLETE!" + " "*19 + "║")
    print("╚" + "="*58 + "╝\n")
    
    print(f"All outputs saved to: {output_dir}")
    print(f"Step visualizations saved to: {step_output_dir}")


if __name__ == '__main__':
    main()
