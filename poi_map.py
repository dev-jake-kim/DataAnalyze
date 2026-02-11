import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box
import sys
from pathlib import Path

# ==========================================
# 1. 설정 및 경로 (verify_graph.py와 동일하게 유지)
# ==========================================
# 경로 설정 (사용자 환경에 맞게 수정 필요)
BASE_DIR = Path('.') # 현재 경로 기준
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

# 파일 경로
GRAPH_JSON_PATH = OUTPUT_DIR / 'graph_data.json'
ORIGIN_CSV_PATH = DATA_DIR / 'origin_data.csv'
POI_CSV_PATH = DATA_DIR / 'poi_data.csv' # POI 파일 경로

# 좌표계 및 그리드 설정
TARGET_CRS = 'EPSG:5174'
GRID_SIZE = 500  # create_graph.py의 GS 값과 동일해야 함 (예: 500m)
US_RANGE_X = (128_950739, 129_500739)
US_RANGE_Y = (35_756099, 35_305729)

# 한글 폰트 설정
plt.rc('font', family='NanumGothic') 
plt.rcParams['axes.unicode_minus'] = False 

# 전처리 모듈 (없으면 간단 처리)
sys.path.append(str(BASE_DIR / 'control_origin_data'))

# ==========================================
# 2. 핵심 함수 정의
# ==========================================

def get_grid_bounds(origin_csv_path):
    """
    origin_data.csv를 읽어 그리드 생성의 기준이 되는 min_x, max_y를 계산합니다.
    이 과정이 없으면 JSON의 행/열 인덱스를 실제 좌표로 바꿀 수 없습니다.
    """
    print("1. 그리드 기준점 계산 중 (Origin Data 로드)...")
    try:
        df = pd.read_csv(origin_csv_path, encoding='cp949')
        
        # 좌표 변환 (기존 로직 준수)
        df['lon'] = df['xpos'] / 1_000_000
        df['lat'] = df['ypos'] / 1_000_000
        
        # GeoDataFrame 변환 및 투영
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326'
        ).to_crs(TARGET_CRS)
        
        min_x = gdf.geometry.x.min()
        max_y = gdf.geometry.y.max()
        
        return min_x, max_y
    except Exception as e:
        print(f"[오류] Origin Data 처리 중 실패: {e}")
        sys.exit(1)

def create_active_area_mask(json_path, min_x, max_y, grid_size):
    """
    JSON 파일의 node/cells 정보를 읽어 실제 지도상의 Polygon(사각형)들로 변환합니다.
    """
    print("2. 활성 그리드 마스크 생성 중...")
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    polygons = []
    
    for node in graph_data['nodes']:
        cells = node.get('cells', [])
        for r, c in cells:
            # 그리드 인덱스 -> 실제 좌표 변환 공식 (verify_graph.py 로직 역산)
            # Col(x): min_x + c * SIZE
            # Row(y): max_y - r * SIZE (위에서 아래로 내려옴)
            
            cell_min_x = min_x + c * grid_size
            cell_max_x = cell_min_x + grid_size
            
            cell_max_y = max_y - r * grid_size
            cell_min_y = cell_max_y - grid_size
            
            # 사각형 생성
            rect = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)
            polygons.append(rect)
            
    # 모든 사각형을 담은 GeoDataFrame 생성
    mask_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=TARGET_CRS)
    
    # (선택) 겹치는 부분을 합쳐서 하나의 큰 영역으로 만들면 연산이 더 빠를 수 있음
    # 여기서는 개별 셀 유지가 시각적으로 유리할 수 있어 그대로 둠
    return mask_gdf

# ==========================================
# 3. 메인 실행 로직
# ==========================================

# (1) 그리드 기준점 확보
min_x, max_y = get_grid_bounds(ORIGIN_CSV_PATH)

# (2) 수요가 있는 영역(Active Area)을 GeoDataFrame으로 변환
active_area_gdf = create_active_area_mask(GRAPH_JSON_PATH, min_x, max_y, GRID_SIZE)
print(f"   -> 총 {len(active_area_gdf)}개의 활성 셀이 로드되었습니다.")

# (3) POI 데이터 로드 및 GeoDataFrame 변환
print("3. POI 데이터 로드 및 필터링...")
poi_df = pd.read_csv(POI_CSV_PATH)
poi_df = poi_df.dropna(subset=['좌표정보x(epsg5174)', '좌표정보y(epsg5174)'])

poi_gdf = gpd.GeoDataFrame(
    poi_df,
    geometry=gpd.points_from_xy(poi_df['좌표정보x(epsg5174)'], poi_df['좌표정보y(epsg5174)']),
    crs=TARGET_CRS
)

# (4) [핵심] 공간 결합 (Spatial Join): POI가 활성 영역 '안에(within)' 있는 경우만 남김
# how='inner' -> 교집합만 남김 (즉, 영역 밖 POI는 삭제됨)
filtered_poi_gdf = gpd.sjoin(poi_gdf, active_area_gdf, how='inner', predicate='within')

print(f"   -> 전체 POI: {len(poi_gdf)}개")
print(f"   -> 필터링된 POI(수요 영역 내): {len(filtered_poi_gdf)}개")

# (5) 시각화
print("4. 지도 생성 중...")
fig, ax = plt.subplots(figsize=(12, 12))

# 좌표계 변환 (Web Mercator)
filtered_web = filtered_poi_gdf.to_crs(epsg=3857)
active_area_web = active_area_gdf.to_crs(epsg=3857)

# A. 수요 그리드 영역 표시 (연한 회색 배경으로 표시하여 범위 확인)
active_area_web.plot(ax=ax, color='blue', alpha=0.1, edgecolor='blue', linewidth=0.3, label='수요 영역')

# B. 필터링된 POI 찍기
filtered_web.plot(
    column='개방서비스아이디',
    ax=ax,
    legend=True,
    cmap='tab10',
    markersize=15,
    alpha=0.9,
    legend_kwds={'bbox_to_anchor': (1.05, 1), 'loc': 'upper left'}
)

# C. 배경 지도 추가
try:
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
except Exception:
    pass

ax.set_axis_off()
plt.title(f'수요 발생 영역 내 POI 분포 (총 {len(filtered_poi_gdf)}개)', fontsize=15)

plt.tight_layout()
plt.savefig('filtered_poi_in_demand_grid.png', dpi=300, bbox_inches='tight')
print("완료: 'filtered_poi_in_demand_grid.png' 저장됨.")