import geopandas as gpd
import matplotlib.pyplot as plt

# 1. 파일 경로 설정 (사용자님의 파일명으로 변경하세요)
file_path = "data/UPIS_C_UQ111.shp" 

# 2. 파일 읽기 (한글 깨짐 방지를 위해 encoding 설정 필수)
try:
    gdf = gpd.read_file(file_path, encoding='euc-kr')
except:
    gdf = gpd.read_file(file_path, encoding='cp949') # euc-kr이 안 되면 cp949 시도

# 3. 데이터 전처리: 암호 같은 코드를 '한글'로 변환
# UQA1xx: 주거, UQA2xx: 상업, UQA3xx: 공업, UQA4xx: 녹지
zoning_map = {
    'UQA1': 'housing',
    'UQA2': 'commercial',
    'UQA3': 'industrial',
    'UQA4': 'green_space'
}

# 'ATRB_SE' 컬럼의 앞 4글자만 잘라서 매핑 (값이 없는 경우 '기타'로 처리)
if 'ATRB_SE' in gdf.columns:
    gdf['land_use'] = gdf['ATRB_SE'].str[:4].map(zoning_map).fillna('other')
else:
    print("경고: 'ATRB_SE' 컬럼이 없습니다. 컬럼명을 확인하세요.")
    print(gdf.columns)
    # 만약 컬럼명이 다르면 여기서 멈추도록 처리하거나 다른 컬럼을 사용해야 합니다.

# 4. 시각화 (지도 그리기)
# figsize: 그림 크기 (가로, 세로)
fig, ax = plt.subplots(figsize=(12, 12))

# column='land_use': 이 컬럼을 기준으로 색깔을 다르게 칠함
# legend=True: 범례 표시
# cmap='Set2': 색상 테마 (Pastel1, Set1, viridis 등 변경 가능)
gdf.plot(column='land_use', 
         ax=ax, 
         legend=True, 
         cmap='Set1', 
         edgecolor='black', 
         linewidth=0.1)

# 5. 그래프 꾸미기
plt.title('purpose of land use', fontsize=15)
plt.axis('off') # x, y축 눈금 제거 (깔끔하게 보기 위해)

# 6. 화면에 출력 (또는 파일로 저장)
plt.savefig("ulsan_map_check.png", dpi=600)  # 고해상도 파일로 저장
# plt.savefig("ulsan_map_check.png") # 파일로 저장하려면 주석 해제