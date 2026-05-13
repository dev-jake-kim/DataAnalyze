import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.patheffects as pe

# =========================
# 1. 울산 경계 읽기
# =========================
ulsan_path = "./data/UPIS_C_UQ111.shp"
ulsan = gpd.read_file(ulsan_path)

if ulsan.crs is None:
    ulsan = ulsan.set_crs(epsg=4326)

ulsan = ulsan.to_crs(epsg=5179)
ulsan_union = ulsan.union_all() if hasattr(ulsan, "union_all") else ulsan.unary_union

# =========================
# 2. 전체 bounds의 중앙 70%로 crop
# =========================
minx, miny, maxx, maxy = ulsan.total_bounds

crop_ratio = 0.5  # 중앙 70%
cx = (minx + maxx) / 2
cy = (miny + maxy) / 2

full_w = maxx - minx
full_h = maxy - miny

crop_w = full_w * crop_ratio
crop_h = full_h * crop_ratio

crop_minx = cx - crop_w / 2
crop_maxx = cx + crop_w / 2
crop_miny = cy - crop_h / 2
crop_maxy = cy + crop_h / 2

# crop 영역 시각화/격자 생성을 위한 박스
cropped_bbox = box(crop_minx, crop_miny, crop_maxx, crop_maxy)

# =========================
# 3. H x W 격자 생성
# =========================
H = 12
W = 8

cell_width = (crop_maxx - crop_minx) / W
cell_height = (crop_maxy - crop_miny) / H

# =========================
# 3. H x W 격자 생성 (수정)
# =========================
grid_cells = []
labels_info = []

for i in range(H):
    for j in range(W):
        x1 = crop_minx + j * cell_width
        y1 = crop_miny + (H - 1 - i) * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height

        cell = box(x1, y1, x2, y2)

        # 🔥 필터 제거 → 모든 격자 포함
        grid_cells.append({
            "row": i + 1,
            "col": j + 1,
            "geometry": cell
        })

        center = cell.centroid
        labels_info.append({
            "row": i + 1,
            "col": j + 1,
            "x": center.x,
            "y": center.y
        })

grid = gpd.GeoDataFrame(grid_cells, crs=ulsan.crs)

# =========================
# 4. 시각화
# =========================
fig, ax = plt.subplots(figsize=(10, 10))

# 지도
ulsan.plot(
    ax=ax,
    color="white",
    edgecolor="dimgray",
    linewidth=1.0
)

# 격자
grid.plot(
    ax=ax,
    facecolor="skyblue",
    edgecolor="royalblue",
    alpha=0.22,
    linewidth=0.8
)
from shapely.geometry import box

center_i, center_j = 6, 4

radius = 2

i_min = center_i - radius
i_max = center_i + radius
j_min = center_j - radius
j_max = center_j + radius

# grid에서 해당 셀들만 필터
selected_cells = grid[
    (grid["row"] >= i_min) & (grid["row"] <= i_max) &
    (grid["col"] >= j_min) & (grid["col"] <= j_max)
]

# bounding box 계산
minx, miny, maxx, maxy = selected_cells.total_bounds

highlight_box = gpd.GeoDataFrame(
    geometry=[box(minx, miny, maxx, maxy)],
    crs=grid.crs
)

# 시각화
highlight_box.boundary.plot(
    ax=ax,
    color='red',
    linewidth=2.5
)

# 라벨
for item in labels_info:
    i = item["row"]
    j = item["col"]

    is_center = (i == 6 and j == 4)

    ax.text(
        item["x"],
        item["y"],
        rf'$\ell_{{{i},{j}}}$',
        fontsize=28 if is_center else 24,
        fontweight='bold' if is_center else 'normal',
        ha='center',
        va='center',
        color='red' if is_center else 'white',
        path_effects=[
            pe.withStroke(linewidth=2.2, foreground='black')
        ]
    )

# 보기 범위를 crop 영역으로 제한
ax.set_xlim(crop_minx, crop_maxx)
ax.set_ylim(crop_miny, crop_maxy)

# ax.set_title(r'Ulsan Map with 70% Cropped Grid and $\ell_{i,j}$ Labels', fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("ulsan_grid.png", dpi=1200)