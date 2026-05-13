from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

try:
    import contextily as ctx
except ImportError:
    ctx = None

sys.path.append(str(Path(__file__).resolve().parent.parent / 'control_origin_data'))
from extract_frame import remove_other_region


US_RANGE_X = (128_950739, 129_500739)
US_RANGE_Y = (35_756099, 35_305729)
TARGET_CRS = 'EPSG:3857'


def load_all_points(csv_path: Path, apply_region_filter: bool = True) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, encoding='cp949', usecols=['xpos', 'ypos'])

    # if apply_region_filter:
    #     df = remove_other_region(
    #         df,
    #         x_col='xpos',
    #         y_col='ypos',
    #         x_range=US_RANGE_X,
    #         y_range=US_RANGE_Y
    #     )

    # df = df.dropna(subset=['xpos', 'ypos']).copy()
    df['lon'] = df['xpos'] / 1_000_000
    df['lat'] = df['ypos'] / 1_000_000

    gdf = gpd.GeoDataFrame(
        df[['xpos', 'ypos']],
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326'
    )
    return gdf.to_crs(TARGET_CRS)


def plot_all_points_with_basemap(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    point_size: float = 0.6,
    alpha: float = 0.08,
):
    if gdf.empty:
        raise ValueError('No points available to plot')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_x, min_y, max_x, max_y = gdf.total_bounds
    pad_x = (max_x - min_x) * 0.03
    pad_y = (max_y - min_y) * 0.03

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    if ctx is not None:
        ctx.add_basemap(
            ax,
            crs=gdf.crs,
            source=ctx.providers.CartoDB.Positron,
            attribution=False
        )

    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        s=point_size,
        c='#d7301f',
        alpha=alpha,
        linewidths=0,
        rasterized=True
    )

    ax.set_title('All Demand Points')
    ax.set_xlabel('X (EPSG:3857)')
    ax.set_ylabel('Y (EPSG:3857)')

    fig.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved map to {output_path}')
    print(f'Total plotted points: {len(gdf):,}')
    if ctx is None:
        print('contextily is not installed, basemap was skipped')


def main():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / 'data' / 'origin_data.csv'
    output_path = base_dir / 'output' / 'all_data_points_map.png'

    gdf = load_all_points(csv_path)
    plot_all_points_with_basemap(gdf, output_path)


if __name__ == '__main__':
    main()
