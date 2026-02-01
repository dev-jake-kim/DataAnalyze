import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'
    output_dir = base_dir.parent / 'output'
    
    gdf = gpd.read_file(data_dir / "UPIS_C_UQ111.shp", encoding='euc-kr')
    gdf = gdf.to_crs(epsg=4326)