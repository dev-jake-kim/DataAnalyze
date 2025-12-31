# python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 데이터 로드
df = pd.read_csv(Path('data') / 'origin_data.csv', encoding='cp949')

# local_idx를 숫자로 변환(비숫자는 NaN)
df['local_idx'] = pd.to_numeric(df['local_idx'], errors='coerce')

# 유효한(local_idx가 있는) 포인트와 결측 포인트 분리
mask_valid = df['local_idx'].notna()
mask_nan = ~mask_valid

# 카테고리화(원래 값의 순서/레이블을 보존)
cats = pd.Categorical(df.loc[mask_valid, 'local_idx'])
codes = cats.codes  # 0 .. n-1
n = len(cats.categories)

# 컬러맵 선택: 범주가 많으면 연속형/광범위 cmap 사용
cmap_name = 'tab20' if n <= 20 else 'nipy_spectral'
cmap = plt.get_cmap(cmap_name, n)

# 경계 정규화(BoundaryNorm을 사용해 정수 코드별로 색을 고정)
norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n + 0.5, 1), ncolors=n)

plt.figure(figsize=(8, 6))

# 유효한 포인트 산점도 (색은 codes 사용)
sc = plt.scatter(
    df.loc[mask_valid, 'xpos'],
    df.loc[mask_valid, 'ypos'],
    c=codes,
    cmap=cmap,
    norm=norm,
    s=20,
    alpha=0.8,
    edgecolors='none'
)

# 결측(local_idx 없는) 포인트는 검정으로 표시
if mask_nan.any():
    plt.scatter(
        df.loc[mask_nan, 'xpos'],
        df.loc[mask_nan, 'ypos'],
        c='black',
        s=20,
        alpha=0.8,
        label='missing local_idx'
    )

# 컬러바 생성: scatter에서 받은 mappable(sc)를 전달
cbar = plt.colorbar(sc, ticks=np.arange(0, n))
cbar.ax.set_yticklabels([str(v) for v in cats.categories])
cbar.set_label('local_idx')

plt.tight_layout()
plt.show()