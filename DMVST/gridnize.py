import pandas as pd
from pathlib import Path
import numpy as np
import math

def dtw(grid, pos_i, pos_j):
    time_seq = grid.shape[0]
    i = grid[:, pos_i].reshape((time_seq, 1))
    j = grid[:, pos_j].reshape((time_seq, 1))
    #print(i.shape, j.shape)
    dtw_matrix = abs(i - j.T)
    #print(dtw_matrix.shape)

    for idx in range(1, dtw_matrix.shape[0]):
        dtw_matrix[idx, 0] += dtw_matrix[idx-1, 0]

    for jdx in range(1, dtw_matrix.shape[1]):
        dtw_matrix[0, jdx] += dtw_matrix[0, jdx-1]

    for idx in range(1, dtw_matrix.shape[0]):
        for jdx in range(1, dtw_matrix.shape[1]):
            dtw_matrix[idx, jdx] += min(dtw_matrix[idx-1, jdx], dtw_matrix[idx, jdx-1], dtw_matrix[idx-1, jdx-1])
    return dtw_matrix[-1, -1]


GRID_SIZE= 9

if __name__ == '__main__':
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / 'data'
    img_dir = base_dir.parent / 'imgs'

    # 1) 데이터 로드
    csv_path = data_dir / 'extraction.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    for col in ['xpos', 'ypos', 'call_date']:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼 누락: {col}")
    print("데이터 로드 완료")

    file_name = f'grid{df["call_date"].min()}-{df["call_date"].max()}.npy'

    df['call_date'] = pd.to_datetime(df['call_date'])

    min_dt = df['call_date'].min()

    df['time_idx'] = ((df['call_date'] - min_dt) / pd.Timedelta(hours=1)).astype(int)
    max_time_idx = df['time_idx'].max()
    print(f"최소 시간 인덱스: {df['time_idx'].min()}, 최대 시간 인덱스: {max_time_idx}")

    grid = np.zeros((max_time_idx+1, GRID_SIZE, GRID_SIZE), dtype=np.int32)
    print(f"그리드 크기: {grid.shape}")

    x_min, x_max = df['xpos'].min(), df['xpos'].max()
    y_min, y_max = df['ypos'].min(), df['ypos'].max()
    print(f"x 범위: {x_min} ~ {x_max}, y 범위: {y_min} ~ {y_max}")

    x_grid_size = math.ceil((x_max - x_min) / grid.shape[1])
    y_grid_size = math.ceil((y_max - y_min) / grid.shape[2])
    print(f"x 그리드 크기: {x_grid_size}, y 그리드 크기: {y_grid_size}")
    for _, row in df.iterrows():
        # print(row['time_idx'], row['ypos'] // y_grid_size, row['xpos'] // x_grid_size)
        grid[
            row['time_idx'],
            (row['ypos']- y_min) // y_grid_size,
            (row['xpos'] - x_min) // x_grid_size
        ] += 1

    np.save(data_dir / file_name, grid)

    # python
    # grid에 저장된 각 값의 개수 출력 및 CSV로 저장
    unique_vals, counts = np.unique(grid, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        print(f"value {val}: {cnt}")
    print('mean: ', grid.mean())
    print('std: ', grid.std())
    print('sum: ', grid.sum())

    weight_matrix = np.zeros((GRID_SIZE*GRID_SIZE, GRID_SIZE*GRID_SIZE))

    grid = grid.reshape((grid.shape[0], -1))
    for i in range(grid.shape[1]):
        for j in range(grid.shape[1]):
            weight_matrix[i, j] = dtw(grid, i, j)
            print(f"DTW between {i} and {j}: {weight_matrix[i, j]}")

    np.save(data_dir / 'weight_matrix.npy', weight_matrix)
