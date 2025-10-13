import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
from math import sqrt
from typing import List, Tuple, Dict, Any

R = 1_000
MAX_CONNECT_DIST = 5_000
MIN_DATA = 100

class OutJson:
    def __init__(self, minHour, maxHour, self_loop_w=1.0, total_nodes=0, edges=None, demands=None):
        self.minHour = minHour
        self.maxHour = maxHour
        self.self_loop_w = self_loop_w
        self.total_nodes = total_nodes
        self.edges = edges if edges is not None else []
        self.demands = demands if demands is not None else []

    def to_dict(self) -> Dict[str, Any]:
        # hours 리스트 생성
        total_hours = int((self.maxHour - self.minHour).total_seconds() // 3600) + 1
        hours = []
        for h in range(total_hours):
            hour = self.minHour + pd.Timedelta(hours=h)
            hours.append(hour.strftime("%Y-%m-%dT%H:%M:%S"))
        return {
            "meta": {
                "hours": hours,
                "self_loop_w": self.self_loop_w,
                "edge_weight": "geom_mean"
            },
            "nodes": [{"id": i} for i in range(self.total_nodes)],
            "edges": [
                {"u": int(e[0]), "v": int(e[1]), "w": float(e[2])}
                for e in self.edges
            ],
            "x": self.demands
        }

    def save_json(self, path: Path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                separators=(',', ':')  # 공백 제거(콤마/콜론 뒤 공백 없음)
            )

def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """2차원 점 a와 b(형상 (2,)) 사이의 유클리드 거리를 반환합니다."""
    return float(sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

def load_ptrs(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, encoding='cp949')
    df['ptr_id'] = df.index
    ptrs = df[['ptr_id','ptr_x', 'ptr_y']].to_numpy(dtype=np.int64)
    return ptrs

def load_datas(csv_path: Path) -> Tuple[np.ndarray, pd.Series]:
    df = pd.read_csv(csv_path, encoding='cp949')
    #data (idx, xpos, ypos)
    df['idx'] = df.index
    data = df[['idx', 'xpos', 'ypos']].to_numpy(dtype=np.int64)
    df['time'] = df['call_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H'))
    time = df[['idx', 'time']]
    return data, time

if __name__ == "__main__":
    data_dir = Path('data')
    essential_csv_path = data_dir / 'essential_columns.csv'
    ptrs_csv_path = data_dir / 'ptrs_summary.csv'

    #ptr 불러오기
    ptrs = load_ptrs(ptrs_csv_path)
    print('Loaded ptrs:', len(ptrs))

    datas, times = load_datas(essential_csv_path)
    print('Loaded datas:', len(datas))

    max_xpos = int(np.max(datas[:,1]))
    max_ypos = int(np.max(datas[:,2]))
    print(f"Max xpos: {max_xpos}, Max ypos: {max_ypos}")
    min_time = times['time'].min()
    max_time = times['time'].max()
    print(f"Data from {min_time} to {max_time}")

    #각 ptr에 할당된 data들 삽입
    ptr_datas = [[] for _ in range(len(ptrs))]
    for data in tqdm(datas, desc="Assigning data to ptrs", unit="data"):
        min_dist = float('inf')
        min_ptr = 0
        for ptr in ptrs:
            dist = distance((data[1], data[2]), (ptr[1], ptr[2]))
            if dist < min_dist:
                min_dist = dist
                min_ptr = ptr[0]
        if min_dist <= R:
            ptr_datas[min_ptr].append(data[0]) #ptr_id에 data idx 삽입

    #할당된 data가 MIN_DATA개 미만인 ptr 제거
    valid_ptrs = []
    for ptr in ptrs:
        idx, x, y = ptr
        if len(ptr_datas[idx]) >= MIN_DATA:
            valid_ptrs.append(ptr)
    ptrs = np.array(valid_ptrs, dtype=np.int64)

    print('After filtering, valid ptrs:', len(ptrs), 'sample:', ptrs[100] if len(ptrs) > 100 else ptrs[0])

    #valid ptrs에 맞춰 ptr_datas 재구성
    tmp_ptr_datas = []
    for valid_ptr in reversed(valid_ptrs):
        idx = valid_ptr[0]
        tmp_ptr_datas.append(ptr_datas[idx])
    ptr_datas = list(reversed(tmp_ptr_datas))
    print('len(ptr_datas):', len(ptr_datas))


    adjacency_matrix = np.full((len(valid_ptrs), len(valid_ptrs)), -1.0, dtype=float)
    for i in tqdm(range(len(valid_ptrs)), desc="Calculating adjacency matrix"):
        for j in range(len(valid_ptrs)):
            if i == j:
                continue
            else:
                dist = distance((valid_ptrs[i][1], valid_ptrs[i][2]), (valid_ptrs[j][1], valid_ptrs[j][2]))
                if dist <= MAX_CONNECT_DIST:
                    adjacency_matrix[i, j] = dist


    edges = []
    max_weight = adjacency_matrix.max()
    for i in tqdm(range(len(valid_ptrs)), desc="Creating edges list"):
        for j in range(len(valid_ptrs)):
            if adjacency_matrix[i, j] != -1.0:
                edges.append([int(i), int(j), float(adjacency_matrix[i, j]) / max_weight])
    print('Total edges:', len(edges))

    #각 ptr에 할당된 data들의 시간 정보를 dense하게 만듦
    idx_to_time = times.set_index("idx")["time"]

    # 총 시간 수(시간 단위)
    total_hours = int((max_time - min_time).total_seconds() // 3600) + 1

    # \`demands\`: 시간(granularity 1시간) × 유효 ptr 개수
    demands = [[0] * len(valid_ptrs) for _ in range(total_hours)]

    # 각 ptr에 할당된 data의 시간을 시간 인덱스로 변환해 카운트
    for ptr_idx, data_indices in enumerate(ptr_datas):
        for data_idx in data_indices:
            t = idx_to_time.at[data_idx]
            hour_idx = int((t - min_time).total_seconds() // 3600)
            if 0 <= hour_idx < total_hours:
                demands[hour_idx][ptr_idx] += 1


    outJson = OutJson(minHour=min_time, maxHour=max_time, self_loop_w=1.0, total_nodes=len(valid_ptrs), edges=edges, demands=demands)
    outJson.save_json(data_dir / 'output.json')
    print('Saved output.json')