# python
from __future__ import annotations

from pathlib import Path
from collections import deque, defaultdict
import pandas as pd
from tqdm import tqdm
from typing import Optional
import matplotlib.pyplot as plt

US_RANGE_X = (128_950739, 129500739) #550,000 ->1000 = 100m
US_RANGE_Y = (35_756099, 35305729) # 450,370 -> 1000 = 100m


# ---------- I/O ----------
def load_csv(csv_path: Path, encoding: str = 'cp949') -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding=encoding)


def save_csv(df: pd.DataFrame, csv_path: Path, encoding: str = 'cp949') -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding=encoding)


# ---------- Filters (all pure) ----------
def remove_other_region(df: pd.DataFrame,
                        x_col: str = 'xpos', y_col: str = 'ypos',
                        x_range: tuple[int, int] = US_RANGE_X,
                        y_range: tuple[int, int] = US_RANGE_Y) -> pd.DataFrame:
    """
    US_RANGE 밖의 레코드를 제거합니다.
    참고: data_transe.proceed_first의 범위 로직을 그대로 사용합니다.
    - x: x_range[0] <= x <= x_range[1]
    - y: y_range[1] <= y <= y_range[0]  (주의: y_range는 상한이 앞에 옴)
    """
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"컬럼 없음: {c}")

    with tqdm(total=3, desc='remove_other_region', unit='step') as pbar:
        mask = (
            (df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1]) &
            (df[y_col] >= y_range[1]) & (df[y_col] <= y_range[0])
        )
        pbar.update(1)
        out = df.loc[mask].copy()
        pbar.update(1)
        # 진행 정보 부가 출력
        tqdm.write(f"remove_other_region: kept {len(out)} / {len(df)} rows, remain {len(out)/len(df)*100:.2f}% data")
        pbar.update(1)
    return out


def drop_nonessential_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only essential columns and shift xpos/ypos to min-origin.
    """
    essential = ['clientid', 'call_date', 'xpos', 'ypos']
    missing = [c for c in essential if c not in df.columns]
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}")

    # 단계 기반 진행률(벡터화 유지)
    with tqdm(total=3, desc='drop_nonessential_columns', unit='step') as pbar:
        out = df[essential].copy()
        pbar.update(1)

        min_x = out['xpos'].min()
        min_y = out['ypos'].min()
        pbar.update(1)

        out['xpos'] = out['xpos'] - min_x
        out['ypos'] = out['ypos'] - min_y
        pbar.update(1)

    return out


def change_time_format(df: pd.DataFrame, time_col: str = 'call_date',
                       out_fmt: str = '%Y-%m-%d %H') -> pd.DataFrame:
    """
    Convert time column to hour-level string.
    Works for both microsecond and hour-only inputs.
    """
    if time_col not in df.columns:
        raise KeyError(f"컬럼 없음: {time_col}")

    out = df.copy()
    with tqdm(total=2, desc='change_time_format', unit='step') as pbar:
        dt = pd.to_datetime(out[time_col], errors='coerce')  # robust parse
        pbar.update(1)
        out[time_col] = dt.dt.strftime(out_fmt)
        pbar.update(1)
    return out


def eleminate_duplicates(df: pd.DataFrame,
                         client_col: str = 'clientid',
                         time_col: str = 'call_date',
                         window_seconds: int = 600) -> pd.DataFrame:
    """
    Remove rows where the same client appears within 10 minutes of a previous appearance.
    Replicates the original queue-based logic without using global df.
    """
    for col in (client_col, time_col):
        if col not in df.columns:
            raise KeyError(f"컬럼 없음: {col}")

    out_idx_to_drop = []
    q: deque[tuple[object, Optional[pd.Timestamp]]] = deque()
    in_window_count = defaultdict(int)

    # parse per-row timestamps robustly
    def parse_ts(val) -> Optional[pd.Timestamp]:
        ts = pd.to_datetime(val, errors='coerce')
        if pd.isna(ts):
            return None
        return ts

    # Keep original order to match prior behavior
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='eleminate_duplicates', unit='row'):
        client = row[client_col]
        ts = parse_ts(row[time_col])

        # Evict entries older than window
        if ts is not None:
            while q and (q[0][1] is not None) and (ts - q[0][1]).total_seconds() > window_seconds:
                old_client, _ = q.popleft()
                in_window_count[old_client] -= 1
                if in_window_count[old_client] <= 0:
                    del in_window_count[old_client]

        # If any same client exists within window, mark duplicate
        if (ts is not None) and in_window_count.get(client, 0) > 0:
            out_idx_to_drop.append(idx)
            continue

        # Keep row and push to queue if timestamp is valid
        if ts is not None:
            q.append((client, ts))
            in_window_count[client] += 1

    return df.drop(index=out_idx_to_drop)


def crop_filter(df: pd.DataFrame, threshold_rate: float = 0.85, step_rate: float = 0.001) -> pd.DataFrame:
    """
    중심점에서 멀리 있는 점들을 반복적으로 제거하여 데이터 수를 threshold_rate 비율까지 줄입니다.
    - 입력: df(xpos, ypos 필수)
    - 출력: 남은 행만 포함한 df 사본
    - 진행률: 데이터 수집/반복 제거 과정을 tqdm로 표시
    """
    for c in ('xpos', 'ypos'):
        if c not in df.columns:
            raise KeyError(f"컬럼 없음: {c}")

    # datas: [x, y, dist, index]
    datas: list[list[int | float]] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='crop: build datas', unit='row'):
        datas.append([row['xpos'], row['ypos'], 0, idx])

    threshold = int(len(datas) * threshold_rate)

    def refine_middle_point_local(items: list[list[int | float]]) -> tuple[int, int]:
        x_sum = 0
        y_sum = 0
        for x, y, _, _ in items:
            x_sum += x
            y_sum += y
        # 정수 중심점 유지(기존 로직 호환)
        return x_sum // len(items), y_sum // len(items)

    def del_outliers_local(x_mean: int, y_mean: int, items: list[list[int | float]], rate: float) -> None:
        for d in items:
            dx, dy = abs(d[0] - x_mean), abs(d[1] - y_mean)
            d[2] = max(dx, dy)
        items.sort(key=lambda t: t[2])
        cut_len = int(len(items) * rate)
        if cut_len > 0:
            del items[-cut_len:]

    it = 0
    with tqdm(desc='crop: iter', unit='iter') as pbar:
        while len(datas) > threshold and len(datas) > 0:
            it += 1
            x_mid, y_mid = refine_middle_point_local(datas)
            del_outliers_local(x_mid, y_mid, datas, rate=step_rate)
            pbar.set_postfix({'remain': len(datas), 'target': threshold})
            pbar.update(1)
            # 안전장치: 너무 적게 잘리는 경우 무한 루프 방지
            if step_rate == 0 or it > 50000:
                break

    remaining_indices = [int(d[3]) for d in datas]
    out = df.loc[remaining_indices].copy()
    tqdm.write(f"crop_filter: kept {len(out)} / {len(df)} rows")
    return out


def visualize_scatter_filter(df: pd.DataFrame,
                              x_col: str = 'xpos', y_col: str = 'ypos',
                              s: int = 3, alpha: float = 0.5,
                              out_path: Path | str = Path('../imgs') / 'pipeline_scatter.png',
                              figsize=(10, 10)) -> pd.DataFrame:
    """
    df를 산점도로 저장하는 부수효과성 필터. df는 수정하지 않고 그대로 반환.
    - 파일은 out_path로 저장하며, 디렉토리가 없으면 생성.
    """
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"컬럼 없음: {c}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=2, desc='visualize_scatter', unit='step') as pbar:
        plt.figure(figsize=figsize)
        plt.scatter(df[x_col], df[y_col], alpha=alpha, s=s)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title('Pipeline Scatter')
        plt.grid(True)
        pbar.update(1)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        pbar.update(1)

    tqdm.write(f"visualize_scatter: saved -> {out_path}")
    return df


# ---------- Pipeline ----------
def apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
    """
    Apply a sequence of filter functions to df.
    Each filter must be a callable: (pd.DataFrame) -> pd.DataFrame
    """
    for f in filters:
        df = f(df)
        print(f"Applied filter: {f.__name__} ({len(df)} rows)")

    return df


# ---------- __main__ ----------
if __name__ == '__main__':
    # 입력/출력 경로
    input_csv = Path('../data') / 'origin_data.csv'  # 필요 시 다른 원천 파일로 교체
    output_csv = Path('../data') / 'extraction.csv'

    # 적용할 필터를 원하는 순서로 구성
    # 예시: 지역 필터 -> 축소(crop) -> 시각화 -> 필수 컬럼 -> 중복 제거 -> 시간 포맷
    filters_to_apply = [
        remove_other_region,
        drop_nonessential_columns,
        eleminate_duplicates,
        crop_filter,
        visualize_scatter_filter,
        change_time_format,
    ]

    df0 = load_csv(input_csv, encoding='cp949')
    df_out = apply_filters(df0, filters_to_apply)
    save_csv(df_out, output_csv, encoding='cp949')
    print(f"완료: {output_csv} ({len(df_out)} rows)")
