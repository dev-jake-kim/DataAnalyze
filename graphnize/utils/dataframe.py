from pathlib import Path
import pandas as pd
import numpy as np
def data_load(root:Path, filename:str, encoding:str='cp949', essential_cols=None) -> pd.DataFrame:
    data_path = root / filename
    df = pd.read_csv(data_path, encoding=encoding)
    if essential_cols:
        for col in essential_cols:
            if col not in df.columns:
                raise KeyError(f"필수 컬럼 누락: {col}")
    return df

def min_max(df:pd.DataFrame, col:str) -> tuple:
    return df[col].min(), df[col].max()

def zero_based_regulization(df:pd.DataFrame, col:str) -> pd.Series:
    col_min = df[col].min()
    return (df[col] - col_min)

def df2np(df:pd.DataFrame) -> np.ndarray:
    min_t = df['call_date'].min()
    df['time_idx'] = (df['call_date'] - min_t).dt.total_seconds() // 3600
    return df[['time_idx', 'xpos', 'ypos']].to_numpy(dtype=int)