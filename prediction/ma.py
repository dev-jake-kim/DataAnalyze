import numpy as np
import pandas as pd

def get_sma_prediction(data, window_size):
    """Simple Moving Average 예측"""
    n, t = data.shape
    predictions = np.zeros((n, t))
    mask = np.zeros((n, t), dtype=bool)
    
    # SMA는 window_size 이전의 데이터가 없으면 예측 불가
    for i in range(window_size, t):
        # 직전 window_size 만큼의 평균으로 현재(i) 예측
        predictions[:, i] = np.mean(data[:, i-window_size:i], axis=1)
        mask[:, i] = True
        
    return predictions, mask

def get_ema_prediction(data, span):
    """
    Exponential Moving Average 예측
    span: window size와 유사한 개념 (alpha = 2 / (span + 1))
    """
    n, t = data.shape
    predictions = np.zeros((n, t))
    mask = np.zeros((n, t), dtype=bool)
    
    # EMA 계산을 위한 alpha 값
    alpha = 2 / (span + 1)
    
    # EMA 초기값 설정 (보통 첫 번째 데이터 포인트로 시작)
    # EMA[t]는 t시점까지의 관측치를 반영한 값
    ema_values = np.zeros((n, t))
    ema_values[:, 0] = data[:, 0] 
    
    # t=1 부터 루프 시작
    for i in range(1, t):
        # 1. 예측: 시점 i의 예측값은 시점 i-1까지 계산된 EMA 값
        predictions[:, i] = ema_values[:, i-1]
        mask[:, i] = True
        
        # 2. 업데이트: 시점 i의 실제 데이터를 관측하여 EMA 업데이트
        # EMA_new = alpha * current_val + (1-alpha) * EMA_old
        ema_values[:, i] = alpha * data[:, i] + (1 - alpha) * ema_values[:, i-1]
        
    return predictions, mask

def calculate_mae(y_true, y_pred, mask):
    """마스크 된 유효 영역에 대해서만 MAE 계산"""
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def tune_moving_averages(data, window_sizes):
    """
    여러 Window Size에 대해 SMA와 EMA를 수행하고 결과를 비교
    """
    results = []
    
    print(f"{'Type':<10} | {'Window/Span':<12} | {'MAE':<10}")
    print("-" * 36)
    
    for w in window_sizes:
        # 1. SMA 평가
        sma_pred, sma_mask = get_sma_prediction(data, w)
        sma_mae = calculate_mae(data, sma_pred, sma_mask)
        
        results.append({'Type': 'SMA', 'Window': w, 'MAE': sma_mae})
        print(f"{'SMA':<10} | {w:<12} | {sma_mae:.4f}")
        
        # 2. EMA 평가
        ema_pred, ema_mask = get_ema_prediction(data, w)
        ema_mae = calculate_mae(data, ema_pred, ema_mask)
        
        results.append({'Type': 'EMA', 'Window': w, 'MAE': ema_mae})
        print(f"{'EMA':<10} | {w:<12} | {ema_mae:.4f}")
        
    return pd.DataFrame(results)

# --- 실행 예시 ---

# 1. 데이터 생성 (노드 10개, 타임스텝 100개)
import json
from pathlib import Path
source_path = Path('output')
with open(source_path / 'gwn_data.json', 'r', encoding='utf-8') as f:
    gwn_data = json.load(f)
data = np.array(gwn_data['x'])
print(data.shape)

# 2. 튜닝할 윈도우 사이즈 목록
window_list = [3, 5, 10, 20, 24, 48]

# 3. 튜닝 및 결과 출력
print(">>> Evaluating Moving Average Models...\n")
df_results = tune_moving_averages(data, window_list)

# 4. 최적의 설정 찾기
best_row = df_results.loc[df_results['MAE'].idxmin()]
print("-" * 36)
print(f"Best Configuration:\nType: {best_row['Type']}, Window: {best_row['Window']}, MAE: {best_row['MAE']:.4f}")