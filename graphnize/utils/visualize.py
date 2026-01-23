from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np


def visualize_dtw(dtw_matrix: np.ndarray, save_path: Path) -> None:
    """DTW 유사도 행렬 시각화 및 저장."""
    plt.figure(figsize=(10, 8))
    plt.imshow(dtw_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='DTW Distance')
    plt.title('DTW Similarity Matrix')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.savefig(save_path)
    plt.close()

def visualize_demands(demand_series: np.ndarray, save_path: Path, max_plot = 1) -> None:
    """수요 시계열 시각화 및 저장."""
    original_reshaped = demand_series.reshape(demand_series.shape[0], -1, 7*24)
    weekly_profile = original_reshaped.mean(axis=1)
    deviations = original_reshaped - weekly_profile[:, np.newaxis, :]
    demand_series = deviations.reshape(demand_series.shape[0], -1)
    plt.figure(figsize=(40, 6))
    for i in range(min(demand_series.shape[0], max_plot)):
        plt.plot(demand_series[i][:7*24*2], label=f'Node {i}')
    plt.title('Demand Time Series')
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    data_path = Path('output')
    img_path = Path('imgs')
    demand_series = np.load(data_path / 'demands25_10x10.npy')
    visualize_demands(demand_series, img_path / 'demand_series.png')