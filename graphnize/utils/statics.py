from pathlib import Path
import numpy as np


def five_num(data:np.ndarray, title="") -> None:
    five_num = np.percentile(data, [0, 25, 50, 75, 100])
    print(f"========= {title} =========")
    print(f"최솟값: {five_num[0]}")
    print(f"Q1 (25%): {five_num[1]}")
    print(f"중앙값 (50%): {five_num[2]}")
    print(f"Q3 (75%): {five_num[3]}")
    print(f"최댓값: {five_num[4]}")
    print("======================================")

if __name__ == "__main__":
    data_path = Path('output')
    demands_flat = np.load(data_path / 'demands25_10x10.npy').flatten()
    count_zero = np.sum(demands_flat == 0)
    print(f"Number of zero demands: {count_zero} out of {demands_flat.shape[0]} percentage: {count_zero / demands_flat.shape[0] * 100:.2f}%")
    five_num(demands_flat, 'Demands Statistics')