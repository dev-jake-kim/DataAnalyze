import pandas as pd
import numpy as np

from graphnize.utils.dataframe import min_max

def create_grid(data:np.ndarray, grid_size:int) -> np.ndarray:
    t_max = data[:,0].max() + 1
    x_max = data[:,1].max() + 1
    y_max = data[:,2].max() + 1
    grid = np.zeros((t_max, (x_max // grid_size) + 1, (y_max // grid_size) + 1), dtype=np.int32)
    for t, x, y in data:
        x_idx = x // grid_size
        y_idx = y // grid_size
        grid[t,  y_idx, x_idx] += 1
    return grid

def softmax_on_t(grid):
    max_t = np.max(grid, axis=0, keepdims=True)
    exp_grid = np.exp(grid - max_t)
    sum_t = np.sum(exp_grid, axis=0, keepdims=True)
    return exp_grid / sum_t

def sim(arr1, arr2):
    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)



    
    
