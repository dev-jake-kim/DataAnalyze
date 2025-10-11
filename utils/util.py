import numpy as np

def sparse2hit_map(x_arr, y_arr, grid_size = (10,10)):
    x_min, x_max = min(x_arr), max(x_arr)
    y_min, y_max = min(y_arr), max(y_arr)
    map_shape = (int((y_max - y_min)//grid_size[1])+1, int((x_max - x_min)//grid_size[0])+1)
    hit_map = np.zeros(map_shape)
    for x, y in zip(x_arr, y_arr):
        x_idx = int((x - x_min)//grid_size[0])
        y_idx = int((y - y_min)//grid_size[1])
        hit_map[y_idx, x_idx] += 1
    return hit_map