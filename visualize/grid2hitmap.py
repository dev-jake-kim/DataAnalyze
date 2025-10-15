
import matplotlib.pyplot as plt
import numpy as np

def plot_grid_hitmap(grid, title="Grid Hitmap", save_path=None):
    # grid가 2차원 리스트(각 셀이 리스트)라면, 각 셀의 길이로 변환
    if isinstance(grid, list) and isinstance(grid[0][0], list):
        grid = np.array([[len(cell) for cell in row] for row in grid])
    else:
        grid = np.array(grid)

    plt.imshow(grid, cmap='Greys')
    plt.colorbar()

    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    if save_path:
        plt.savefig(save_path)
    plt.show()
