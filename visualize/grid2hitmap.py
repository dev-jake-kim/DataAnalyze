import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def gridhitmap(grid, logscale: bool = False, title="Grid Hitmap", save_path=None):
    # grid를 2차원 리스트에서 카운트 배열로 변환
    if isinstance(grid[0][0], list):
        arr = np.array([[len(cell) for cell in row] for row in grid])
    else:
        arr = np.array(grid)
    # 0인 곳은 완전히 하얗게
    cmap = plt.get_cmap('Reds')
    cmap.set_under('white')
    if logscale:
        norm = mcolors.LogNorm(vmin=0.1, vmax=arr.max() if arr.max() > 0 else 1)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=arr.max() if arr.max() > 0 else 1)
    plt.figure(facecolor='white')
    im = plt.imshow(arr, cmap=cmap, norm=norm, origin='upper')
    plt.colorbar(im)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"Heatmap saved to {save_path}")
    plt.close()
