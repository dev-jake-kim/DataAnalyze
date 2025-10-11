import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
def function2graph(discrete_x, discrete_y, fmt = 'o-g'):
    plt.grid()
    plt.plot(discrete_x, discrete_y, fmt)
    plt.show()

def function2scatter(discrete_x, discrete_y, fmt = 'o'):
    plt.axis('equal')
    plt.grid()
    plt.plot(discrete_x, discrete_y, fmt)

    # x=0, y=0 축선 추가
    plt.axhline(y=0, color='red', linewidth=1)  # y=0 수평선
    plt.axvline(x=0, color='red', linewidth=1)  # x=0 수직선

    plt.show()

def various_data_graph(x,*datas):
    # plt.figure(figsize=)
    total_data = len(datas)
    for i, data in enumerate(datas):
        plt.subplot(total_data, 1, i+1)
        plt.plot(x, data)
        plt.title(f"data{i}")
        plt.axhline(y=0, color='black', linewidth=0.3)  # y=0 수평선
    plt.show()

def visualize_hit_map(ax:plt.Axes, hit_map:np.ndarray, cmap='Blues'):
    ax.clear()
    cax = ax.matshow(hit_map, cmap=cmap)
    ax.set_title('Hit Map')
    ax.set_xlabel('X Grid')
    ax.set_ylabel('Y Grid')


def create_animation(output_filename: str, hit_map: np.ndarray, cmap='Blues'):
    interval = hit_map.shape[0]
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        visualize_hit_map(ax, hit_map[frame], cmap=cmap)
        return ax,

    ani = animation.FuncAnimation(
        fig, update, frames=hit_map.shape[0], interval=interval, blit=False
    )
    ani.save(output_filename, writer='pillow', fps=10)
    plt.close(fig)

# if __name__ == "__main__":
#     # 예시 데이터 생성
#     hit_map = np.random.randint(0, 10, (20, 10, 10))  # 20 프레임, 각 프레임은 10x10 그리드
#
#     # 애니메이션 생성 및 저장
#     create_animation("hit_map_animation.gif", hit_map)



