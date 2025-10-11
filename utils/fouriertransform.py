from collections.abc import Callable
import numpy as np


def discretization(func: Callable, start:float, end:float, partition:int):
    discrete_x = np.linspace(start, end, partition)
    discrete_y = func(discrete_x)
    return discrete_x, discrete_y

def map_to_complex_plane(t_space, y_space, frequency):
    theta = t_space*frequency*2*np.pi #주파수를 곱하고(주기로 나누고) 라디안으로 바꿔줌
    real = np.cos(theta) * y_space
    imagine = np.sin(theta) * y_space #오일러 공식 사용
    return real, imagine

def middle_point(points):
    return np.average(points)


# if __name__ == "__main__":
#     x, y = discretization(func, 0, 5*np.pi, 10000)
#     function2graph(x, y)
#     real, imagine = map_to_complex_plane(x, y, 1/(2*np.pi))
#     function2scatter(real, imagine)
#
#     frequencies = np.linspace(1e-100, 1)
#     mid_x = np.zeros_like(frequencies)
#     mid_y = np.zeros_like(frequencies)
#     for idx, frequency in enumerate(frequencies):
#         real, imagine = map_to_complex_plane(x,y, frequency)
#         mid_x[idx] = middle_point(real)
#         mid_y[idx] = middle_point(imagine)
#     various_data_graph(frequencies, mid_x, mid_y, mid_x+mid_y)

