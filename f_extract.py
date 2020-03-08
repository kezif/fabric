import matplotlib.pyplot as plt
import numpy as np
from stl import mesh
from f_tools import cart2pol as cart2pol_, circle_fit
from math import pi


def layer_from_dots(dots, height):
    h0 = max(dots[:, 2])
    h = h0 - height  # extract all dots from h
    margin = .3  # with tolerance -+
    index = (dots[:, 2] < h + margin) & (dots[:, 2] > h - margin)  # filter using boolean indexing
    valid_dots = dots[index]
    return np.array(valid_dots)


def read_stl(filename, circle_h=3):
    d_mesh = mesh.Mesh.from_file(filename, calculate_normals=False)
    dots = d_mesh.vectors  # it looks like .vectors and .dots return same shape result..
    dots = np.unique(dots.reshape(len(dots) * 3, 3), axis=0)  # extract all dots from file
    # f = len(dots) // 15000  # if file size (n of dots) is too big - some dots will be skipped
    # dots = dots[::f]
    dots = dots[dots[:, 2].argsort()]  # sort by z (for faster filtering)
    layer = layer_from_dots(dots, circle_h)
    dx, dy, _ = circle_fit(layer, save_fig=True)  # find the "real" center of data using layer
    dots = dots - [dx, dy, 0]  # so yea
    return dots


def slice_dots(dots, n: int, max_h: int, ignore_before_h: int = 15, top_h: int = 3, rot: float = 0):
    slices = []
    h = []
    bottom_h = ignore_before_h  # + top_h  # h from disk (0) to valid layers
    hh = (max_h - bottom_h) / (n - 1)  # step size
    for i in range(n):
        layer_h = (top_h + bottom_h + hh * i)  # find layer height
        layer = layer_from_dots(dots, layer_h)
        layer_polar = cart2pol_(layer) + [0, rot]
        slices.append(layer_polar)  # save slice
        h.append(layer_h)  # - top_h)  # and corresponding h
    slices = np.array(slices)
    return slices, h


def shadow_area(dots):
    raw_dots = dots
    z = raw_dots[:, 2]

    dots = cart2pol_(raw_dots)
    dots = np.swapaxes(np.array([dots[:, 0], dots[:, 1], z]), 0, 1)

    dots[:, 1] = np.around(dots[:, 1] * 180 / pi)  # convert to degrees and round result
    by_angle = [dots[dots[:, 1] == angle] for angle in np.unique(dots[:, 1])]  # separate r values by angle
    by_angle = np.array([arr[np.argmax(arr[:, 0])] for arr in by_angle])  # keep only highest r value
    by_angle[:, 1] = by_angle[:, 1] / 180 * pi  # convert back to radians

    rs = by_angle[:, 0]  # keep only r values
    d_alpha = 2 * pi / len(rs)  # shadow area calculation
    r1 = rs[:-1]
    r2 = rs[1:]
    cos_alpha = np.cos(d_alpha)
    a = np.sqrt(r1 ** 2 + r2 ** 2 - (2 * r1 * r2 * cos_alpha))
    p = (r1 + r2 + a) / 2
    s = np.sqrt(p * (p - r1) * (p - r2) * (p - a))
    a_s = np.sum(s)
    return a_s


def drap_coef(filename, r, R, show_plot=False):
    raw_dots = read_stl(filename)
    z = raw_dots[:, 2]

    dots = cart2pol_(raw_dots)
    dots = np.swapaxes(np.array([dots[:, 0], dots[:, 1], z]), 0, 1)

    dots[:, 1] = np.around(dots[:, 1] * 180 / pi)  # convert to degrees with rounding
    by_angle = [dots[dots[:, 1] == _] for _ in np.unique(dots[:, 1])]  # separate values by angle
    by_angle = np.array([arr[np.argmax(arr[:, 0])] for arr in by_angle])  # extract by highest r value
    by_angle[:, 1] = by_angle[:, 1] / 180 * pi  # convert back to radians

    rs = by_angle[:, 0]  # keep only r values
    d_alpha = 2 * pi / len(rs)  # shadow area
    r1 = rs[:-1]
    r2 = rs[1:]
    cos_alpha = np.cos(d_alpha)
    a = np.sqrt(r1 ** 2 + r2 ** 2 - (2 * r1 * r2 * cos_alpha))
    p = (r1 + r2 + a) / 2
    s = np.sqrt(p * (p - r1) * (p - r2) * (p - a))
    a_s = np.sum(s)

    a_r = pi * r ** 2
    a_R = pi * R ** 2

    D = a_R
    d = a_r
    S = a_r
    T = a_r - d
    O_d = a_r - d

    d_11 = T / O_d
    d_15 = (D - S) / D
    d_16 = (D - S) / (D - d)
    d_117 = S / D

    d_1_2 = (a_s - a_r) / (a_R - a_r)
    d_1_5 = (a_R - a_s) / a_R
    d_1_7 = (a_R - a_s) / (a_R - a_r)

    '''name = filename.split('\\')[-1]
    print(
        f'File - {name}\nR area - {a_R:.0f} mmsqr\nr area - {a_r:.2f} mmsqr\nShadow area - {a_s:.0f} mmsqr\n\n1.2 - {d_1_2:.2f}\n1.5 - {d_1_5:.2f}\n1.7 - {d_1_7:.2f}')'''
    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        skip = 10
        n = 1  # int(len(raw_dots) / 2)
        ax.scatter(raw_dots[:-n, 0][::skip], raw_dots[:-n, 1][::skip], raw_dots[:-n, 2][::skip], s=.2)

        b = by_angle[by_angle[:, 1].argsort()]

        def p2c(r, th):
            return r * np.cos(th), r * np.sin(th)

        x, y = p2c(b[:, 0], b[:, 1])
        ax.plot(x, y, b[:, 2], c='r')
        ax.plot(x, y, c='m')

        ax.set_zlim3d(10, -70)
        plt.show()

    return d_1_2, d_1_5, d_1_7


if __name__ == "__main__":
    # c, h = slice_stl('data/fabric.stl', 6, 28, 15, 3, 0.14, 120, 0)
    # plot_clices(c, h, 6)
    drap_coef('data//2019_12_05-1-30-18.stl', 180 / 2, 300 / 2, show_plot=True)
