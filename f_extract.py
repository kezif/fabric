import numpy as np
from stl import mesh
from f_tools import cart2pol, circle_fit, shadow_layer


def layer_from_dots(dots, height):
    h0 = max(dots[:, 2])
    h = h0 - height  # extract all dots from h
    margin = .3  # with tolerance -+
    index = (dots[:, 2] < h + margin) & (dots[:, 2] > h - margin)  # filter using boolean indexing
    valid_dots = dots[index]
    return np.array(valid_dots)


def read_stl(filename, circle_h=3, save_fig=True):
    d_mesh = mesh.Mesh.from_file(filename, calculate_normals=False)
    dots = d_mesh.vectors  # it looks like .vectors and .dots return same shape result..
    dots = np.unique(dots.reshape(len(dots) * 3, 3), axis=0)  # extract all dots from file
    # f = len(dots) // 15000  # if file size (n of dots) is too big - some dots will be skipped
    # dots = dots[::f]
    dots = dots[dots[:, 2].argsort()]  # sort by z (for faster filtering)
    layer = layer_from_dots(dots, circle_h)
    dx, dy, _ = circle_fit(layer, save_fig=save_fig)  # find the "real" center of data using layer
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
        layer_polar = cart2pol(layer) + [0, rot]
        slices.append(layer_polar)  # save slice
        h.append(layer_h - top_h)  # and corresponding h
    slices.append(shadow_layer(dots)[:, :2])  # and also append shadow layer
    h.append('shadow')
    slices = np.array(slices)
    return slices, h


