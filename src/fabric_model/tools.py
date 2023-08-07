import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from numpy import pi

NUM_SLICES = 5  # didn't used anywhere expect here
MODEL_PIC_PATH = [f'temp\\model{i}.png' for i in range(1, NUM_SLICES + 1)]

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def cart2pol(xy):
    x, y = xy[:, 0], xy[:, 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # return np.array([rho, phi])
    arr = np.swapaxes(np.array([rho, phi]), 0, 1)
    # arr = arr[arr[:, 1].argsort()]  # sort
    return arr


def pol2cart3d(theta, r, z):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = z
    return x, y, z


def cart2pol_(x, y, x0=0, y0=0):
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    phi = np.arctan2((y - y0), (x - x0))
    return rho, phi


def merge_slices_into_pd(slices, h):
    h = ['{:.1f}'.format(val) if isfloat(val) else val for val in h]  # round float values
    pd_slices = [pd.DataFrame(slice, columns=[h_, 'theta'])  # convert arrays to pandas frame array
                     .apply(lambda x: np.ceil(
        x * 100 / 5) * 5 / 100 if x.name == 'theta' else x)  # round theta to nearest 5 w 2 point of precision
                     .groupby(['theta']).mean()  # remove (group by mean) values w repeated theta
                 for slice, h_ in zip(slices, h)]
    result = pd.concat([df.stack() for df in pd_slices], axis=0).unstack()  # concat array to big frame
    result = result.interpolate()  # remove NaN values
    return result


def extract_slices_df(slices_df):
    theta = slices_df.index.values  # indexes are thetas
    h = slices_df.columns.values  # columns are h
    r = slices_df.values.T
    r_ = np.array([np.c_[sli, theta] for sli in r])  # join theta column to each slice
    r_ = [r[~np.isnan(r[:, 0])] for r in r_]  # remove nan
    return r_, h


def plot_df_on_ax(slices_df, ax):
    ax.cla()
    theta = slices_df.index.values  # indexes are thetas
    h = slices_df.columns.values  # columns are h
    r_ = slices_df.values.T
    if not ax.lines:
        for r in r_:
            ax.plot(theta, r)
        fontP = FontProperties()
        fontP.set_size('small')
        ax.legend(h, prop=fontP, loc='lower left', bbox_to_anchor=(-.1, -.1))
    else:
        for line, r in zip(ax.lines, r_):
            line.set_xdata(theta)
            line.set_ydata(r)


def circle(x0, y0, r):
    theta = np.linspace(0, 2 * np.pi)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0
    return x, y


def circle_fit(target, eps=.1, n_max=20, show_info=False, save_fig=False):
    df = pd.DataFrame({  # filter target x and y
        'x': target[:, 0],  # so if we will convert it to polar
        'y': target[:, 1],  # we'll get ~360 points
        'phi': np.arctan2(target[:, 0], target[:, 1]),  # for each degree
    }).apply(lambda x: np.around(x / np.pi * 180) if x.name == 'phi' else x) \
        .groupby(['phi']).mean()  #
    target = df[['x', 'y']].values  #

    if show_info or save_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')
        ax.set_title('Circle fitting')
        ax.scatter(target[::5][:, 0], target[::5][:, 1], s=10, marker='*', c='m')

    m = len(target)
    xn, yn, rn = 0, 0, max(target[:, 0])
    vn_history = []
    for n_iter in range(n_max):
        if show_info or save_fig:
            x_, y_ = circle(xn, yn, rn)
            ax.plot(x_, y_, c='#9f9f9f', linewidth=.2)
            ax.scatter(xn, yn, s=8, c='b', marker='.')

        ro_t, phi_t = cart2pol_(target[:, 0], target[:, 1], xn, yn)
        an = ro_t - rn  # compute CAF

        drn = np.sum(an) / m
        dxn = np.sum((an * np.cos(phi_t))) / m
        dyn = np.sum((an * np.sin(phi_t))) / m
        rn += drn
        xn = xn + dxn
        yn = yn + dyn

        vn = dxn ** 2 + dyn ** 2 + drn ** 2
        vn_history.append(vn)
        if vn < eps or n_iter > n_max:
            if show_info or save_fig:
                text = f'Fitting done in {n_iter + 1}/{n_max} iterations\nLyapunov function value - {vn:.5f}, with epsilon - {eps} \nFinal values are x0 - {xn:.3f}, y0 - {yn:.3f}, r - {rn:.3f}'
                x_, y_ = circle(xn, yn, rn)
                ax.plot(x_, y_, c='g')
                ax.scatter(xn, yn, s=40, c='g', marker='x')
                if save_fig:
                    try:
                        fig.savefig('temp\\circle_fit.png', dpi=200)
                        plt.close(fig)
                    except OSError as e:
                        print(e)
                        pass
                '''ax2 = fig.add_subplot(122)
                ax2.set_xlabel('Number of iterations')
                ax2.set_ylabel(r'$V_n$')
                ax2.plot(range(len(vn_history)), vn_history)'''
                if show_info:
                    print(text)
                    plt.show()
            return xn, yn, rn


def shadow_area(dots):
    by_angle = shadow_layer(dots)

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


def shadow_layer(dots):
    raw_dots = dots
    z = raw_dots[:, 2]
    dots = cart2pol(raw_dots)
    dots = np.swapaxes(np.array([dots[:, 0], dots[:, 1], z]), 0, 1)
    dots[:, 1] = np.around(dots[:, 1] * 180 / pi)  # convert to degrees and round result
    by_angle = [dots[dots[:, 1] == angle] for angle in np.unique(dots[:, 1])]  # separate r values by angle
    by_angle = np.array([arr[np.argmax(arr[:, 0])] for arr in by_angle])  # keep only highest r value
    by_angle[:, 1] = by_angle[:, 1] / 180 * pi  # convert back to radians
    return by_angle