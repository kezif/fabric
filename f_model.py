from scipy.optimize import minimize, Bounds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from math import ceil, pi


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


data_tuple = namedtuple('dots', ['vars', 'model', 'var_map', 'df', 'text', 'dots', 'x0'])
globals_names = {'np': np}
str_expr = 'r0 + r1 * ((1 + np.sin(n * thetas + dfi1)) / 2) ** k1 + r2 * ((1 + np.sin(2 * thetas + dfi2)) / 2) ** k2'


def loss(target, predicted):  # define loss function
    return np.mean(np.sum((target - predicted) ** 2))


def f(thetas):
    def closure(par):
        r0, r1, r2, n, dfi1, dfi2, k1, k2 = par
        n = int(n)
        locals_names = {'r0': r0, 'r1': r1, 'r2': r2, 'n': n, 'dfi1': dfi1, 'dfi2': dfi2, 'k1': k1, 'k2': k2,
                        'thetas': thetas}
        # local_names = dict(zip(var_names, par))
        return eval(str_expr, globals_names, locals_names)

    return closure


def f_model(par):
    r0, r1, r2, n, dfi1, dfi2, k1, k2 = par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7]
    return lambda thetas: eval(str_expr, globals_names,
                               {'r0': r0, 'r1': r1, 'r2': r2, 'n': n, 'dfi1': dfi1, 'dfi2': dfi2, 'k1': k1, 'k2': k2,
                                'thetas': thetas})


def f_model_h(par, line_df):
    _, _, _, n, dfi1, dfi2, _, k2 = par[0], par[1], par[2], par[3], par[4], par[5], par[6], par[7]

    def mod(thetas, h):
        r0, r1, r2, k1 = np.polyval(line_df.values.T, h)
        return eval(str_expr, globals_names,
                    {'r0': r0, 'r1': r1, 'r2': r2, 'n': n, 'dfi1': dfi1, 'dfi2': dfi2, 'k1': k1, 'k2': k2,
                     'thetas': thetas})

    return mod


def generate_n(r):
    middle = np.median(r)  # find median
    r = r - middle  # split values into lower and higher then med
    r = r[r != 0.]  # remove 0
    sign = np.sign(r)  # map vales into -1 if lower or 1
    sign = sign[1:][sign[1:] != sign[:-1]]  # remove values if there are repeated: 1,1,1,1,-1 -> 1,-1
    print(len(sign), len(sign) // 2)
    return ceil(len(sign) / 2)


def generate_x0(r, n=None):
    if n is not None:
        n = n  # pass
    else:
        n = generate_n(r)
    r0 = ceil(np.min(r))
    r1 = ceil(np.max(r) - np.min(r)) * .8  # ceil(r0 / 10)
    r2 = ceil(r0 / 15)
    return np.array([r0, r1, r2, n, 0, 0, 1, 4])


'''def split(data, test_size=0.25):
    # shuffle(data)
    middle = int(len(data) * test_size)
    test = data[:middle]
    train = data[middle:]
    return test, train'''


def calc_r2(x_actual, x_model):  # coefficient of determination
    ss_tot = loss(x_actual, np.mean(x_actual))
    ss_res = loss(x_actual, x_model)
    return 1 - ss_res / ss_tot


# compose everything into one fun
def fit(x0, data_r_th, save_plot_path=None, show_plot=False, H=None, text_output=False):
    r, th = data_r_th[:, 0], data_r_th[:, 1]
    # Find model coefficients using minimization method
    fun = f(th)  # 1. Put variables into model
    g = lambda par: loss(r, fun(par))  # 2. define loss metric for given coeffs
    bounds = Bounds([0., 0., 0., 3, -np.inf, -np.inf, 1., 4.], [200, 100, 100, 12, np.inf, np.inf, 3., 4.])
    opt_result = minimize(g, x0, method='L-BFGS-B', bounds=bounds,
                          options={'eps': .001})  # 3. Find coeffs using scipy
    weights = opt_result.x
    model = f_model(weights)

    num = len(r)
    k = 7  # number of model coeffs
    error = loss(r, model(th))
    mean_of_observed = np.mean(r)
    std = (error / (num - 1)) ** (1 / 2)
    '''ss_tot = loss(r, mean_of_observed)
    # ss_reg = loss(model(th), mean_of_observed)
    ss_res = loss(r, model(th))
    r2 = 1 - ss_res / ss_tot'''
    r2 = calc_r2(r, model(th))
    r2_adj = r2  # 1 - (1 - r2) * (num - 1) / (num - k - 1)

    text = '''Results:
n - {}
Quadratic cost - {:.2f}
Standard deviation - {:.2f}
Mean - {:.2f}
R-squared - {:.2f}
Adjusted R-squared - {:.2f}
|r0    {:.2f}
r1     {:.2f}
r2     {:.2f}
n      {:.0f}
dfi1   {:.2f}
dfi2   {:.2f}
k1     {:.2f}
k2     {:.2f}
'''.format(num, error, std, mean_of_observed, r2, r2_adj, *weights)
    if text_output:
        print(text)

    if show_plot or save_plot_path is not None:
        name = (r'$H = ' + H + '$\n' + r'$R^2 = $' + '{:.2f}'.format(r2_adj)) if H is not None and isfloat(H) else ''
        plot_data_n_model(data_r_th, model, save_plot_path, plot_name=name)

    var_d = {'r0': weights[0], 'r1': weights[1], 'r2': weights[2], 'n': int(weights[3]), 'dfi1': weights[4],
             'dfi2': weights[5], 'k1': weights[6], 'k2': weights[7], 'ar2': r2_adj}
    df = pd.DataFrame(var_d, index=[H])
    res = data_tuple(vars=weights, model=model, var_map=var_d, df=df, text=text, dots=data_r_th, x0=x0)
    return res


def plot_data_n_model(data_r_th, model, save_im_path=None, plot_name=' '):
    data_r_th = data_r_th[::2]
    r, th = data_r_th[:, 0], data_r_th[:, 1]
    fig = plt.figure(figsize=(3.75, 3.75))
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111, projection='polar')
    th_l = np.linspace(0, 2 * pi, num=150)

    ax.plot(th_l, model(th_l), color='g')
    ax.scatter(th, r, s=8, color='r',
               marker='x')
    format_ax(ax, r)
    ax.set_xticklabels([r'$0^{\circ}$', r'$45^{\circ}$'])  # , '', '', '', r'$180^{\circ}$'])
    ax.set_title(plot_name, loc='left')
    ax.legend(['Prediction', 'Target'])
    if save_im_path is not None:
        fig.savefig(save_im_path, dpi=200)
    else:
        plt.show()


def format_ax(ax, r):  # move to another file
    r = r[~np.isnan(r)]
    ax.set_ylim([0, np.max(r) + 10])

    def format_func(value, tick_number):
        if value < .75 * np.min(r):
            return ''
        else:
            return '{:.0f}'.format(value)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))


'''def plot_result(axes, data, model):
    r, th = data[:, 0], data[:, 1]  # функция - копия plot_data_n_model f_model
    th_l = np.linspace(0, 2 * pi, num=150)  # 150 точек от 0 до 2пи (360 гралусов)

    axes.plot(th_l, model(th_l), color='g')
    n = len(th) // 100
    axes.scatter(th[::n], r[::n], s=8, color='r',
                 marker='x')  # наносим на график каждую n точку. С размером s ну и цветом
    # axes.legend(['Prediction', 'Target'])
    return axes
'''

if __name__ == "__main__":
    pass
