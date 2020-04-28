import pandas as pd
import numpy as np
import pickle
from scipy.stats import mode
from f_model import fit, generate_x0, generate_n, f_model_h, calc_r2


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
    h = ['{:.1f}'.format(_) if isfloat(_) else _ for _ in h]  # round float values
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
        ax.legend(h)
    else:
        for line, r in zip(ax.lines, r_):
            line.set_xdata(theta)
            line.set_ydata(r)


def save_slices_df(df, path):
    df.to_csv(path)


def load_slices_df(path):
    return pd.read_csv(path, index_col='theta')


def make_models_from_df(slices_df, shadow_a=None):
    models_df, model_pic_paths = create_model_df(slices_df, pictures=True)
    line_df = create_line_eq_df(models_df.iloc[:-1])
    big_ar2 = r2_for_whole_model(slices_df.iloc[:, :-1], models_df.iloc[:-1], line_df)
    data_dict = {'model': models_df, 'line': line_df, 'data': slices_df, 'pic_paths': model_pic_paths,
                 'shadow_a': shadow_a, 'big_ar2': big_ar2}
    return data_dict


def save_results(data_dict, save_path):
    model_df, line_df, data_df, model_pic_paths, shadow_a, big_ar2 = data_dict.values()  # bruh unpacking
    write_results_to_excel(model_df, line_df, data_df, save_path, model_pic_paths, shadow_a, big_ar2)


def write_results_to_excel(models, line, data, path, model_pics_paths, shadow_a, big_ar2):
    filename = path
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    worksheet_result = workbook.add_worksheet('Result')
    worksheet_data = workbook.add_worksheet('Data')
    writer.sheets['Result'] = worksheet_result
    writer.sheets['Data'] = worksheet_data
    bold_cell = workbook.add_format({'bold': True, 'border': 1})
    subscript = workbook.add_format({'font_script': 2})

    df = models.loc[:, ['n', 'r0', 'r1', 'dfi1', 'k1', 'r2', 'dfi2', 'ar2', 'A']]  # reshape in cool format
    df.ar2 = df.ar2 * 100
    # df.columns = ['r₀', 'r₁', 'r₂', 'n', 'Δφ₁', 'Δφ₂', 'k₁', 'k₂', 'R²']
    df.columns = ['n', 'R₀', 'ΔR₁', 'Δφ₁', 'k₁', 'ΔR₂', 'Δφ₂', 'R², %',  'A, %']  # set pretty text formatting
    df.to_excel(writer, sheet_name='Result', startrow=0, startcol=0)
    worksheet_result.write_string(models.shape[0] + 3, 0, 'Коэфициенты пропорциональности')
    line.to_excel(writer, sheet_name='Result', startrow=df.shape[0] + 4, startcol=0)
    worksheet_result.write_string(0, 0, 'H', bold_cell)
    worksheet_result.write(df.shape[0] + 1, df.shape[1] - 1, big_ar2 * 100)
    worksheet_result.write_string(df.shape[0] + 1, df.shape[1] - 4, 'R² для поверхности пробы:')

    data.to_excel(writer, sheet_name='Data', startrow=0, startcol=0)
    worksheet_data.insert_image(2, data.shape[1] + 2, 'temp\\slices.png')
    worksheet_data.insert_image(20, data.shape[1] + 2, 'temp\\circle_fit.png', {'x_scale': 0.5, 'y_scale': 0.5})
    for pat, i, j in zip(model_pics_paths, [0, 0, 1, 1], [0, 1, 0, 1]):
        worksheet_result.insert_image(models.shape[0] + line.shape[0] + 6 + 19 * i, 5 + 6 * j, pat)
    if shadow_a is not None:  # ridiculous if statement :/Dₒₔ
        cols = [5, 6, 8, 9, 10, 11, 12]
        words = ['r', 'r', 'A', 'A', 'm', 'm', 'D']
        subs = ['пробы', 'диска', 's', 'o', 'sa', 'pr', 'd']
        [worksheet_result.write_rich_string(8, col, word, subscript, sub, bold_cell) for col, word, sub in
         zip(cols, words, subs)]
        cols = [8, 9, 10, 11]  # same as prev :(
        words = ['Ф.I', 'Ф.II', 'Ф.III', 'Ф.IV']
        [worksheet_result.write_string(11, col, word, bold_cell) for col, word in zip(cols, words)]
        worksheet_result.write_rich_string(12, 7, 'К', subscript, 'Д', ',%')

        worksheet_result.write_number(9, 8, shadow_a)  # I10 S
        worksheet_result.write_formula(9, 9, '=PI() * F10 * F10')  # J10 D
        worksheet_result.write_formula(9, 10, '=I10 - M10')  # K10 T
        worksheet_result.write_formula(9, 11, '=J10 - M10')  # L10 O
        worksheet_result.write_formula(9, 12, '=PI() * G10 * G10')  # M10 d
        worksheet_result.write_formula(12, 8, '=IF(L10 > 0, K10 / L10 * 100, "")')  # T/O
        worksheet_result.write_formula(12, 9, '=IF(J10 > 0, (J10 - I10) / J10 * 100, "")')  # (D - S) / D
        worksheet_result.write_formula(12, 10, '=IF(J10 >0, (J10 - I10) / (J10 - M10) * 100, "")')  # (D - S) / (D - d)
        worksheet_result.write_formula(12, 11, '=IF(J10 > 0, I10 / J10 * 100, "")')  # S/D
    writer.save()


def save_model(res, filename='data/test.pkl'):
    with open(filename, 'wb') as outfile:
        pickle.dump(res, outfile, -1)


def load_model(filename='data/test.pkl'):
    with open(filename, 'rb') as infile:
        o = pickle.load(infile)
    return o


def r2_for_whole_model(slice_df, m_df, lines_df):
    mode_par = mode(m_df).mode[0]  # extract mode values from each column
    model = f_model_h(mode_par, lines_df)

    thetas = slice_df.index.values
    hhs = np.asfarray(slice_df.columns.values, float)

    model_slices = np.array([model(thetas, h) for h in hhs]).T  # get model result for each theta for each h

    res = calc_r2(slice_df, model_slices)
    return res


def create_model_df(df, pictures=False):
    slices, hs = extract_slices_df(df)
    if not pictures:
        save_path = [None] * len(slices)
    else:
        save_path = [f'temp\\model{i}.png' for i in range(1, len(slices) + 1)]
    n = generate_n(slices[-1][:, 0])  # extract n from deepest layer
    fitt = [fit(generate_x0(d[:, 0], n=n), d, H=h, save_plot_path=path) for d, h, path in zip(slices, hs, save_path)]
    coefs = pd.concat([m.df for m in fitt])
    coefs['A'] = coefs['r2'] * 100 / coefs['r0']
    if not pictures:
        return coefs
    else:
        return coefs, save_path


def create_line_eq_df(df_model):
    x = [float(v) for v in df_model.index.values]
    values = df_model[['r0', 'r1', 'r2', 'k1']].values.T

    ar = []
    for y in values:
        ar.append(np.polyfit(x, y, 1))

    line_eq = pd.DataFrame(data=ar, index=['r0', 'r1', 'r2', 'k1'], columns=['a', 'b'])
    return line_eq


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
        import matplotlib.pyplot as plt
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
                    except e:
                        pass
                '''ax2 = fig.add_subplot(122)
                ax2.set_xlabel('Number of iterations')
                ax2.set_ylabel(r'$V_n$')
                ax2.plot(range(len(vn_history)), vn_history)'''
                if show_info:
                    print(text)
                    plt.show()
            return xn, yn, rn