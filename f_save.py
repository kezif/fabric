import pickle

import pandas as pd
from f_tools import MODEL_PIC_PATH

def save_slices_df(df, path):
    df.to_csv(path)


def load_slices_df(path):
    return pd.read_csv(path, index_col='theta')


def save_results(model_data_dict, shadow_a=0, rsmoll=0, rbig=0, save_path='test.xlsx'):
    model_df, line_df, data_df, big_ar2 = model_data_dict.values()  # bruh unpacking
    write_results_to_excel(model_df, line_df, data_df, save_path, shadow_a, big_ar2, rsmoll, rbig)


def write_results_to_excel(models, line, data, path, shadow_a, big_ar2, rsmoll, rbig):
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
    #df.ar2 = df.ar2 * 100
    # df.columns = ['r₀', 'r₁', 'r₂', 'n', 'Δφ₁', 'Δφ₂', 'k₁', 'k₂', 'R²']
    df.columns = ['n', 'R₀', 'ΔR₁', 'Δφ₁', 'k₁', 'ΔR₂', 'Δφ₂', 'R²',  'A, %']  # set pretty text formatting
    df.to_excel(writer, sheet_name='Result', startrow=0, startcol=0)
    worksheet_result.write_string(models.shape[0] + 3, 0, 'Коэфициенты пропорциональности')
    line.to_excel(writer, sheet_name='Result', startrow=df.shape[0] + 4, startcol=0)
    worksheet_result.write_string(0, 0, 'H', bold_cell)
    worksheet_result.write(df.shape[0] + 1, df.shape[1] - 1, big_ar2 )
    worksheet_result.write_string(df.shape[0] + 1, df.shape[1] - 4, 'R² для поверхности пробы:')

    worksheet_result.write('F10', rbig)
    worksheet_result.write('G10', rsmoll)

    data.to_excel(writer, sheet_name='Data', startrow=0, startcol=0)
    worksheet_data.insert_image(2, data.shape[1] + 2, 'temp\\slices.png')
    worksheet_data.insert_image(20, data.shape[1] + 2, 'temp\\circle_fit.png', {'x_scale': 0.5, 'y_scale': 0.5})
    for pat, i, j in zip(MODEL_PIC_PATH, [0, 0, 1, 1], [0, 1, 0, 1]):
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