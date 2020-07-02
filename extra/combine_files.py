import pandas as pd
import numpy as np
import os.path
import os
import sys
from collections import defaultdict
from scipy.stats import mannwhitneyu, ttest_ind
from traitsui import message
from traitsui.editors import CheckListEditor
from traits.has_traits import HasTraits
from traits.trait_types import Button, List, Directory, Enum, String, Int
from pyface.api import FileDialog, OK
from traitsui.api import View, Item
from traitsui.group import HGroup

MANNWHITNEYTABLE = [None, None, None, 0, 2, 5, 8, 13, 17, 23, 30, 37, 45, 55, 64, 75, 87, 99, 113, 127]
TTABLE = [6.314, 2.920, 2.353, 2.132, 2.015,
          1.943, 1.895, 1.860, 1.833, 1.812,
          1.796, 1.782, 1.771, 1.761, 1.753,
          1.746, 1.740, 1.734, 1.729, 1.725,
          1.721, 1.717, 1.714, 1.711, 1.708,
          1.706, 1.703, 1.701, 1.699, 1.697,
          ]


def load_data_from_excel(path):
    try:
        # config file with save format mby? And normal way for reading file
        df = pd.read_excel(path, sheet_name=0)  # bad indexing. Would broke
        DRAP = df.loc[11, 'R², %']  # bad indexing. Would broke
        DETER = df.loc[5, 'R², %']  # bad indexing. Would broke
        ANTIZOT = df.loc[4, 'A, %']  # bad indexing. Would broke
        name = os.path.basename(path).split('.')[0]
        return DRAP, DETER, ANTIZOT, name
    except KeyError:
        return [None] * 4


def load_data_from_excel_old(path):
    try:
        # config file with save format mby? And normal way for reading file
        df = pd.read_excel(path, sheet_name=0)  # bad indexing. Would broke
        DRAP = df.iloc[11, 8]  # bad indexing. Would broke
        if DRAP is None:
            DRAP = 0
        DETER = df.iloc[5, 8]  # bad indexing. Would broke
        ANTIZOT = df.iloc[4, 6] / df.iloc[4, 2] * 100
        # df.to_csv('dsad.csv')
        name = os.path.basename(path).split('.')[0]
        return DRAP, DETER, ANTIZOT, name
    except KeyError:
        return [None] * 4


def load_n(path):
    try:
        # config file with save format mby? And normal way for reading file
        df = pd.read_excel(path, sheet_name=0)  # bad indexing. Would broke
        n = df.iloc[4, 1]
        return n
    except KeyError:
        return None

def load_n_ffolder(folder_path):
    list_n = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            xlsx_path = os.path.join(folder_path, file)
            n = load_n(xlsx_path)
            list_n.append(n)
    return list_n


def find_number_of_samples(folder_path):
    counter = 0
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            counter += 1
    return counter


def exctr_data_ffiles(folder_path):
    list_models = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            xlsx_path = os.path.join(folder_path, file)
            data = load_data_from_excel(xlsx_path)
            if data[0] is None:
                data = load_data_from_excel_old(xlsx_path)
            list_models.append(data)

    folder_models = pd.DataFrame(list_models, columns=['D', 'R2', 'A', 'name'])
    folder_models.set_index('name', inplace=True)
    # print(folder_models)
    folder_models.to_csv(os.path.join(folder_path, 'combined.csv'))
    return folder_models


def mannwhit_test(a, b, cols):
    test_res = [mannwhitneyu(a.iloc[:, i], b.iloc[:, i]).statistic for i in
                range(3)]  # *list(map(lambda x: x.iloc[:, 0], list_res)))
    res = dict(zip(cols, test_res))
    text = f'Результаты теста с использованием кртерия Манна-Уитни {res}'
    # print(text)
    return res, text


def t_student_abs_value(a, b, cols):
    test_res = ttest_ind(a, b).statistic
    res = dict(zip(cols, abs(test_res)))
    text = f'Результаты теста c использованием критерия Стьюдента {res}'
    # print(text)
    return res, text


def get_data_from_folders(folder_path1, folder_path2):
    samples = defaultdict(None)  #
    samples[os.path.basename(folder_path1)] = exctr_data_ffiles(folder_path1)
    samples[os.path.basename(folder_path2)] = exctr_data_ffiles(folder_path2)
    return samples


def compare_folders(folder_path1, folder_path2):
    """
    Function to couple functions from above
    """
    samples = get_data_from_folders(folder_path1, folder_path2)
    # print([(k, i) for k, i in samples.items()])

    res = [i for i in samples.values()]
    cols = res[0].columns  # ['a', 'b', 'c']
    text_compar = ('Сравнение между папками {} и {}'.format(*[i for i in samples.keys()]))
    man, text_man = mannwhit_test(*res, cols)
    stu, text_stu = t_student_abs_value(*res, cols)
    return man, stu, text_compar


class DataProperties(HasTraits):
    def __init__(self):
        super(DataProperties, self).__init__()

    folder = Directory()
    compute = Button()

    traits_view = View(
        Item('folder', show_label=True),
        Item('compute'),
        width=.3, height=.1, resizable=True, title='_'
    )

    def _compute_fired(self):
        data = exctr_data_ffiles(self.folder)
        ns = load_n_ffolder(self.folder)
        n = sum(ns) / len(ns)
        data[['D']] = data[['D']] * 100
        data['R2'] = data['R2'].apply(lambda x: x if x < 1 else x*100)
        mean, std = data.mean(), data.std()
        v = (100 - mean['D']) * np.sqrt(n * mean['R2']) / (3 * (mean['A'] + 2 * std.loc['A']))

        wildcard = '*.txt'
        default_path = os.path.join(os.path.dirname(self.folder), os.path.basename(
            self.folder) + ' показатели' + '.txt')
        dialog = FileDialog(title='Save results',
                            action='save as', wildcard=wildcard,
                            default_path=default_path
                            )
        if dialog.open() == OK:
            with open(dialog.path, 'w', encoding="utf-8") as file:
                file.write(f'Показатели для {os.path.basename(self.folder)}\n\n')
                file.write('Среднее:\n')
                file.write(mean.to_string())
                file.write('\n\nСредне квадратичное отклоенение:\n')
                file.write(std.to_string())
                file.write('\n\nV - ')
                file.write(str(v))
            print('done')

class CompareFolders(HasTraits):
    def __init__(self):
        super(CompareFolders, self).__init__()

    # methods_dic = {'mannwhit_test': mannwhit_test, 't_student': t_student}
    methods_list = ['U-критерий', 't-критерий']
    folder1 = Directory()
    folder2 = Directory()
    methods1 = List(editor=CheckListEditor(values=methods_list, ), value=methods_list)
    methods2 = List(editor=CheckListEditor(values=methods_list, ), value=methods_list)
    methods3 = List(editor=CheckListEditor(values=methods_list, ), value=methods_list)
    method_str = String('Выбор методов для параметров:')
    ALPHA_VALUES = ['.05', 'вручную']
    alpha = Enum(*ALPHA_VALUES)
    alpha_manual = Int(label='alpha')
    compute = Button()

    traits_view = View(
        Item('folder1', show_label=True),
        Item('folder2', show_label=True),
        '_',
        Item('method_str', style='readonly', show_label=False),
        HGroup(
            Item('methods1', springy=True, style='custom', label='D'),
            Item('methods2', springy=True, style='custom', label='R2'),
            Item('methods3', springy=True, style='custom', label='A'),
        ),

        Item('alpha', style='custom'),
        Item('alpha_manual'),
        Item('compute'),
        width=.3, height=.2, resizable=True, title='Compare folders'
    )

    def _compute_fired(self):
        # print(self.alpha)
        if self.methods1 != [] and self.methods2 != [] and self.methods3 != [] and self.folder1 is not '' and self.folder2 is not '':
            man, stu, text = compare_folders(self.folder1, self.folder2)
            number_of_samples = find_number_of_samples(self.folder1)

            # mann_value = 0
            if self.alpha == self.ALPHA_VALUES[0]:  # table lookup
                mann_crit = MANNWHITNEYTABLE[
                    number_of_samples - 1]  # table is zero indexed, so for 18 files we need 17 value
            elif self.alpha == self.ALPHA_VALUES[1]:  # manual
                mann_crit = self.alpha_manual
            else:
                mann_crit = 0

            t_value = TTABLE[number_of_samples - 2]  # same as above, but coeff of freedom is n-1

            wildcard = '*.txt'
            default_path = os.path.dirname(self.folder1) + '/' + os.path.basename(
                self.folder1) + '__' + os.path.basename(
                self.folder2) + '.txt'
            dialog = FileDialog(title='Save results',
                                action='save as', wildcard=wildcard,
                                default_path=default_path
                                )
            if dialog.open() == OK:
                with open(dialog.path, 'w', encoding="utf-8") as file:
                    file.write(
                        text + f'\nКритическое значение критерия Манна - Уитни: {mann_crit}\nКритическое значение коэфициента Стьюднта: {t_value}\n')

                    file.write(f'\nПоказатель Кд:\n')
                    if self.methods_list[
                        0] in self.methods1:  # revritre in better way, also refactor to separete function would be great
                        file.write(
                            f'\tРасчетное значение критерия Манна — Уитни: {man["D"]:.3f}\n' +  # at least move repeated chunks of code into function
                            f'\tСравнение с критическим значением: {man["D"]:.3f} и {mann_crit}\n' +
                            f'\tРасчетное значение {"меньше критического - различия" if man["D"] <= mann_crit else "больше критического - различия не"} подтверждены статистически\n'
                        )  # bruh
                    if self.methods_list[1] in self.methods1:
                        file.write(f'\tРасчетное значение коэфициента Стьюдента: {stu["D"]:.3f}\n' +
                                   f'\tСравнение с критическим значением: {stu["D"]:.3f} и {t_value}\n' +
                                   f'\tРасчетное значение {"больше критического - различия" if stu["D"] >= t_value else "меньше критического - различия не"} подтверждены статистически\n'
                                   )

                    file.write(f'\nПоказатель R2:\n')
                    if self.methods_list[0] in self.methods2:
                        file.write(f'\tРасчетное значение критерия Манна — Уитни: {man["R2"]:.3f}\n' +
                                   f'\tСравнение с критическим значением: {man["R2"]:.3f} и {mann_crit}\n' +
                                   f'\tРасчетное значение {"меньше критического - различия" if man["R2"] <= mann_crit else "больше критического - различия не"} подтверждены статистически\n'
                                   )
                    if self.methods_list[1] in self.methods2:
                        file.write(f'\tРасчетное значение коэфициента Стьюдента: {stu["R2"]:.3f}\n' +
                                   f'\tСравнение с критическим значением: {stu["R2"]:.3f} и {t_value}\n' +
                                   f'\tРасчетное значение {"больше критического - различия" if stu["R2"] >= t_value else "меньше критического - различия не"} подтверждены статистически\n'
                                   )

                    file.write(f'\nПоказатель A:\n')
                    if self.methods_list[0] in self.methods3:
                        file.write(f'\tРасчетное значение критерия Манна — Уитни: {man["A"]:.3f}\n' +
                                   f'\tСравнение с критическим значением: {man["A"]:.3f} и {mann_crit}\n' +
                                   f'\tРасчетное значение {"меньше критического - различия" if man["A"] <= mann_crit else "больше критического - различия не"} подтверждены статистически\n'
                                   )
                    if self.methods_list[1] in self.methods3:
                        file.write(f'\tРасчетное значение коэфициента Стьюдента: {stu["A"]:.3f}\n' +
                                   f'\tСравнение с критическим значением: {stu["A"]:.3f} и {t_value}\n' +
                                   f'\tРасчетное значение {"больше критического - различия" if stu["A"] >= t_value else "меньше критического - различия не"} подтверждены статистически\n'
                                   )

                    print('done')
        else:
            message.error('Выберите метод и папки')


if __name__ == '__main__':
    DataProperties().configure_traits()
    CompareFolders().configure_traits()
'''folder1 = r'E:\A\fabric_\data/results/melanzhevye-dannye/меланжевые-данные'
folder2 = r'E:\A\fabric_\data/results/salatovye_dannye/салатовые данные'
samples = get_data_from_folders(folder1, folder2)
man, stu, text_compar, _, _ = compare_folders(folder1, folder2)
print(stu)'''

# folders = folder_selected1, folder_selected2
# compare_folders(*folders)
# folder1, folder2 = [os.path.basename(f) for f in folders]
