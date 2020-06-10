import pandas as pd
import os.path
import os
import sys
from collections import defaultdict
from scipy.stats import mannwhitneyu, ttest_ind
from traitsui import message
from traitsui.editors import CheckListEditor
from traits.has_traits import HasTraits
from traits.trait_types import Button, List, Directory, Enum, String
from pyface.api import FileDialog, OK
from traitsui.api import View, Item
from traitsui.group import HGroup


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


def exctr_data_ffiles(folder_path):
    list_models = []
    for file in os.listdir(folder_path):
        if file.endswith('.xlsx'):
            xlsx_path = os.path.join(folder_path, file)
            list_models.append(load_data_from_excel_old(xlsx_path))

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
    #print(text)
    return res, text


def t_student(a, b, cols):
    test_res = ttest_ind(a, b).statistic
    res = dict(zip(cols, test_res))
    text = f'Результаты теста c использованием критерия Стьюдента {res}'
    #print(text)
    return res, text


def get_data_from_folders(folder_path1, folder_path2):
    samples = defaultdict(None)  #
    samples[os.path.basename(folder_path1)] = exctr_data_ffiles(folder_path1)
    samples[os.path.basename(folder_path2)] = exctr_data_ffiles(folder_path2)
    return samples


def compare_folders(folder_path1, folder_path2):
    samples = get_data_from_folders(folder_path1, folder_path2)
    # print([(k, i) for k, i in samples.items()])

    res = [i for i in samples.values()]
    cols = res[0].columns  # ['a', 'b', 'c']
    text_compar = ('Сравнение между папками {} и {}'.format(*[i for i in samples.keys()]))
    man, text_man = mannwhit_test(*res, cols)
    stu, text_stu = t_student(*res, cols)
    return man, stu, text_compar


class CompareFolders(HasTraits):
    def __init__(self):
        super(CompareFolders, self).__init__()

    #methods_dic = {'mannwhit_test': mannwhit_test, 't_student': t_student}
    methods_list = ['U-критерий', 't-критерий']
    folder1 = Directory()
    folder2 = Directory()
    methods1 = List(editor=CheckListEditor(values=methods_list,), value=methods_list)
    methods2 = List(editor=CheckListEditor(values=methods_list,), value=methods_list)
    methods3 = List(editor=CheckListEditor(values=methods_list,), value=methods_list)
    method_str = String('Выбор методов для параметров:')
    alpha = Enum('.05', '.01')
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

        Item('compute'),
        width=.3, height=.2, resizable=True, title='Compare folders'
    )

    def _compute_fired(self):
        print(self.alpha)
        if self.methods1 != [] and self.methods2 != [] and self.methods3 != [] and self.folder1 is not '' and self.folder2 is not '':
            man, stu, text = compare_folders(self.folder1, self.folder2)

            wildcard = '*.txt'
            default_path = os.path.dirname(self.folder1) + '/' + os.path.basename(self.folder1) + os.path.basename(self.folder2) + '.txt'
            dialog = FileDialog(title='Save results',
                                action='save as', wildcard=wildcard,
                                default_path=default_path
                                )
            if dialog.open() == OK:
                with open(dialog.path, 'w', encoding="utf-8") as file:
                    file.write(text + '\n')
                    if self.methods_list[0] in self.methods1:  # revritre in better form
                        file.write(f'{man["D"]:.3f}\n')
                    if self.methods_list[1] in self.methods1:
                        file.write(f'{stu["D"]:.3f}\n')
                    if self.methods_list[0] in self.methods2:
                        file.write(f'{man["R2"]:.3f}\n')
                    if self.methods_list[1] in self.methods2:
                        file.write(f'{stu["R2"]:.3f}\n')
                    if self.methods_list[0] in self.methods3:
                        file.write(f'{man["A"]:.3f}\n')
                    if self.methods_list[1] in self.methods3:
                        file.write(f'{stu["A"]:.3f}\n')
        else:
            message.error('Выберите метод и папки')


CompareFolders().configure_traits()

# folders = folder_selected1, folder_selected2
# compare_folders(*folders)
# folder1, folder2 = [os.path.basename(f) for f in folders]
