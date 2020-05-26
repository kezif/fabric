import traitsui

from gui_libs import *
from f_extract import *
from f_tools import *
from f_save import *
from f_model import format_ax, make_models_from_df


class DataExtract(HasTraits):

    def __init__(self):
        super(DataExtract, self).__init__()
        self.dots = None
        self.slices_df = None
        self.figure.set_size_inches(5, 5)
        self.data_dict = {}
        ax = self.figure.add_subplot(111, projection='polar')
        ax.set_title('No data', loc='center')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    file_name = File(value='', filter=['*.stl'])
    open = Button('Open...')
    height = Range(0, 50, 25, label='top H')
    top_h = Float(3, label='Disk height ')
    rsmoll = Int(0, label='r диска')
    rbig = Int(0, label='r пробы')
    rot = Range(-3.14, 3.14, 0.0, label='Rotate by, rad')
    figure = Instance(Figure, ())
    save = Button('Build models and save')
    save_data = Button('Save data')
    start_from_h = Range(0, 50, 15, label='bottom H')
    circle_plot = Button('plot')

    main_group = Group(
        HGroup(
            Item('open', show_label=False),
            '_',
            Item('file_name', style='readonly', springy=True)
        ),
        Item('figure', editor=MPLFigureEditor(),
             show_label=False),
        Item('height'), Item('start_from_h'), 
        HGroup(
            Item('top_h'),
            Item('circle_plot', show_label=False),
            ), 
        Item('rot'),
        '10', '_', '5',
        HGroup(
            Item('rsmoll'),
            Item('rbig'),
        ),
    )
    save_group = Group(
        '5', '_', '5',
        HGroup('_',
               Item('save', springy=True,  show_label=False),
               ),
        HGroup('_',
               Item('save_data', springy=True, show_label=False)
               ),
    )
    save_item = Item('save', springy=True)

    traits_view = View(  # view for multiple slices.
        main_group,
        save_group,
        '_',
        width=.3, height=.8, resizable=True, title='Data extraction 2.'
    )

    def _circle_plot_fired(self):
        layer = layer_from_dots(self.dots, self.top_h)
        circle_fit(layer, show_info=True)

    def _save_fired(self):
        extns = ['*.xlsx', ]  # seems to handle only one extension...
        wildcard = '|'.join(extns)
        name = os.path.splitext(os.path.basename(self.file_name))[0]  # extract name from path "/dir/file.jpg" -> "file"
        default_path = '\\'.join([ROOT_DIR, 'data', 'results', name])  #
        default_path += f'_({self.slices_df.columns.values[0]}-{self.slices_df.columns.values[-2]}){self.top_h}.xlsx'
        dialog = FileDialog(title='Save results',
                            action='save as', wildcard=wildcard,
                            default_path=default_path)
        if dialog.open() == OK:
            path = dialog.path
            if path.endswith('.xlsx'):
                pass
            else:
                path = path + '.xlsx'
            self.figure.savefig('temp//slices.png', dpi=100)
            sh_a = shadow_area(self.dots)
            self.data_dict = make_models_from_df(self.slices_df)
            save_results(self.data_dict, sh_a, self.rsmoll, self.rbig, path)  # save not only data, but complete spreadsheet, w models, pics and etc
        print('Done.')    

    def _save_data_fired(self):
        extns = ['*.csv', ]  # seems to handle only one extension...
        wildcard = '|'.join(extns)
        name = os.path.splitext(os.path.basename(self.file_name))[0]  # extract name from path "/dir/file.jpg" -> "file"
        default_path = '\\'.join([ROOT_DIR, 'data', 'results', name])  #
        default_path += f'_({self.slices_df.columns.values[0]}-{self.slices_df.columns.values[-1]}).csv'
        dialog = FileDialog(title='Save data',
                            action='save as', wildcard=wildcard,
                            default_path=default_path)
        if dialog.open() == OK:
            path = dialog.path
            if path.endswith('.csv'):
                pass
            else:
                path = path + '.csv'
            save_slices_df(self.slices_df, path)

    def _open_fired(self):
        """ Handles the user clicking the 'Open...' button.
        """
        extns = ['*.stl', ]  # seems to handle only one extension...
        wildcard = '|'.join(extns)

        dialog = FileDialog(title='Select file',
                            action='open', wildcard=wildcard,
                            default_path=ROOT_DIR)
        if dialog.open() == OK:
            self.file_name = dialog.path
            self.update_dots()
        else:
            traitsui.message.error('can\'t excract data from file')

    @on_trait_change(['top_h'])
    def update_dots(self):
        self.dots = read_stl(self.file_name, self.top_h)
        self.update_slices()

    @on_trait_change(['height', 'rot', 'skip_each_n_dot', 'start_from_h'])
    def update_slices(self):
        if self.dots is None:
            traitsui.message.error('File is not loaded')
        slices, h = slice_dots(self.dots, 4, self.height, self.start_from_h, self.top_h, self.rot)
        self.slices_df = merge_slices_into_pd(slices, h)
        self.plot_()

    def plot_(self):
        ax = self.figure.axes[0]
        plot_df_on_ax(self.slices_df, ax)
        format_ax(ax, self.slices_df.iloc[:, -1].values)
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()


if __name__ == "__main__":
    try:
        os.makedirs('data//results', exist_ok=True)
        os.makedirs('temp', exist_ok=True)
    except OSError:
        pass
    DataExtract().configure_traits()
