import matplotlib
from os import getcwd
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import File, Button, Range, Int, Float, String
from traits.etsconfig.api import ETSConfig
from pyface.api import FileDialog, OK
from traitsui.api import View, Item, HGroup, Group, VGroup
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from traits.api import Instance
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory
import os
matplotlib.use('Qt4Agg')
ROOT_DIR = getcwd() + '\\'
ETSConfig.toolkit = 'qt4'
    
class _MPLFigureEditor(Editor):  # 2 classes for ploting
    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # matplotlib commands to create a canvas
        mpl_canvas = FigureCanvas(self.value)
        return mpl_canvas


class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor