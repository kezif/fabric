import json
import numpy as np
from fabric_model.extract import read_stl, slice_dots
from fabric_model.gui.plotly_main import get_fig
from fabric_model.tools import extract_slices_df, merge_slices_into_pd

filename= r'data\drap\2019_12_05-2-30-18.stl'
points = read_stl(filename, save_fig=False)


slices, h = slice_dots(points, 4, 25)
slices_df = merge_slices_into_pd(slices, h)
slices, h = extract_slices_df(slices_df)

fig = get_fig(points, slices, h)
fig.update_layout(title='Dash Data Visualization', height=800)