import base64
import io
import dash
from dash import dcc, html, Input, Output, State, callback
import json
import tempfile
import numpy as np
from ..extract import read_stl_center_info, slice_dots
from ..tools import merge_slices_into_pd, extract_slices_df
from .plotly_main import get_fig_center_info, go, missing_data_layout

from ..logger import get_logger


log = get_logger('log.log')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets) # call flask server

# run following in command
# gunicorn graph:app.server -b :8000




slider_tooltip = {"placement": "bottom", "always_visible": True}

app.layout = html.Div([
    
    html.Div(children=[
        dcc.Graph(
        id='main-plot',
        figure=go.Figure(layout=missing_data_layout), 
    ),
    ], style={'flex': 1}, className="eight columns"),

    html.Div(children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'margin-bottom': '20px',
            },
            
        ),
        html.Div([
        html.Label('top H'),
        dcc.Slider(0, 50, 1,
                value=25,
                    marks={0: '0', 50: '50'}, 
                id='top-h-slider', 
                tooltip = slider_tooltip, 
                ),
        html.Label('bottom H'),           
        dcc.Slider(0, 50, 1, 
                value=15,
                marks={0: '0', 50: '50'}, 
                id='bottom-h-slider', 
                tooltip = slider_tooltip,
                ),
        html.Label('disk height'),
        dcc.Input(id='disk_height_input',
                value=3,
                type='number',
                placeholder='disk height',
            ),
        dcc.Checklist(options=['Show center info'], value=['Show center info'], id='show_info_checklist'),
        html.Label('rotate by °'),  
        dcc.Slider(-3.14, 3.14, 0.01,
                    value=0., 
                    marks={-3.14: '-3.14°', 3.14: '3.14°'}, 
                    id='rotate-degree-slider', 
                    tooltip = slider_tooltip,
                    ),   
    ], style={'margin-bottom': '20px',}),          
        html.Div([
            html.Hgroup([
            html.Label('disk height'),
            dcc.Input(id='disk_height_r',
                    type='number',
                    placeholder='0',
                ),
            ]),
            html.Hgroup([
                html.Label('disk sample'),
                dcc.Input(id='disk_height_R', 
                        type='number',
                        placeholder='0',
                    ),
            ]),
            ], style={'margin-bottom': '20px',}
            ), 

        html.Div([
        html.Button('build-models', id='build-models-button', n_clicks=0, style={'margin-bottom': '10px',
                                 'verticalAlign': 'middle'}),
        html.Br(),
        html.Button('save-data', id='save-data-button', n_clicks=0, style={'margin-bottom': '10px',
                                 'verticalAlign': 'middle'}),
        dcc.Store(id='dots-upload', storage_type='local'),
        html.Label('test', id='test-label')
        ], style={'verticalAlign': 'middle'}),
        ], style={'padding': 30, 'flex': 1, 'max-width': 300}, className="four columns"
    ),


], style={'display': 'flex', 'flex-direction': 'row'})


@callback(Output('main-plot', 'figure', allow_duplicate=True),
            [State('main-plot', 'figure'),
            Input('show_info_checklist', 'value')],
            prevent_initial_call=True)   
def visible_center(fig, checkbox, *args):
    log.debug(f'{checkbox=}')

    if len(fig['data']) == 0: # if empty
        return dash.no_update
    fig['layout']['uirevision'] = 'some_value'
    fig['data'][6]['visible'] = bool(checkbox)  # when not selected - list is empty, therefor False
    fig['data'][7]['visible'] = bool(checkbox)
    return fig


#Output('dots-upload', 'data'),
 
@callback(Output('dots-upload', 'data'),
            Input('disk_height_input', 'value'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'),
            prevent_initial_call=True)
def update_output(disk_h, content, filename, last_modified):
    if content is None:
        log.debug(f'file is empty {filename}')
        return dash.no_update
    
    log.debug(f'loading file {filename}')
    points, center_info = parse_contents(disk_h, content, filename)
    #log.debug(points)
    d = dict(points=points.tolist(), center_layer=center_info[0].tolist(), dx=center_info[1], dy=center_info[2], circle_h=center_info[3])
    return json.dumps(d)



def parse_contents(disk_h, contents, filename):
    log.debug(f'Loading file  {filename}')
    content_type, content_string = contents.split(',')
    if not filename.endswith('stl'):
        return dash.no_update
    
    log.debug(f'Decoding base64 string and reading to file {filename}')
    decoded = base64.b64decode(content_string)
    with open(r'temp\temp_file.stl','wb') as temp_file:
        temp_file.write(decoded)
    
    log.debug(f'Reading stl file {filename}')
    points, center_data = read_stl_center_info(r'temp\temp_file.stl', disk_h, save_fig=False)
    log.debug(f'Returning points {filename}')
    return points, center_data
    

# TODO update only slices when slice parameters updated
# TODO save figure viewpoint when changing it
# TODO store input's file data as base64 string. Read directly from it 

@callback(
    output=[Output('main-plot', 'figure', allow_duplicate=True)],
    inputs=[Input('dots-upload', 'data'),
    State('top-h-slider', 'value'),
    State('bottom-h-slider', 'value'),
    State('rotate-degree-slider', 'value'),
    State('disk_height_input', 'value')],
    prevent_initial_call=True)
def update_figure(points_json, top_h, bottom_h, rot, disk_h):
    log.debug('Updating figure')

    if points_json is None:
        log.debug('Points data is empty')
        return dash.no_update
    log.debug('Loading points data')
    d = json.loads(points_json)
    points = np.array(d['points'])
    center_layer = np.array(d['center_layer'])
    dx = float(d['dx'])
    dy = float(d['dy'])
    circle_h = float(d['circle_h'])

    log.debug('Caltulating slices')
    slices, h = slice_dots(points, 4, top_h, bottom_h, disk_h, rot)
    slices_df = merge_slices_into_pd(slices, h)  # normalize data by passing it to this function
    slices, h = extract_slices_df(slices_df)

    log.debug('Acquiring figure')
    fig = get_fig_center_info(points, slices, h, center_layer, dx, dy, circle_h)
    fig.update_layout(title='Dash Data Visualization', height=800)
    log.debug('Returning figure')
    return [fig]


'''@callback(
    output=[Output('main-plot', 'figure')],
    inputs=[State('dots-upload', 'data'),
    Input('top-h-slider', 'value'),
    Input('bottom-h-slider', 'value'),
    Input('rotate-degree-slider', 'value'),
    Input('disk_height_input', 'value'),
    State('main-plot', 'figure')],
    prevent_initial_call=True)
def update_slices(points_json, top_h, bottom_h, rot, disk_h, fig):
    #update only slices on existing plot
    log.debug('UPDATING SLICES')
    return fig'''




if __name__ == '__main__':
    app.run_server(debug=True)