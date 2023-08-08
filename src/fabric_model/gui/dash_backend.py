import base64
import io
import dash
from dash import dcc, html, Input, Output, State, callback
import json
import tempfile
import numpy as np
from ..extract import read_stl, slice_dots
from ..tools import merge_slices_into_pd, extract_slices_df
from .plotly_main import get_fig, go

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
        figure=go.Figure(), 
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


#Output('dots-upload', 'data'),
 
@callback(Output('dots-upload', 'data'),
            Input('disk_height_input', 'value'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename'),
            State('upload-data', 'last_modified'),
            prevent_initial_call=True)
def update_output(disk_h, content, filename, last_modified):
    if content is None:
        return dash.no_update
    
    log.debug(f'loading file {filename}')
    points = parse_contents(disk_h, content, filename)
    #log.debug(points)
    return json.dumps(points.tolist())
    
    

def parse_contents(disk_h, contents, filename):
    content_type, content_string = contents.split(',')
    if not filename.endswith('stl'):
        return dash.no_update
    
    decoded = base64.b64decode(content_string)
    
    with open(r'temp\temp_file.stl','wb') as temp_file:
        temp_file.write(decoded)
    
    points = read_stl(r'temp\temp_file.stl', disk_h)
    return points
    


@callback(
    output=[Output('main-plot', 'figure')],
    inputs=[Input('dots-upload', 'data'),
    Input('top-h-slider', 'value'),
    Input('bottom-h-slider', 'value'),
    Input('rotate-degree-slider', 'value'),
    State('disk_height_input', 'value')],
    prevent_initial_call=True)
def update_figure(points_json, top_h, bottom_h, rot, disk_h):
    log.warning('UPDATING FIGURE')

    points = np.array(json.loads(points_json))

    slices, h = slice_dots(points, 4, top_h, bottom_h, disk_h, rot)
    slices_df = merge_slices_into_pd(slices, h)  # normalize data by passing it to this function
    slices, h = extract_slices_df(slices_df)

    fig = get_fig(points, slices, h)
    fig.update_layout(title='Dash Data Visualization', height=800)

    return [fig]







if __name__ == '__main__':
    app.run_server(debug=True)