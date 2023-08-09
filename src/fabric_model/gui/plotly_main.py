
import plotly.graph_objects as go


from stl import mesh
import numpy as np
from ..tools import pol2cart3d

missing_data_layout = {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
        "annotations": [
            {
                "text": "No matching data found",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }
            }
        ]
    }

def get_fig(points:np.array, slices: list[np.array], h:list) -> go.Figure:

  

    faces_index = np.arange(len(points))
    fig = go.Figure(data=[
        go.Mesh3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            # i, j and k give the vertices of triangles
            i = faces_index[0::3],
            j = faces_index[1::3],
            k = faces_index[2::3],
            opacity = 0.8,
            name='fabric',
            color='#7F8076'
        )
    ])


    style_dict = dict(showlegend=True, marker_symbol='circle-open', mode='lines+markers',marker_size=5,)
    for slice, h_ in zip(slices, h):  # iterate over slices
        slice_x, slice_y, _ = pol2cart3d(slice[:,1], slice[:,0], 0)   # internaly slice data it stored in polar coordinates, convert it back to cartiseian

        if h_ == 'shadow':  # in case of shadow plot I'll want projection to z axis. Hopefully library have this functionality. It's a little bit hacky but original data would be plotted with 0 opacity. SO we would see only
            fig.add_trace(go.Scatter3d(x=slice_x, y=slice_y, z=np.full((slice_x.shape),-float(0)), 
                                    name=f'Shadow',
                                    projection={'z':{'show':True}},
                                    opacity=.0,
                                    **style_dict
                                    ))
            continue

        
        fig.add_trace(go.Scatter3d(x=slice_x, y=slice_y, z=np.full((slice_x.shape),-float(h_)), 
                                    name=f'slice at {h_}', 
                                    **style_dict                                
                            ),
                        )
               
    fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=0.5))

    return fig
    

def get_fig_center_info(points:np.ndarray, slices: list[np.ndarray], h:list, top_layer_dots: np.ndarray, dx:float, dy:float, disk_h:int) -> go.Figure:
    fig = get_fig(points, slices, h)
    fig.add_trace(go.Scatter3d(x=top_layer_dots[:,0], y=top_layer_dots[:,1], z=np.full((len(top_layer_dots)),-disk_h),
                               **dict(name=f'slice of which data is centered', 
                                        showlegend=False,
                                        marker_symbol='diamond-open',
                                        mode='markers',
                                        marker_size=2,
                                        opacity=0.1,
                                        marker_color='#5D63A2')))
    fig.add_trace(go.Scatter3d(x=[dx], y=[dy], z=[-disk_h],
                               **dict(name=f'Center', 
                                        showlegend=False,
                                        marker_symbol='cross',
                                        mode='markers',
                                        marker_size=20,
                                        opacity=0.5,
                                        marker_color='#5D63A2')))
    return fig

def get_empty_fig(n_slices=4):
    fig = go.Figure(data=[
        go.Mesh3d(
            opacity = 0.8,
            name='fabric',
            color='#7F8076'
        )
    ])


    style_dict = dict(showlegend=True, marker_symbol='circle-open', mode='lines+markers',marker_size=5,)
    for _ in range(n_slices):  # iterate over slices
        fig.add_trace(go.Scatter3d(name=f'Shadow',
                                projection={'z':{'show':True}},
                                opacity=.0,
                                **style_dict
                                ))
   

        
    fig.add_trace(go.Scatter3d(name=f'slice at {_}',**style_dict),)
               
    fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=0.5))

    fig.add_trace(go.Scatter3d(name=f'slice of which data is centered', 
                                        showlegend=False,
                                        marker_symbol='diamond-open',
                                        mode='markers',
                                        marker_size=2,
                                        opacity=0.1,
                                        marker_color='#5D63A2'))
    fig.add_trace(go.Scatter3d(name=f'Center', 
                                        showlegend=False,
                                        marker_symbol='cross',
                                        mode='markers',
                                        marker_size=20,
                                        opacity=0.3,))
    return fig
                  