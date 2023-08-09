
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
    


