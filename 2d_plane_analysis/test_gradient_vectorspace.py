import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

grid_size = 20
lin = np.linspace(-10, 10, grid_size)
U, V = np.meshgrid(lin, lin)
grid = np.column_stack([U.ravel(), V.ravel()])

grid_softmax = softmax(grid, axis=-1)
red_true = np.tile([1, 0], (len(grid), 1))
dL_dZ_red = grid_softmax - red_true
vector_update_red = -dL_dZ_red

fig = go.Figure(
    data=go.Cone(
        x=grid[:, 0],
        y=grid[:, 1],
        z=np.zeros(len(grid)),
        u=vector_update_red[:, 0],
        v=vector_update_red[:, 1],
        w=np.zeros(len(grid)),
        sizemode="absolute",
        sizeref=2,
        colorscale="Reds",
        showscale=False,
        anchor="tail"
    )
)

fig.update_layout(
    scene=dict(
        zaxis=dict(visible=False),
        xaxis=dict(range=[-10, 10]),
        yaxis=dict(range=[-10, 10])
    ),
    template="plotly_dark",
    width=1000,
    height=1000
)

fig.show()
