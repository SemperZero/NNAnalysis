import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.special import expit  # Sigmoid function
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.express as px
import os

try_nr = 0
n_dim = 7

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val!=max_val:
        normalized = (arr - min_val) / (max_val - min_val)
        scaled = 2 * normalized - 1
    else:
        scaled = arr
    
    return scaled

def apply_grayshift(X, n):
    x = []
    for i in np.linspace(0, 1, n):
        x.append(i * X)
    x = np.vstack(x) 
    return x

MAX_SCALE = 100

random_points = np.random.uniform(-1, 1 ,  (3000, n_dim))

pca = PCA(n_components=3)
pca = pca.fit(sigmoid(random_points * MAX_SCALE))

random_points_grayshift = apply_grayshift(random_points, 100)

animation_folder_path = f"animations\\sigmoid_{n_dim}_try_{try_nr}"# nr_interp_frames, skip_step
if not os.path.isdir(animation_folder_path):
    os.makedirs(animation_folder_path)


degrees_per_sec = 10
degrees_per_iter = degrees_per_sec / 60

frame_nr = 0

speed = 0.05
max_val = 0.1
acc = 0
while max_val < MAX_SCALE:
    print(max_val)
    
    random_points_scaled = sigmoid(random_points_grayshift * max_val)

            


    fig = go.Figure()

    pca_result = pca.transform(random_points_scaled)

    pca_result[:,0] = min_max_normalize(pca_result[:,0])
    pca_result[:,1] = min_max_normalize(pca_result[:,1])
    pca_result[:,2] = min_max_normalize(pca_result[:,2])

    fig.add_trace(go.Scatter3d(
        x=pca_result[:, 0], 
        y=pca_result[:, 1], 
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.5, color = list(range(len(pca_result))), colorscale = "ylorrd_r")
    ))

    
    angle = frame_nr * degrees_per_iter * np.pi/180
    vertical_angle = 45*np.pi/180
    r = 1.5

    fig.update_layout(scene_camera=dict(eye=dict(x=r*np.cos(vertical_angle)*np.cos(angle), y=r*np.cos(vertical_angle)*np.sin(angle), z=r*np.sin(vertical_angle))))

    fig.update_layout(template="plotly_dark", width = 1500, height = 1500)

    fig.write_image(f"{animation_folder_path}\\image_{frame_nr:05}.png")
    frame_nr+=1

    
    acc = 0
    if max_val < 7:
        acc = 0
    elif speed < 0.3:
        acc = 0.02
   # else:
    #    acc = 0

    speed += acc

    
    max_val += speed / 4
    #fig.show()
