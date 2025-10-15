import numpy as np
from scipy.optimize import leastsq
import plotly.graph_objects as go
import os
from sklearn.manifold import TSNE
from PIL import Image
import io
from scipy.interpolate import griddata
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import h5py
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import pickle
import matplotlib.colors as mcolors
import plotly.io as pio

pio.renderers.default = "browser"

np.random.seed(40)


# -----------------------------
# Global parameters
# -----------------------------
hamming_threshold = 1
first_frame = 0
nr_global_frames = 3000

nr_digits = 10



nr_neurons = 300
nr_inner_neurons = nr_neurons
activation = "relu"
nr_epochs_train = 7001
try_nr = f'plane_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0
chosen_model = "PCA"
run_nr = 34

# -----------------------------
# Helper functions
# -----------------------------

def get_interpolated_matrix(current_layer, next_layer, nr_interp_frames):
    interp_matrix = np.zeros((nr_interp_frames, current_layer.shape[0], current_layer.shape[1]))
    for i in range(current_layer.shape[0]):
        for j in range(current_layer.shape[1]):
            point_trace = np.linspace(current_layer[i][j], next_layer[i][j], nr_interp_frames)
            interp_matrix[:, i, j] = point_trace
    return interp_matrix

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0, x)

def sigmoid_force_quad(z):
    return np.where(z <= 0, -1, 1)

def bitstring_labels(X_hd, offset_value=0):
    """Return list of bitstrings for each sample."""
    return [''.join(str(int(b)) for b in (row > offset_value)) for row in X_hd]

def quadrant_clusters(X_hd, offset_value):
    labels = [''.join(str(int(b)) for b in (row > offset_value)) for row in X_hd]
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters, labels

def compute_hamming_graph_lsh(labels, max_hamming=15):
    n = len(labels)
    print(n, "nr_labels")

    max_len = max(len(l) for l in labels)
    binary_matrix = np.zeros((n, max_len), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, bit in enumerate(label):
            binary_matrix[i, j] = int(bit)

    dists = squareform(pdist(binary_matrix, metric='hamming')) * max_len
    edges = np.argwhere((dists <= max_hamming) & (dists > 0))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edge_distances = {}
    for i, j in edges:
        dist = int(dists[i, j])
        G.add_edge(i, j)
        edge_distances[(labels[i], labels[j])] = dist
        edge_distances[(labels[j], labels[i])] = dist

    return G, labels, edge_distances, dists

# -----------------------------
# Global color assignment
# -----------------------------

def assign_global_colors(X_latent_all, offset_value=0.5, seed=40):
    """Scan all frames to find all unique labels, assign random persistent colors."""
   # np.random.seed(seed)
    global_labels = set()

    print("Scanning all frames for global labels...")
    for frame in X_latent_all:
        frame_labels = [''.join(str(int(b)) for b in (row > offset_value)) for row in frame]
        global_labels.update(frame_labels)

    global_labels = sorted(global_labels)
    n_labels = len(global_labels)
    rng = np.random.default_rng(seed)
    colors = [mcolors.to_hex(rng.random(3)) for _ in range(n_labels)]
    label_to_color = dict(zip(global_labels, colors))

    print(f"Total unique labels encountered: {n_labels}")
    return label_to_color

def get_axis_ranges(X_latent_fit):
    print(X_latent_fit.shape)

    return {
        'xmin': np.min(X_latent_fit[:, :, 0]),
        'xmax': np.max(X_latent_fit[:, :, 0]),
        'ymin': np.min(X_latent_fit[:, :, 1]),
        'ymax': np.max(X_latent_fit[:, :, 1]),
    }
    


def generate_spiral(n_points_per_class=100, noise=0.02, n_turns=3):
    n = np.arange(0, n_points_per_class)
    theta = np.linspace(0, n_turns * np.pi, n_points_per_class)
    r = np.linspace(0.0, 1.0, n_points_per_class)

    # Two opposite spirals (two classes)
    x1 = r * np.sin(theta)
    y1 = r * np.cos(theta)
    x2 = -r * np.sin(theta)
    y2 = -r * np.cos(theta)

    data = np.vstack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1)
    ])
    labels = np.hstack([
        np.zeros(n_points_per_class, dtype = np.int32),
        np.ones(n_points_per_class, dtype = np.int32)
    ])

    # Add Gaussian noise
    data += np.random.randn(*data.shape) * noise

    return data, labels
def softmax(x, axis=-1):
    # Numerically stable softmax
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def plot_training_spacebent_all_smooth():
    global try_nr, nr_neurons, hamming_threshold, chosen_model, run_nr


   
    selected_layer = 0
    predict_layer = 1

    # load latent states

   # X_latent_train = np.load(f'trainings2/training_digits_{try_nr}/X_bent_output_layers_{selected_layer}.npy')
   # X_latent_space = np.load(f'trainings2/training_digits_{try_nr}/space_bent_output_layers_{selected_layer}.npy')
    X_pred_train = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
   # X_pred_space = np.load(f'trainings2\\training_digits_{try_nr}\\space_bent_output_layers_{predict_layer}.npy')

    x_train, y_train = generate_spiral(n_points_per_class=1000, noise=0.03, n_turns=6)


   # print(len(X_latent_space), "len(X_latent)")
   # exit()

    # make output dirs
    animation_folder_path = f"animations2/latent_graph_global/{try_nr}"
    layer_folder_path = f"{animation_folder_path}/layer_nr_{selected_layer}_hamm_{hamming_threshold}_nointerp_{chosen_model}_{run_nr}"
    os.makedirs(layer_folder_path, exist_ok=True)

    # PCA on last frame

   # X_latent_space = relu(X_latent_space)


   # pca = PCA(n_components=2).fit(X_latent_space[-1])

    # persistent label-color mapping
    label_to_color = {}
    rng = np.random.default_rng(42)

    # random background space
    #random_points_base = sigmoid(np.random.uniform(-30, 30, (100000, nr_neurons)))
    #random_points_base_pca = pca.transform(random_points_base)

    

    # grid_size = int(np.sqrt(300,300))
    # lin = np.linspace(-4, 4, grid_size)
    # U, V = np.meshgrid(lin, lin)
    # U_flat = U.ravel()
    # V_flat = V.ravel()

    # grid = np.column_stack(U_flat, V_flat)
    # grid_softmax = softmax(grid, axis = -1)

    # red_true = np.array([ [1,0] * len(grid) ])
    # green_true = np.array([ [0,1] * len(grid) ])

    # dL_dZ_red = grid_softmax - red_true
    # dL_dZ_green = grid_softmax - green_true


    # vector_update_red = - dL_dZ_red
    # vector_update_green = - dL_dZ_green

    # fig = go.Figure()

    # for i, vector_update_red in enumerate(vector_update_red):

    #         fig.add_trace(go.Scatter(
    #             x=[grid[i][0], grid[i][0] + vector_update_red[0]/100], 
    #             y=[grid[i][1], grid[i][1] + vector_update_red[1]/100], 
    #             mode='lines',
    #             line=dict(width=2, color="red"),
    #             showlegend=False,
    #             opacity=1
    #         ))

    # for i, vector_update_green in enumerate(vector_update_green):

    #     fig.add_trace(go.Scatter(
    #         x=[grid[i][0], grid[i][0] + vector_update_green[0]/100], 
    #         y=[grid[i][1], grid[i][1] + vector_update_green[1]/100], 
    #         mode='lines',
    #         line=dict(width=2, color="green"),
    #         showlegend=False,
    #         opacity=1
    #     ))

    # fig_layer.update_layout(template="plotly_dark",)
    # fig.show()

    # exit()




    axis_ranges = get_axis_ranges(X_pred_train)


    C = y_train.max() + 1
    y_true_one_hot = np.eye(C)[y_train]




    

    frame_nr = 0
    for ii in range(0, len(X_pred_train), 1):


        z2 = X_pred_train[ii]

        y_pred_train = softmax(z2, axis = -1)

        y_true = y_true_one_hot
        
        dL_dZ = y_pred_train - y_true

        vectors_update = -dL_dZ

        print(vectors_update)


        fig_layer = go.Figure()


        colors_map_labels = {0:"red", 1:"green"}


        fig_layer.add_trace(go.Scatter(
            x=z2[:,0], 
            y=z2[:,1], 
            mode='markers',
            marker=dict(size=7, color=y_train, colorscale = [[0, "red"], [1, "green"]], opacity=0.5),
            showlegend=False
        ))

        # for i, vector_update in enumerate(vectors_update):

        #     fig_layer.add_trace(go.Scatter(
        #         x=[z2[i][0], z2[i][0]+ vector_update[0]/100], 
        #         y=[z2[i][1], z2[i][1] + vector_update[1]/100], 
        #         mode='lines',
        #         line=dict(width=2, color=colors_map_labels[y_train[i]]),
        #         showlegend=False,
        #         opacity=1
        #     ))


        fig_layer.add_trace(go.Scatter(
                x=[-0.4, 0.4], 
                y=[-0.4, 0.4], 
                mode='lines',
                line=dict(width=2, color="white"),
                showlegend=False,
                opacity=0.5
            ))
        

        fig_layer.update_layout(template="plotly_dark",)
     #   fig_layer.show()
    #    exit()



        # fig_layer.update_layout(
                
        #          xaxis=dict(
        #         #     showgrid=False,
        #         #     showticklabels=False,
        #         #     showline=False,
        #         #     zeroline=False,
        #         #     ticks="",
        #              range=[axis_ranges["xmin"], 1]#axis_ranges["xmax"]]
        #          ),
        #         yaxis=dict(
        #         #     showgrid=False,
        #         #     showticklabels=False,
        #         #     showline=False,
        #         #     zeroline=False,
        #         #     ticks="",
        #              range=[-1,axis_ranges["ymax"]]
        #          )
        #     )
        
        fig_layer.update_layout(
                
                 xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    showline=False,
                    zeroline=False,
                    ticks="",
                 #    range=[-2.5, 2.5]
                 ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    showline=False,
                    zeroline=False,
                    ticks="",
                 #    range=[-2.5, 2.5]
                 )
            )

        fig_layer.update_layout(
            template="plotly_dark",
            width=2500,
            height=1500,
         #   xaxis=dict(visible=False),
          #  yaxis=dict(visible=False),
           # title="Reconstructed Original 2D Grid"
        )

        fig_layer.write_image(f"{layer_folder_path}/image_{frame_nr:05}.png")
        frame_nr += 1

# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":

    
    #tf.random.set_seed(40)
    plot_training_spacebent_all_smooth()
