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
import numpy as np
import os
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # ensures FigureCanvasAgg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

pio.renderers.default = "browser"

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
nr_epochs_train = 7000
try_nr = f'plane_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0
chosen_model = "PCA"
run_nr = 17

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

def plot_training_spacebent_all_smooth_matplotlib(pixel_scale=2):
    global try_nr, nr_neurons, hamming_threshold, chosen_model, run_nr

    selected_layer = 0
    predict_layer = 1
    X_latent = np.load(f'trainings2/training_digits_{try_nr}/X_bent_output_layers_{selected_layer}.npy')
    X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')

    animation_folder_path = f"animations2/latent_graph_global/{try_nr}"
    layer_folder_path = f"{animation_folder_path}/layer_nr_{selected_layer}_hamm_{hamming_threshold}_nointerp_{chosen_model}_{run_nr}"
    os.makedirs(layer_folder_path, exist_ok=True)

    pca = PCA(n_components=3).fit(X_latent[-1])
    label_to_color = {}
    rng = np.random.default_rng(42)

    grid_size = int(np.sqrt(X_latent[0].shape[0]))
    lin = np.linspace(-1, 1, grid_size)
    U, V = np.meshgrid(lin, lin)
    U_flat, V_flat = U.ravel(), V.ravel()

    frame_nr = 0
    for ii in range(first_frame, first_frame + nr_global_frames):
        X_latent_ii = X_latent[ii]
        pred_labels = np.argmax(X_pred[ii], axis = 1)
        #clusters, labels = quadrant_clusters(X_latent_ii, 0.5)

        # # persistent color mapping
        # for lbl in labels:
        #     if lbl not in label_to_color:
        #         label_to_color[lbl] = mcolors.to_hex(rng.random(3))
        #point_colors = np.array([mcolors.to_rgb(label_to_color[l]) for l in pred_labels])


        color_map = {
            0: (1, 0, 0),  # red
            1: (0, 0, 1),  # blue
            2: (0, 1, 0),  # green
        }

        point_colors = np.array([color_map.get(l, (0.5, 0.5, 0.5)) for l in pred_labels])



        rgb_matrix = point_colors.reshape(grid_size, grid_size, 3)

        # scale pixels (each pixel -> n×n block)
        if pixel_scale > 1:
            rgb_matrix = np.kron(rgb_matrix, np.ones((pixel_scale, pixel_scale, 1)))

        img = Image.fromarray((rgb_matrix * 255).astype(np.uint8))
        img.save(f"{layer_folder_path}/image_{frame_nr:05}.png")

        frame_nr += 1


# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":

    np.random.seed(42)
    plot_training_spacebent_all_smooth_matplotlib(pixel_scale=4)
