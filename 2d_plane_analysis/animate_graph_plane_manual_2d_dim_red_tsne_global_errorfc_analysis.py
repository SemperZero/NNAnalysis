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

# -----------------------------
# Global parameters
# -----------------------------
hamming_threshold = 1
first_frame = 0
nr_global_frames = 3000

nr_digits = 10



nr_neurons = 303
nr_inner_neurons = nr_neurons
activation = "relu"
nr_epochs_train = 7000
try_nr = f'plane_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0
chosen_model = "PCA"
run_nr = 28

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

def assign_global_colors(X_latent_all, offset_value=0.5, seed=42):
    """Scan all frames to find all unique labels, assign random persistent colors."""
    np.random.seed(seed)
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
    
# -----------------------------
# Main plotting function
# -----------------------------
def plot_training_spacebent_all_smooth():
    global try_nr, nr_neurons, hamming_threshold, chosen_model, run_nr


   
    selected_layer = 0
    predict_layer = 1

    # load latent states
    X_latent = np.load(f'trainings2/training_digits_{try_nr}/X_bent_output_layers_{selected_layer}.npy')
    X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
    
    print(len(X_latent), "len(X_latent)")
   # exit()

    # make output dirs
    animation_folder_path = f"animations2/latent_graph_global/{try_nr}"
    layer_folder_path = f"{animation_folder_path}/layer_nr_{selected_layer}_hamm_{hamming_threshold}_nointerp_{chosen_model}_{run_nr}"
    os.makedirs(layer_folder_path, exist_ok=True)

    # PCA on last frame

    X_latent = relu(X_latent)


    pca = PCA(n_components=2).fit(X_latent[-1])

    # persistent label-color mapping
    label_to_color = {}
    rng = np.random.default_rng(42)

    # random background space
    random_points_base = sigmoid(np.random.uniform(-30, 30, (100000, nr_neurons)))
    random_points_base_pca = pca.transform(random_points_base)

    

    grid_size = int(np.sqrt(X_latent[0].shape[0]))
    lin = np.linspace(-1, 1, grid_size)
    U, V = np.meshgrid(lin, lin)
    U_flat = U.ravel()
    V_flat = V.ravel()


    axis_ranges = get_axis_ranges(X_pred)

    frame_nr = 0
    for ii in range(first_frame, first_frame + nr_global_frames, 1):
        # X_latent_ii = X_latent[ii]
        # clusters, labels = quadrant_clusters(X_latent_ii, 0.0)

        # for lbl in labels:
        #     if lbl not in label_to_color:
        #         label_to_color[lbl] = mcolors.to_hex(rng.random(3))

        # proj3d_mesh = pca.transform(X_latent_ii)
        # point_colors = [label_to_color[lbl] for lbl in labels]


        pred_labels = np.argmax(X_pred[ii], axis = 1)




        fig_layer = go.Figure()

        # fig_layer.add_trace(go.Scatter(
        #     x=U_flat, 
        #     y=V_flat, 
        #     mode='markers',
        #     marker=dict(size=6, color=point_colors, opacity=0.5),
        #     showlegend=False
        # ))


        fig_layer.add_trace(go.Scatter(
            x=X_pred[ii][:,0], 
            y=X_pred[ii][:,1], 
            mode='markers',
            marker=dict(size=4, color=pred_labels, colorscale = [[0, "red"], [1, "green"]], opacity=0.5),
            showlegend=False
        ))


        # fig_layer.add_trace(go.Scatter(
        #     x=proj3d_mesh[:,0], 
        #     y=proj3d_mesh[:,1], 
        #     mode='markers',
        #     marker=dict(size=6, color=point_colors, opacity=0.5),
        #     showlegend=False
        # ))

        

    #     label_list = list(set(labels))
    #     G_indices, _, edge_distances, dist_matrix = compute_hamming_graph_lsh(
    #         label_list, max_hamming=hamming_threshold
    #     )
                
    #     # Convert to label-based graph
    #     G = nx.Graph()
    #     for node in G_indices.nodes():
    #     # if G_indices.degree(node) > 0:  # Only add connected nodes
    #         G.add_node(label_list[node])
    #     for u, v in G_indices.edges():
    #         G.add_edge(label_list[u], label_list[v])

    #   #  components = list(nx.connected_components(G))


    #     for edge in G.edges():

    #         #print(clusters[edge[0]])
                  
    #         points_first_lr_X = np.array([U_flat[i] for i in clusters[edge[0]]])
    #         points_first_lr_Y = np.array([V_flat[i] for i in clusters[edge[0]]])

    #         points_second_lr_X = np.array([U_flat[i] for i in clusters[edge[1]]])
    #         points_second_lr_Y = np.array([V_flat[i] for i in clusters[edge[1]]])

    #       #  print(points_first_lr)
    #       #  print(np.mean(points_first_lr))



    #         fig_layer.add_trace(go.Scatter( x = [np.mean(points_first_lr_X),  np.mean(points_second_lr_X)] ,  
    #                                         y =  [np.mean(points_first_lr_Y),  np.mean(points_second_lr_Y)],
    #                                         mode = "lines+markers",
    #                                       #  line = dict(width = 1, color="#00CED1",),
    #                                        # mode = "lines+markers",
    #                                         line = dict(width = 1, color=label_to_color[edge[0]],),
    #                                         marker = dict(size = 4, color = [label_to_color[edge[0]], label_to_color[edge[1]]]),
    #                                         showlegend=False))
            


        fig_layer.update_layout(
                
                 xaxis=dict(
                #     showgrid=False,
                #     showticklabels=False,
                #     showline=False,
                #     zeroline=False,
                #     ticks="",
                     range=[axis_ranges["xmin"], 1]#axis_ranges["xmax"]]
                 ),
                yaxis=dict(
                #     showgrid=False,
                #     showticklabels=False,
                #     showline=False,
                #     zeroline=False,
                #     ticks="",
                     range=[-1,axis_ranges["ymax"]]
                 )
            )

        fig_layer.update_layout(
            template="plotly_dark",
            width=1500,
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

    np.random.seed(42)
    plot_training_spacebent_all_smooth()
