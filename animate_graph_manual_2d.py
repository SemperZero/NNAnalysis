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
import plotly.io as pio
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import h5py

pio.renderers.default = "browser"



hamming_threshold = 1  
first_frame = 0

nr_neurons = 101
nr_digits = 10
activation = "sigmoid"
nr_epochs_train = 50

try_nr = f'digits_{nr_digits}_nodes_{nr_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0

# import tensorflow as tf
# (x_train, y_train_mapped), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
# x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0
# x_train = x_train_flattened[::2]
# y_labels = y_train[::2]




def get_interpolated_matrix(current_layer, next_layer, nr_interp_frames):
    interp_matrix = np.zeros((nr_interp_frames, current_layer.shape[0], current_layer.shape[1]))
    for i in range(current_layer.shape[0]):
        for j in range(current_layer.shape[1]):
            point_trace = np.linspace(current_layer[i][j], next_layer[i][j], nr_interp_frames)
            interp_matrix[:, i, j] = point_trace

    return interp_matrix

# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))

def sigmoid_force_quad(z):
    return np.where(z <= 0, -1, 1)

def get_quadrant_labels(X):
    """Assign a bitstring label to each sample based on sign(X-0.5)."""
    # X: (n_samples, n_features)
    return np.packbits((X > 0).astype(np.uint8), axis=1, bitorder='little').flatten()

def bitstring_labels(X_hd, offset_value = 0):
    """Return list of bitstrings for each sample."""
    return [''.join(str(int(b)) for b in (row > offset_value)) for row in X_hd]

def quadrant_clusters(X_hd, offset_value = 0):
    """Return a dict mapping quadrant label to list of indices."""
    labels = bitstring_labels(X_hd, offset_value)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters, labels



def compute_hamming_graph_lsh(labels, max_hamming=15, sample_rate=0.1):
    """
    Compute approximate Hamming graph using LSH for efficiency.
    For very large graphs, we sample edges probabilistically.
    """
    n = len(labels)
    
    # Convert labels to binary matrix
    max_len = max(len(l) for l in labels)
    binary_matrix = np.zeros((n, max_len), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, bit in enumerate(label):
            binary_matrix[i, j] = int(bit)
    
    # For small graphs, compute exact
    if n < 20000:
        # Exact computation
        dists = squareform(pdist(binary_matrix, metric='hamming')) * max_len
        edges = np.argwhere((dists <= max_hamming) & (dists > 0))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        return G, labels
    
    # For large graphs, use approximate nearest neighbors
    print(f"Using approximate method for {n} nodes")
    
    # Use LSH via random projections
    n_projections = min(32, max_len // 2)  # Number of hash functions
    
    # Random projection matrix
    np.random.seed(42)  # For reproducibility
    proj_matrix = np.random.randn(n_projections, max_len)
    
    # Project binary vectors
    projections = binary_matrix @ proj_matrix.T
    
    # Find approximate neighbors using ball tree
    from sklearn.neighbors import BallTree
    tree = BallTree(projections, metric='euclidean')
    
    # Query for neighbors within approximate distance
    # We need to calibrate the distance threshold
    approx_radius = np.sqrt(max_hamming * n_projections / max_len) * 1.5
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Sample nodes for edge computation
    sample_size = min(n, max(100, int(n * sample_rate)))
    sampled_indices = np.random.choice(n, sample_size, replace=False)
    
    for idx in sampled_indices:
        # Find approximate neighbors
        indices = tree.query_radius([projections[idx]], r=approx_radius)
        neighbors = indices[0]  # Get the array of neighbor indices
        
        # Verify actual Hamming distance for candidates
        for neighbor_idx in neighbors:
            neighbor_idx = int(neighbor_idx)  # Ensure it's an integer
            if neighbor_idx != idx:
                actual_dist = np.sum(binary_matrix[idx] != binary_matrix[neighbor_idx])
                if actual_dist <= max_hamming:
                    G.add_edge(idx, neighbor_idx)
    
    return G, labels

def plot_training_spacebent_all_smooth():
    
    global try_nr

    losses = np.load(f'trainings2\\training_digits_{try_nr}\\loss.npy')
    
    nr_layers = 4
    
    layer_nr = selected_layer
    predict_layer = 1

    # Define custom digit colors
    digit_colors = [
        "#9400D3", "#78FF9A", "#0000FF", "#FFEE02", "#006B2D",
        "#FF7F00", "#FF0000", "#FF1493", "#00CED1", "#F0614E"
    ]

    X_bent_output_layers_0 = {}
    
    h5_filepath = f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.h5"
    if os.path.exists(h5_filepath):
        with h5py.File(h5_filepath, 'r') as f:
            X_latent = np.array(f['data'])
    
    h5_filepath_1 = f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_1.h5"
    if os.path.exists(h5_filepath_1):
        with h5py.File(h5_filepath_1, 'r') as f:
            X_pred = np.array(f['data'])
    


   # X_latent = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy')
   # X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
    accuracies = np.load(f'trainings2\\training_digits_{try_nr}\\accuracy.npy')

    random_points = sigmoid_force_quad(np.random.uniform(-15, 15 ,  (100000, X_latent[0].shape[1])))

    print("total_saves", len(X_latent))
    print("total acc", len(accuracies))

  #  if len(X_latent) != len(accuracies):
   #     print("lengths not equal")
   #     exit()

    pca = PCA(n_components=2)
    pca = pca.fit(X_latent[-1]) # X_latent[-1]-0.5
    trained_pca = pca

        
            
   # X_latent_pca = []
    


    #for i in range(len(X_latent)):
    #    trained_pca.append(trained_pca.transform(X_latent[i]))


    

    degrees_per_sec = 10
    degrees_per_iter = degrees_per_sec / 60
   # print("total descents", len(X_bent_output_layers[0]), len(space_bent_output_all[0]),len(layer_weights), len(predictions), X_bent_output_layers[0][0].shape, space_bent_output_all[0][0].shape)
   # exit()
   
    frame_nr = 0
    skip_smooth_interval = 1
    
    animation_folder_path = f"animations2\\latent_graph\\{try_nr}_{skip_smooth_interval}"# nr_interp_frames, skip_step
    if not os.path.isdir(animation_folder_path):
        os.makedirs(animation_folder_path)

    
    if not os.path.isdir(f"{animation_folder_path}\\layer_nr{layer_nr}"):
        os.makedirs(f"{animation_folder_path}\\layer_nr{layer_nr}")
        
        
    for ii in range(first_frame, len(X_latent), skip_smooth_interval):
        print(ii)


        angle_ = frame_nr * degrees_per_iter * np.pi/180
    
        
        X_latent_ii = X_latent[ii][::4]
        X_pred_ii = X_pred[ii][::4]

        if activation == "sigmoid":
            X_latent_ii = sigmoid_force_quad(X_latent_ii-0.5)
        elif activation == "relu":
            X_latent_ii = sigmoid_force_quad(X_latent_ii)




        X_latent_pca = trained_pca.transform(X_latent_ii)
        X_pred_label = np.argmax(X_pred_ii, axis=1)
        


        clusters, labels = quadrant_clusters(X_latent_ii)


        label_list = list(labels)
        G_indices, _ = compute_hamming_graph_lsh(label_list, max_hamming=hamming_threshold, sample_rate=0.2)
                
        # Convert to label-based graph
        G = nx.Graph()
        for node in G_indices.nodes():
           # if G_indices.degree(node) > 0:  # Only add connected nodes
            G.add_node(label_list[node])
        for u, v in G_indices.edges():
            G.add_edge(label_list[u], label_list[v])



        components = list(nx.connected_components(G))
        # Filter to only meaningful components (>2 nodes)
        meaningful_components = [comp for comp in components if len(comp) > 2]


 








        fig_layer = go.Figure()



        # Separate indices for clusters >=3 and <=3
        large_cluster_indices = []
        small_cluster_indices = []
        for comp in components:
            if len(comp) >= 3:
                # Get actual sample indices from clusters, not cluster labels
                for cluster_label in comp:
                    if cluster_label in clusters:
                        large_cluster_indices.extend(clusters[cluster_label])
            else:
                # Get actual sample indices from clusters, not cluster labels
                for cluster_label in comp:
                    if cluster_label in clusters:
                        small_cluster_indices.extend(clusters[cluster_label])

        # Plot large clusters (size 2, opacity 0.9)
        if large_cluster_indices:
            fig_layer.add_trace(go.Scatter(
                name="large clusters",
                x=X_latent_pca[large_cluster_indices, 0],
                y=X_latent_pca[large_cluster_indices, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=[digit_colors[label] for label in np.array(X_pred_label)[large_cluster_indices]],
                    line=dict(width=0.5, color='white')
                ),
                showlegend=False
            ))

        # Plot small clusters (size 1, opacity 0.3)
        if small_cluster_indices:
            fig_layer.add_trace(go.Scatter(
                name="small clusters",
                x=X_latent_pca[small_cluster_indices, 0],
                y=X_latent_pca[small_cluster_indices, 1],
                mode="markers",
                marker=dict(
                    size=4,
                    opacity=0.3,
                    color=[digit_colors[label] for label in np.array(X_pred_label)[small_cluster_indices]],
                  #  line=dict(width=0.2, color='white')
                ),
                showlegend=False
            ))

        # Generate and plot paths for each connected component
        for comp_idx, comp in enumerate(components):
            if len(comp) >= 3:  # Only plot paths for meaningful components
                # Get subgraph for this component
                subgraph = G.subgraph(comp)
                
                # Generate path that covers all edges (Eulerian-like path)
                edges = list(subgraph.edges())
                if edges:
                    # Simple approach: create a path that visits all edges
                    # We'll traverse each edge at least once
                    path_nodes = []
                    visited_edges = set()
                    
                    # Start from any node
                    current_node = list(comp)[0]
                    path_nodes.append(current_node)
                    
                    while len(visited_edges) < len(edges):
                        # Find an unvisited edge from current node
                        found_edge = False
                        for edge in edges:
                            edge_tuple = tuple(sorted([edge[0], edge[1]]))
                            if edge_tuple not in visited_edges:
                                if edge[0] == current_node:
                                    path_nodes.append(edge[1])
                                    current_node = edge[1]
                                    visited_edges.add(edge_tuple)
                                    found_edge = True
                                    break
                                elif edge[1] == current_node:
                                    path_nodes.append(edge[0])
                                    current_node = edge[0]
                                    visited_edges.add(edge_tuple)
                                    found_edge = True
                                    break
                        
                        if not found_edge:
                            # Jump to a node that has unvisited edges
                            for edge in edges:
                                edge_tuple = tuple(sorted([edge[0], edge[1]]))
                                if edge_tuple not in visited_edges:
                                    path_nodes.append(edge[0])
                                    current_node = edge[0]
                                    break
                    
                    # Convert path nodes (cluster labels) to actual sample indices and positions
                    path_indices = []
                    for node_label in path_nodes:
                        if node_label in clusters and clusters[node_label]:
                            # Take the first sample from each cluster as representative
                            path_indices.append(clusters[node_label][0])
                    
                    if len(path_indices) > 1:
                        # Get positions and colors for the path
                        path_positions = X_latent_pca[path_indices]
                        path_colors = [digit_colors[label] for label in np.array(X_pred_label)[path_indices]]
                        
                        # Plot the path as a line trace
                        fig_layer.add_trace(go.Scatter(
                            name=f"component_{comp_idx}_path",
                            x=path_positions[:, 0],
                            y=path_positions[:, 1],
                            mode="lines",
                            line=dict(
                                width=2,
                                color=path_colors[0]  # Use first color for the line
                            ),
                            marker=dict(
                                size=2,
                                color=path_colors
                            ),
                            showlegend=False,
                            opacity=0.5
                        ))

       
        #add lines here






        #fig_layer.add_trace(go.Scatter(name = "sigmoid space", x=X_latent_pca[:,0], y=X_latent_pca[:,1], mode = "markers", marker = dict(opacity=0.7, size = 1, color=X_pred_label,colorscale='rainbow')))

        # Remove 3D camera settings for 2D plot
        fig_layer.update_layout( template = "plotly_dark", width = 2200, height = 1400)
        fig_layer.update_layout( title = f"frame {ii}, hamm thr: {hamming_threshold}")
        fig_layer.update_coloraxes(showscale=False)
        
        # Remove grid from axes
        fig_layer.update_xaxes(showgrid=False)
        fig_layer.update_yaxes(showgrid=False)
        
        # Remove axis numbers, ticks, and zero lines
        fig_layer.update_xaxes(showticklabels=False, showline=False, zeroline=False, ticks="")
        fig_layer.update_yaxes(showticklabels=False, showline=False, zeroline=False, ticks="")


        fig_layer.write_image(f"{animation_folder_path}\\layer_nr{layer_nr}\\image_{frame_nr:05}.png")

        
       # print("framenr", frame_nr)
       # fig_layer.show()
       # exit()
        frame_nr+=1





plot_training_spacebent_all_smooth()


plot_training_spacebent_all_smooth()


