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
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm


pio.renderers.default = "browser"



hamming_threshold = 3
first_frame = 40

nr_neurons = 101
nr_digits = 10
activation = "relu"
nr_epochs_train = 300

run_nr = 21

nr_tsne_frames = 50

selected_digit = 8


try_nr = f'digits_{nr_digits}_nodes_{nr_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0

import tensorflow as tf
(x_train, y_train_mapped), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0
x_train = x_train_flattened
y_labels = y_train

X_train = x_train[::1]




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
    Compute exact Hamming graph and distances.
    Returns:
        G: networkx graph
        labels: original labels
        edge_distances: dictionary of edge distances
        dists: full Hamming distance matrix
    """
    n = len(labels)
    
    # Convert labels to binary matrix
    max_len = max(len(l) for l in labels)
    binary_matrix = np.zeros((n, max_len), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, bit in enumerate(label):
            binary_matrix[i, j] = int(bit)
    
    if n < 20000:
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

        #print(G.edges())
       # exit()
        return G, labels, edge_distances, dists
    
  

def plot_training_spacebent_all_smooth():
    
    global try_nr, x_train

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
    
    # h5_filepath = f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.h5"
    # if os.path.exists(h5_filepath):
    #     with h5py.File(h5_filepath, 'r') as f:
    #         X_latent = np.array(f['data'])
    
    # h5_filepath_1 = f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_1.h5"
    # if os.path.exists(h5_filepath_1):
    #     with h5py.File(h5_filepath_1, 'r') as f:
    #         X_pred = np.array(f['data'])
    


    X_latent = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy')
    X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
    accuracies = np.load(f'trainings2\\training_digits_{try_nr}\\accuracy.npy')


    random_points = sigmoid_force_quad(np.random.uniform(-15, 15 ,  (100000, X_latent[0].shape[1])))

    print("total_saves", len(X_latent))
    print("total acc", len(accuracies))

  #  if len(X_latent) != len(accuracies):
   #     print("lengths not equal")
   #     exit()

    # pca = PCA(n_components=2)
    # pca = pca.fit(X_latent[-1]) # X_latent[-1]-0.5
    # trained_pca = pca

    # n_neighbors= 3  #hamming_threshold * 1
    # min_dist=0.2
    # spread=0.3
    # metric='hamming'
    # random_state=42


    # reducer = umap.UMAP(
    #     n_components = 2,
    #    # n_jobs = 4,
    #     n_neighbors=n_neighbors,     # capture local groups (adjust to 10–30)
    #     min_dist=min_dist,       # keep close points very tight
    #     spread=spread,         # allow distant groups to really spread apart
    #     metric=metric,
    #     random_state=random_state
    #     )
    
    # trained_pca = reducer.fit(X_latent[first_frame])   # fit once
        
            
   # X_latent_pca = []
    


    #for i in range(len(X_latent)):
    #    trained_pca.append(trained_pca.transform(X_latent[i]))


    

    degrees_per_sec = 10
    degrees_per_iter = degrees_per_sec / 60
   # print("total descents", len(X_bent_output_layers[0]), len(space_bent_output_all[0]),len(layer_weights), len(predictions), X_bent_output_layers[0][0].shape, space_bent_output_all[0][0].shape)
   # exit()
   
    frame_nr = 0
    skip_smooth_interval = 1
    
    animation_folder_path = f"animations2\\latent_graph_tsne\\{try_nr}"# nr_interp_frames, skip_step
    if not os.path.isdir(animation_folder_path):
        os.makedirs(animation_folder_path)

    
    layer_folder_path = f"{animation_folder_path}\\layer_nr{layer_nr}_hamm_{hamming_threshold}_nointerp_tsne_{run_nr}"
    if not os.path.isdir(layer_folder_path):
        os.makedirs(layer_folder_path)
        
        
    connected_set = set()
    tsne_train_set = []
    map_label_to_tsne_index = {}
    tsne_index = 0

    frames_tsne_idxs = []
    frames_large_labels = []
    for ii in range(first_frame, first_frame + nr_tsne_frames, skip_smooth_interval):
        print(ii, "ii")


        X_latent_ii = X_latent[ii][::1]

        if activation == "sigmoid":
            X_latent_ii = sigmoid_force_quad(X_latent_ii-0.5)
        elif activation == "relu":
            X_latent_ii = sigmoid_force_quad(X_latent_ii)

        
        X_pred_label = y_labels[::1]

        indices = np.where(X_pred_label == selected_digit)[0]
        
      
        X_pred_label = X_pred_label[indices]
        X_latent_ii = X_latent_ii[indices]
        x_train = X_train[indices]

        print(len(x_train), "len x_train")


        clusters, labels = quadrant_clusters(X_latent_ii)

        label_list = list(labels)
        G_indices, _, edge_distances, dist_matrix = compute_hamming_graph_lsh(
            label_list, max_hamming=hamming_threshold, sample_rate=0.2
        )
                
        # Convert to label-based graph
        G = nx.Graph()
        for node in G_indices.nodes():
           # if G_indices.degree(node) > 0:  # Only add connected nodes
            G.add_node(label_list[node])
        for u, v in G_indices.edges():
            G.add_edge(label_list[u], label_list[v])




        components = list(nx.connected_components(G))

        large_cluster_labels = []
        
        for comp in components:
            if len(comp) >= 2:
                # Get actual sample indices from clusters, not cluster labels
                for cluster_label in comp:
                    if cluster_label in clusters:
                        large_cluster_labels.append(cluster_label)






     #   map_quadr_label_to_coords = {}
     #   for cluster_label in clusters:
     #       map_quadr_label_to_coords[cluster_label] = X_latent_ii[clusters[cluster_label][0]]


        old_len = len(connected_set)
        #connected_set.update(labels)

        tsne_idxs = []
        for label in large_cluster_labels:
            if label not in connected_set:
                tsne_train_set.append(X_latent_ii[clusters[label][0]])

                map_label_to_tsne_index[label] = tsne_index
                tsne_idxs.append(tsne_index)
                tsne_index+=1

        connected_set.update(large_cluster_labels)

        frames_large_labels.append(large_cluster_labels)





        print(len(connected_set) - old_len, "new connected")
        print(len(connected_set), "total connected")
        print()
        continue


    tsne_train_set = np.array(tsne_train_set)
    print(tsne_train_set.shape)
    perplexity = 30
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    X_latent_tsne = tsne.fit_transform(tsne_train_set)

    def get_axis_ranges():

        return {
            'xmin': min(X_latent_tsne[:,0]),
            'xmax': max(X_latent_tsne[:,0]),
            'ymin': min(X_latent_tsne[:,1]),
            'ymax': max(X_latent_tsne[:,1])
        }

    print("start axis_range")
    axis_ranges = get_axis_ranges()



    # fig_layer = go.Figure()

    # for ii in range(first_frame, first_frame + nr_tsne_frames, skip_smooth_interval):
        
    #     large_labels = [map_label_to_tsne_index[l] for l in frames_large_labels[ii-first_frame]]

    #     fig_layer.add_trace(go.Scatter(
    #                 x= X_latent_tsne[large_labels][:, 0],
    #                 y= X_latent_tsne[large_labels][:, 1],
    #                 # z=[ X_latent_pca[clusters[edge[0]][0]][2], X_latent_pca[clusters[edge[1]][0]][2] ],
    #                 mode="markers",
    #                 marker=dict(
    #                     size = 5,
    #                   #  color=digit_colors[selected_digit]
    #                 ),
    #                 #showlegend=False,
    #                 opacity=0.7
    #             ))
        
    #     continue
    #     X_latent_ii = X_latent[ii][::1]

    #     if activation == "sigmoid":
    #         X_latent_ii = sigmoid_force_quad(X_latent_ii-0.5)
    #     elif activation == "relu":
    #         X_latent_ii = sigmoid_force_quad(X_latent_ii)

        
    #     X_pred_label = y_labels[::1]

    #     indices = np.where(X_pred_label == selected_digit)[0]
        
      
    #     X_pred_label = X_pred_label[indices]
    #     X_latent_ii = X_latent_ii[indices]
    #     x_train = X_train[indices]

    #     print(len(x_train), "len x_train")


    #     clusters, labels = quadrant_clusters(X_latent_ii)

    #     label_list = list(labels)
    #     G_indices, _, edge_distances, dist_matrix = compute_hamming_graph_lsh(
    #         label_list, max_hamming=hamming_threshold, sample_rate=0.2
    #     )
                
    #     # Convert to label-based graph
    #     G = nx.Graph()
    #     for node in G_indices.nodes():
    #        # if G_indices.degree(node) > 0:  # Only add connected nodes
    #         G.add_node(label_list[node])
    #     for u, v in G_indices.edges():
    #         G.add_edge(label_list[u], label_list[v])




    #     components = list(nx.connected_components(G))



    #    # fig_layer = go.Figure()

    #     #print(G.edges())

    #     for edge in G.edges():
    #         edge_tuple = tuple(sorted([edge[0], edge[1]]))
    #         if tuple([edge[0], edge[1]]) in list(G.edges()) or tuple([edge[1], edge[0]]) in list(G.edges()):
    #             # Get the Hamming distance for this edge
    #             dist = edge_distances.get((edge[0], edge[1]))
                
    #             if dist == 8:
    #                 line_width = 2
    #                 opacity = 0.75
    #             elif dist == 2:
    #                 line_width = 1
    #                 opacity = 0.5
    #             else:  # dist == 3
    #                 line_width = 0.5
    #                 opacity = 0.3

                
    #            # print(X_latent_tsne[map_label_to_tsne_index[edge[0]]][0], X_latent_tsne[map_label_to_tsne_index[edge[1]]][0], "interp coords")

    #            # exit()
    #             fig_layer.add_trace(go.Scatter(
    #                 x=[ X_latent_tsne[map_label_to_tsne_index[edge[0]]][0], X_latent_tsne[map_label_to_tsne_index[edge[1]]][0]  ],
    #                 y=[ X_latent_tsne[map_label_to_tsne_index[edge[0]]][1], X_latent_tsne[map_label_to_tsne_index[edge[1]]][1] ],
    #                # z=[ X_latent_pca[clusters[edge[0]][0]][2], X_latent_pca[clusters[edge[1]][0]][2] ],
    #                 mode="lines",
    #                 line=dict(
    #                     width=line_width,
    #                     color=digit_colors[X_pred_label[clusters[edge[0]][0]]]
    #                 ),
    #                 showlegend=False,
    #                 opacity=opacity
    #             ))

    # fig_layer.update_layout( template = "plotly_dark", width = 2200, height = 1400)
    # fig_layer.update_layout( title = f"Full tsne frames {first_frame} -> {first_frame + nr_tsne_frames}, hamm thr: {hamming_threshold} \
    #                                     ")
    # fig_layer.update_layout(
    #         template="plotly_dark",
    #         width=2200,
    #         height=1400,
    #         title=f"frame {ii}, hamm thr: {hamming_threshold}",
    #         xaxis=dict(
    #             showgrid=False,
    #             showticklabels=False,
    #             showline=False,
    #             zeroline=False,
    #             ticks="",
    #             range=[axis_ranges['xmin'], axis_ranges['xmax']]
    #         ),
    #         yaxis=dict(
    #             showgrid=False,
    #             showticklabels=False,
    #             showline=False,
    #             zeroline=False,
    #             ticks="",
    #             range=[axis_ranges['ymin'], axis_ranges['ymax']]
    #         )
    #     )
    # fig_layer.update_coloraxes(showscale=False)
    # fig_layer.show()
    # exit()


    for ii in range(first_frame, first_frame + nr_tsne_frames, skip_smooth_interval):
        X_latent_ii = X_latent[ii][::1]

        if activation == "sigmoid":
            X_latent_ii = sigmoid_force_quad(X_latent_ii-0.5)
        elif activation == "relu":
            X_latent_ii = sigmoid_force_quad(X_latent_ii)

        
        X_pred_label = y_labels[::1]

        indices = np.where(X_pred_label == selected_digit)[0]
        
      
        X_pred_label = X_pred_label[indices]
        X_latent_ii = X_latent_ii[indices]
        x_train = X_train[indices]

        print(len(x_train), "len x_train")


        clusters, labels = quadrant_clusters(X_latent_ii)

        label_list = list(labels)
        G_indices, _, edge_distances, dist_matrix = compute_hamming_graph_lsh(
            label_list, max_hamming=hamming_threshold, sample_rate=0.2
        )
                
        # Convert to label-based graph
        G = nx.Graph()
        for node in G_indices.nodes():
           # if G_indices.degree(node) > 0:  # Only add connected nodes
            G.add_node(label_list[node])
        for u, v in G_indices.edges():
            G.add_edge(label_list[u], label_list[v])




        components = list(nx.connected_components(G))



        fig_layer = go.Figure()

        print(G.edges())

        for edge in G.edges():
            edge_tuple = tuple(sorted([edge[0], edge[1]]))
            if tuple([edge[0], edge[1]]) in list(G.edges()) or tuple([edge[1], edge[0]]) in list(G.edges()):
                # Get the Hamming distance for this edge
                dist = edge_distances.get((edge[0], edge[1]))
                
                if dist == 8:
                    line_width = 2
                    opacity = 0.75
                elif dist == 2:
                    line_width = 1
                    opacity = 0.5
                else:  # dist == 3
                    line_width = 0.5
                    opacity = 0.3

                
               # print(X_latent_tsne[map_label_to_tsne_index[edge[0]]][0], X_latent_tsne[map_label_to_tsne_index[edge[1]]][0], "interp coords")

               # exit()
                fig_layer.add_trace(go.Scatter(
                    x=[ X_latent_tsne[map_label_to_tsne_index[edge[0]]][0], X_latent_tsne[map_label_to_tsne_index[edge[1]]][0]  ],
                    y=[ X_latent_tsne[map_label_to_tsne_index[edge[0]]][1], X_latent_tsne[map_label_to_tsne_index[edge[1]]][1] ],
                   # z=[ X_latent_pca[clusters[edge[0]][0]][2], X_latent_pca[clusters[edge[1]][0]][2] ],
                    mode="lines",
                    line=dict(
                        width=line_width,
                        color=digit_colors[X_pred_label[clusters[edge[0]][0]]]
                    ),
                    showlegend=False,
                    opacity=opacity
                ))


     
       
        #add lines here



            


        #fig_layer.add_trace(go.Scatter(name = "sigmoid space", x=X_latent_pca[:,0], y=X_latent_pca[:,1], mode = "markers", marker = dict(opacity=0.7, size = 1, color=X_pred_label,colorscale='rainbow')))

        # Remove 3D camera settings for 2D plot
        fig_layer.update_layout( template = "plotly_dark", width = 2200, height = 1400)
        fig_layer.update_layout( title = f"frame {ii}, hamm thr: {hamming_threshold} \
                                           ")
        fig_layer.update_layout(
                template="plotly_dark",
                width=2200,
                height=1400,
                title=f"frame {ii}, hamm thr: {hamming_threshold}",
                xaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    showline=False,
                    zeroline=False,
                    ticks="",
                    range=[axis_ranges['xmin'], axis_ranges['xmax']]
                ),
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    showline=False,
                    zeroline=False,
                    ticks="",
                    range=[axis_ranges['ymin'], axis_ranges['ymax']]
                )
            )
        fig_layer.update_coloraxes(showscale=False)

        fig_layer.write_image(f"{layer_folder_path}\\image_{frame_nr:05}.png")

        
       # print("framenr", frame_nr)
       # fig_layer.show()
       # exit()
        frame_nr+=1





plot_training_spacebent_all_smooth()



