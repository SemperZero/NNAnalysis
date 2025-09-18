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
import umap

pio.renderers.default = "browser"



hamming_threshold = 5
first_frame = 20

nr_neurons = 101
nr_digits = 10
activation = "relu"
nr_epochs_train = 300

try_nr = f'digits_{nr_digits}_nodes_{nr_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layer = 0

run_nr = 16

nr_tsne_frames = 30

selected_digit = 8

nr_move_frames = 5

import tensorflow as tf
(x_train, y_train_mapped), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0
X_train = x_train_flattened[::1]
y_labels = y_train[::1]




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
        return G, labels, edge_distances, dists
    

def get_component_path_traces(components_prev_large, G_prev_large, clusters, move_labels_prev, x_coords_interp, y_coords_interp, colors_interp_all, opacity):
    all_visited = set()
    traces = []
    #TODO IMPORTANT : are the large nodes (conn comp>3) that are in both frames the ones that don't change quad clusters???
    for comp_idx, comp in enumerate(components_prev_large):
        if len(comp) >= 2:  # Only plot paths for meaningful components
            # Get subgraph for this component
            subgraph = G_prev_large.subgraph(comp)
            
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
                                all_visited.add(edge_tuple)
                                found_edge = True
                                break
                            elif edge[1] == current_node:
                                path_nodes.append(edge[0])
                                current_node = edge[0]
                                visited_edges.add(edge_tuple)
                                all_visited.add(edge_tuple)
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
                    # Get interpolated coordinates for each path node
                    path_x = []
                    path_y = []
                    path_colors = []
                    for node_label in path_nodes:
                        idx = np.where(move_labels_prev == node_label)[0][0]  # Find index in interpolated arrays
                        path_x.append(x_coords_interp[idx])
                        path_y.append(y_coords_interp[idx])
                        path_colors.append(colors_interp_all[idx])

                    traces.append(go.Scatter(
                        x=path_x,
                        y=path_y,
                        mode="lines",
                        line=dict(
                            width=2,
                            color=path_colors[0]  # Use first color for the line
                        ),
                        showlegend=False,
                        opacity=opacity
                    ))
    return traces, all_visited

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
    
    X_latent = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy')
    X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
    accuracies = np.load(f'trainings2\\training_digits_{try_nr}\\accuracy.npy')

    random_points = sigmoid_force_quad(np.random.uniform(-15, 15 ,  (100000, X_latent[0].shape[1])))

    print("total_saves", len(X_latent))
    print("total acc", len(accuracies))




    connected_set = set()
    tsne_train_set = []
    map_label_to_tsne_index = {}
    #edge_distances_global = []
    tsne_index = 0
    
    
    for ii in range(first_frame, first_frame + nr_tsne_frames, 1):
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

        clusters, labels = quadrant_clusters(X_latent_ii)

        label_list = list(labels)
        G_indices, _, edge_distances, dist_matrix = compute_hamming_graph_lsh(
            label_list, max_hamming=hamming_threshold, sample_rate=0.2
        )
                
       # edge_distances_global.append(edge_distances)

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
                for cluster_label in comp:
                    if cluster_label in clusters:
                        large_cluster_labels.append(cluster_label)

      #  print(large_cluster_labels[:10])
        old_len = len(connected_set)

        for label in large_cluster_labels:
            if label not in connected_set:
                tsne_train_set.append(X_latent_ii[clusters[label][0]])

                map_label_to_tsne_index[label] = tsne_index
                tsne_index+=1

        connected_set.update(large_cluster_labels)

        print(len(connected_set) - old_len, "new connected")
        print(len(connected_set), "total connected")
        print()
        continue


    tsne_train_set = np.array(tsne_train_set)
    print(tsne_train_set.shape, "tsne_train_set.shape")
    perplexity = 30
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='pca', random_state=42)
    X_latent_tsne = tsne.fit_transform(tsne_train_set)



    degrees_per_sec = 10
    degrees_per_iter = degrees_per_sec / 60
   # print("total descents", len(X_bent_output_layers[0]), len(space_bent_output_all[0]),len(layer_weights), len(predictions), X_bent_output_layers[0][0].shape, space_bent_output_all[0][0].shape)
   # exit()
   
    frame_nr = 0
    skip_smooth_interval = 1
    
    animation_folder_path = f"animations2\\latent_graph_tsne\\{try_nr}"# nr_interp_frames, skip_step
    if not os.path.isdir(animation_folder_path):
        os.makedirs(animation_folder_path)

    
    layer_folder_path = f"{animation_folder_path}\\layer_nr{layer_nr}_hamm_{hamming_threshold}_dig_{selected_digit}_{run_nr}"
    if not os.path.isdir(layer_folder_path):
        os.makedirs(layer_folder_path)
        
        

    # Get global min/max coordinates for consistent axis ranges
    def get_axis_ranges():

        return {
            'xmin': min(X_latent_tsne[:,0]),
            'xmax': max(X_latent_tsne[:,0]),
            'ymin': min(X_latent_tsne[:,1]),
            'ymax': max(X_latent_tsne[:,1])
        }

    print("start axis_range")
    axis_ranges = get_axis_ranges()
    print("end axis_range")

    def get_graph_data(ii):
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
        


        clusters, labels = quadrant_clusters(X_latent_ii)


        label_list = list(labels)
        G_indices, _, edge_distances, _ = compute_hamming_graph_lsh(label_list, max_hamming=hamming_threshold, sample_rate=0.2)

        # Convert to label-based graph
        G = nx.Graph()
        for node in G_indices.nodes():
           # if G_indices.degree(node) > 0:  # Only add connected nodes
            G.add_node(label_list[node])
        for u, v in G_indices.edges():
            G.add_edge(label_list[u], label_list[v])

        components = list(nx.connected_components(G))
        # Filter to only meaningful components (>2 nodes)
        #meaningful_components = [comp for comp in components if len(comp) > 2]

        # Separate indices for clusters >=3 and <=3
        large_cluster_labels = []
        small_cluster_labels = []
        for comp in components:
            if len(comp) >= 2:
                # Get actual sample indices from clusters, not cluster labels
                for cluster_label in comp:
                    if cluster_label in clusters:
                        large_cluster_labels.append(cluster_label)
            else:
                # Get actual sample indices from clusters, not cluster labels
                for cluster_label in comp:
                    if cluster_label in clusters:
                        small_cluster_labels.append(cluster_label)

      #  print(large_cluster_labels[:10])
        
        map_quadr_label_to_coords = {}
        map_quadr_label_to_colors = {}
        for cluster_label in clusters:

            if cluster_label in large_cluster_labels:
        
                labels_in_cluster = np.array(X_pred_label)[clusters[cluster_label]]
                most_common_label = np.bincount(labels_in_cluster).argmax()
                map_quadr_label_to_coords[cluster_label] = X_latent_tsne[map_label_to_tsne_index[cluster_label]]
                map_quadr_label_to_colors[cluster_label] = digit_colors[most_common_label]
            else:
                map_quadr_label_to_coords[cluster_label] = np.array([np.nan, np.nan])
                map_quadr_label_to_colors[cluster_label] = "#FFFFFF"



        return clusters, label_list, G_indices, None, X_pred_label, np.array(large_cluster_labels), np.array(small_cluster_labels), G, components, map_quadr_label_to_coords, map_quadr_label_to_colors, edge_distances


    def get_move_mapping(clusters_prev, clusters):

        item_to_label_map = {}
        for label in clusters:
            for item_index in clusters[label]:
                item_to_label_map[item_index] = label

        move_mapping = {label_prev : {} for label_prev in clusters_prev} # all will have moves from them
        move_mapping_items = {label_prev : {} for label_prev in clusters_prev}
        for label_prev in clusters_prev:

            #within a quad cluster
            for item_index_prev in clusters_prev[label_prev]:
                label_next = item_to_label_map[item_index_prev]
                if label_next not in move_mapping[label_prev]:
                    move_mapping[label_prev][label_next] = 0
                    move_mapping_items[label_prev][label_next] = []
                move_mapping[label_prev][label_next] += 1
                move_mapping_items[label_prev][label_next].append(item_index_prev)

        return move_mapping, move_mapping_items


    
    clusters_prev, label_list_prev, G_indices_prev, X_latent_pca_prev, X_pred_label_prev, large_cluster_labels_prev, small_cluster_labels_prev, G_prev, components_prev, map_quadr_label_to_coords_prev, map_quadr_label_to_colors_prev, edge_distances_prev = get_graph_data(first_frame)


    for ii in range(first_frame+1, len(X_latent), 1):
        print(ii)

        clusters, label_list, G_indices, X_latent_pca, X_pred_label, large_cluster_labels, small_cluster_labels, G, components, map_quadr_label_to_coords, map_quadr_label_to_colors, edge_distances = get_graph_data(ii)
        
                

        move_mapping_counts, move_mapping_items = get_move_mapping(clusters_prev, clusters)


        x_coords_prev = []
        y_coords_prev = []
        x_coords = []
        y_coords = []
        move_labels = []
        move_labels_prev = []
        colors_arr_prev = []
        colors_arr = []

        for prev_label in move_mapping_counts:
            for next_label in move_mapping_counts[prev_label]:
                x_coords_prev.append(map_quadr_label_to_coords_prev[prev_label][0])
                y_coords_prev.append(map_quadr_label_to_coords_prev[prev_label][1])
                
                x_coords.append(map_quadr_label_to_coords[next_label][0])
                y_coords.append(map_quadr_label_to_coords[next_label][1])

                move_labels.append(next_label)
                move_labels_prev.append(prev_label)
                
                colors_arr_prev.append(map_quadr_label_to_colors_prev[prev_label])
                colors_arr.append(map_quadr_label_to_colors[next_label])


        
        x_coords_prev = np.array(x_coords_prev)
        y_coords_prev = np.array(y_coords_prev)
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        move_labels = np.array(move_labels)
        move_labels_prev = np.array(move_labels_prev)
        colors_arr_prev = np.array(colors_arr_prev)
        colors_arr = np.array(colors_arr)
        




        large_cluster_mask = np.isin(move_labels_prev, large_cluster_labels_prev) & np.isin(move_labels, large_cluster_labels)
        x_coords_prev_large = x_coords_prev[large_cluster_mask]
        y_coords_prev_large = y_coords_prev[large_cluster_mask]
        x_coords_large = x_coords[large_cluster_mask]
        y_coords_large = y_coords[large_cluster_mask]
        colors_prev_large = colors_arr_prev[large_cluster_mask]




        nr_fade_out_frames = 5
        
        nr_fade_in_frames = 5


        # prev are large, new are small
        
        fade_out_masks = np.isin(move_labels_prev, large_cluster_labels_prev) & np.isin(move_labels, small_cluster_labels)
        x_coords_fade_out = x_coords_prev[fade_out_masks]
        y_coords_fade_out = y_coords_prev[fade_out_masks]
        colors_fade_out = colors_arr_prev[fade_out_masks]
        

        fade_in_masks = np.isin(move_labels_prev, small_cluster_labels_prev) & np.isin(move_labels, large_cluster_labels)
        x_coords_fade_in = x_coords[fade_in_masks]
        y_coords_fade_in = y_coords[fade_in_masks]
        colors_fade_in = colors_arr_prev[fade_in_masks]


        small_cluster_mask = (~large_cluster_mask) & (~fade_out_masks) & (~fade_in_masks)
        x_coords_prev_small = x_coords_prev[small_cluster_mask]
        y_coords_prev_small = y_coords_prev[small_cluster_mask]
        x_coords_small = x_coords[small_cluster_mask]
        y_coords_small = y_coords[small_cluster_mask]
        colors_prev_small = colors_arr_prev[small_cluster_mask]


      #  fade_out_masks = []

        # colors_fade_out = []
        # for prev_label in move_mapping_counts:
        #     for next_label in move_mapping_counts[prev_label]:
        #         if prev_label in large_cluster_labels_prev and next_label in small_cluster_labels:
        #             fade_out_masks.append(True)
        #         else:
        #             fade_out_masks.append(False)


        # fade_out_masks = np.array(fade_out_masks)
        # x_coords_fade_out = x_coords[fade_out_masks]
        # y_coords_fade_out = y_coords[fade_out_masks]
        # colors_fade_out = colors_arr[fade_out_masks]


        # fade_in_masks = []

        # colors_fade_in = []
        # for prev_label in move_mapping_counts:
        #     for next_label in move_mapping_counts[prev_label]:
        #         if prev_label in small_cluster_labels_prev and next_label in large_cluster_labels:
        #             fade_in_masks.append(True)
        #         else:
        #             fade_in_masks.append(False)


        # fade_in_masks = np.array(fade_in_masks)
        # x_coords_fade_in = x_coords[fade_in_masks]
        # y_coords_fade_in = y_coords[fade_in_masks]
        # colors_fade_in = colors_arr[fade_in_masks]
    

        large_cluster_labels_prev_filtered = move_labels_prev[large_cluster_mask]

        set_clusters_labels_prev = set(large_cluster_labels_prev_filtered)

        
                                            
        G_prev_large = G_prev.subgraph(np.unique(large_cluster_labels_prev_filtered))
        components_prev_large = list(nx.connected_components(G_prev_large))

        set_clusters_labels_next = set(move_labels)



        for nr_interp_frame in range(nr_move_frames):


           
            t = nr_interp_frame/(nr_move_frames-1)
            size = 4 + (8-4) * t

            
            # Interpolate colors by converting hex to rgb, interpolating, then back to hex
            def interpolate_colors(color1_arr, color2_arr, t):
                # Convert hex to RGB arrays
                rgb1_arr = np.array([int(c[1:3], 16) for c in color1_arr]), np.array([int(c[3:5], 16) for c in color1_arr]), np.array([int(c[5:7], 16) for c in color1_arr])
                rgb2_arr = np.array([int(c[1:3], 16) for c in color2_arr]), np.array([int(c[3:5], 16) for c in color2_arr]), np.array([int(c[5:7], 16) for c in color2_arr])
                
                # Interpolate RGB values
                rgb_interp = tuple(np.round((1-t) * rgb1 + t * rgb2).astype(int) for rgb1, rgb2 in zip(rgb1_arr, rgb2_arr))
                
                # Convert back to hex
                return np.array(['#' + ''.join(f'{c:02x}' for c in rgb) for rgb in zip(*rgb_interp)])

            # Interpolate colors for all three sets
            colors_fade_in_interp = interpolate_colors(colors_arr_prev[fade_in_masks], colors_arr[fade_in_masks], t)
            colors_large_interp = interpolate_colors(colors_prev_large, colors_arr[large_cluster_mask], t)
            colors_small_interp = interpolate_colors(colors_arr_prev[small_cluster_mask], colors_arr[small_cluster_mask], t)
            colors_interp_all = interpolate_colors(colors_arr_prev, colors_arr, t)



            fig_layer = go.Figure()



            x_coords_interp_large = x_coords_prev_large + (x_coords_large - x_coords_prev_large) * nr_interp_frame/(nr_move_frames-1)
            y_coords_interp_large = y_coords_prev_large + (y_coords_large - y_coords_prev_large) * nr_interp_frame/(nr_move_frames-1)
            

            
            x_coords_interp_small = x_coords_prev_small + (x_coords_small - x_coords_prev_small) * nr_interp_frame/(nr_move_frames-1)
            y_coords_interp_small = y_coords_prev_small + (y_coords_small - y_coords_prev_small) * nr_interp_frame/(nr_move_frames-1)
            





            x_coords_prev = np.where(np.isnan(x_coords_prev), x_coords, x_coords_prev)
            x_coords      = np.where(np.isnan(x_coords), x_coords_prev, x_coords)

            y_coords_prev = np.where(np.isnan(y_coords_prev), y_coords, y_coords_prev)
            y_coords      = np.where(np.isnan(y_coords), y_coords_prev, y_coords)

            # Interpolation
            x_coords_interp = x_coords_prev + (x_coords - x_coords_prev) * nr_interp_frame / (nr_move_frames - 1)
            y_coords_interp = y_coords_prev + (y_coords - y_coords_prev) * nr_interp_frame / (nr_move_frames - 1)

            # fig_layer.add_trace(go.Scatter(
            #     name="large clusters",
            #     x=x_coords_interp_large,
            #     y=y_coords_interp_large,
            #     mode="markers",
            #     marker=dict(
            #         size=8,
            #         opacity=0.7,
            #         color=colors_large_interp,
            #         line=dict(width=0.5, color='white')
            #     ),
            #     showlegend=False
            # ))



            # fig_layer.add_trace(go.Scatter(
            #     name="small clusters",
            #     x=x_coords_interp_small,
            #     y=y_coords_interp_small,
            #     mode="markers",
            #     marker=dict(
            #         size=4,
            #         opacity=0.3,
            #         color=colors_small_interp,
            #     #  line=dict(width=0.2, color='white')
            #     ),
            #     showlegend=False
            # ))

            




            # traces, all_visited = get_component_path_traces(components_prev_large, G_prev_large, clusters, move_labels_prev, 
            #                                               x_coords_interp, y_coords_interp, colors_interp_all, 0.5)
            # for trace in traces:
            #     fig_layer.add_trace(trace)


            def get_line_width_opac(dist):
                if dist == 1:
                    line_width = 2
                    opacity = 0.4
                elif dist == 2:
                    line_width = 1
                    opacity = 0.25
                else:  # dist == 3
                    line_width = 0.5
                    opacity = 0.15
                return line_width, opacity

            print("start add traces", len(G_prev.edges()))

            G_edges = set(list(G.edges()))
            G_prev_edges = set(list(G_prev.edges()))

            for edge in G_prev.edges():
                idx_0 = np.where(move_labels_prev == edge[0])[0][0]  # get first match
                idx_1 = np.where(move_labels_prev == edge[1])[0][0]  # get first match

                label_prev_0 = move_labels_prev[idx_0]
                label_next_0 = move_labels[idx_0]
                if label_prev_0 != edge[0]:
                    print(label_prev_0, edge[0], "label_prev_0")
                    exit()

                label_prev_1 = move_labels_prev[idx_1]
                label_next_1 = move_labels[idx_1]

                if label_prev_1 != edge[1]:
                    print(label_prev_1, edge[1], "label_prev_1")
                    exit()

                if tuple([label_next_0, label_next_1]) in G_edges or tuple([label_next_1, label_next_0]) in G_edges:


                    width_prev, opacity_prev = get_line_width_opac(edge_distances_prev.get((label_prev_0, label_prev_1)))
                    width_next, opacity_next = get_line_width_opac(edge_distances.get((label_next_0, label_next_1)))
                
                    line_width = width_prev + (width_next - width_prev) * nr_interp_frame / (nr_move_frames - 1)
                    opacity = opacity_prev + (opacity_next - opacity_prev) * nr_interp_frame / (nr_move_frames - 1)

                    fig_layer.add_trace(go.Scatter(
                        
                        x=[x_coords_interp[idx_0], x_coords_interp[idx_1]],
                        y=[y_coords_interp[idx_0], y_coords_interp[idx_1]],
                        mode="lines",
                        line=dict(
                            width=line_width,
                            color= map_quadr_label_to_colors_prev[move_labels_prev[idx_0]]
                        ),

                        showlegend=False,
                        opacity=opacity
                    ))






            #fig_layer.add_trace(go.Scatter(name = "sigmoid space", x=X_latent_pca[:,0], y=X_latent_pca[:,1], mode = "markers", marker = dict(opacity=0.7, size = 1, color=X_pred_label,colorscale='rainbow')))

            # Remove 3D camera settings for 2D plot
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





            ###########FADE OUT############
            size = 8 + (4-8) * nr_interp_frame/(nr_move_frames-1)


            # fig_layer.add_trace(go.Scatter( # interp ones
            #     name="large clusters",
            #     x=x_coords_interp[fade_out_masks],
            #     y=y_coords_interp[fade_out_masks],
            #     mode="markers",
            #     marker=dict(
            #         size=size,
            #         opacity=0.7,
            #         color=colors_fade_out,
            #     ),
            #     showlegend=False
            # ))


            nr_fade_out_edge = 0
            for edge in G_prev.edges():
                idx_0 = np.where(move_labels_prev == edge[0])[0][0]  # get first match
                idx_1 = np.where(move_labels_prev == edge[1])[0][0]  # get first match

                label_prev_0 = move_labels_prev[idx_0]
                label_next_0 = move_labels[idx_0]

                label_prev_1 = move_labels_prev[idx_1]
                label_next_1 = move_labels[idx_1]


                if tuple([label_next_0, label_next_1]) not in G_edges and tuple([label_next_1, label_next_0]) not in G_edges:
                    nr_fade_out_edge+=1


                    line_width_prev, opacity_prev = get_line_width_opac(edge_distances_prev.get((edge[0], edge[1])))
                    interp_opacity = opacity_prev + (0-opacity_prev) * nr_interp_frame/(nr_move_frames-1)


                    fig_layer.add_trace(go.Scatter(
                        x=[x_coords_interp[idx_0], x_coords_interp[idx_1]],
                        y=[y_coords_interp[idx_0], y_coords_interp[idx_1]],
                        mode="lines",
                        line=dict(
                            width=line_width_prev,
                            color= map_quadr_label_to_colors_prev[move_labels_prev[idx_0]]  # Use first color for the line. it's fading out anyway
                        ),

                        showlegend=False,
                        opacity=interp_opacity
                    ))



            ##########FADE IN ##########

            size = 4 + (8-4) * t

            # fig_layer.add_trace(go.Scatter(
            #     name="large clusters",
            #     x=x_coords_interp[fade_in_masks],
            #     y=y_coords_interp[fade_in_masks],
            #     mode="markers",
            #     marker=dict(
            #         size=size,
            #         opacity=0.7,
            #         color=colors_fade_in_interp,
            #         line=dict(width=0.5, color='white')
            #     ),
            #     showlegend=False
            # ))


            for edge in G.edges():
                idx_0 = np.where(move_labels == edge[0])[0][0]  # get first match
                idx_1 = np.where(move_labels == edge[1])[0][0]  # get first match

                label_prev_0 = move_labels_prev[idx_0]
                label_next_0 = move_labels[idx_0]

                label_prev_1 = move_labels_prev[idx_1]
                label_next_1 = move_labels[idx_1]


                if tuple([label_prev_0, label_prev_1]) not in G_prev_edges and tuple([label_prev_1, label_prev_0]) not in G_prev_edges:


                    line_width, opacity = get_line_width_opac(edge_distances.get((edge[0], edge[1])))


                    interp_opacity = 0 + (opacity - 0) * nr_interp_frame/(nr_move_frames-1)

                    fig_layer.add_trace(go.Scatter(
                        x=[x_coords_interp[idx_0], x_coords_interp[idx_1]],
                        y=[y_coords_interp[idx_0], y_coords_interp[idx_1]],
                        mode="lines",
                        line=dict(
                            width=line_width,
                            color= map_quadr_label_to_colors[move_labels[idx_0]]  # Use first color for the line
                        ),

                        showlegend=False,
                        opacity=interp_opacity
                    ))




            save_path = f"{layer_folder_path}\\image_{frame_nr:05}.png"

            print(save_path)
            fig_layer.write_image(save_path)

            frame_nr+=1





        clusters_prev, label_list_prev, G_indices_prev, X_latent_pca_prev, X_pred_label_prev, large_cluster_labels_prev, small_cluster_labels_prev, G_prev, components_prev, map_quadr_label_to_coords_prev, map_quadr_label_to_colors_prev, edge_distances_prev = clusters, label_list, G_indices, X_latent_pca, X_pred_label, large_cluster_labels, small_cluster_labels, G, components, map_quadr_label_to_coords, map_quadr_label_to_colors, edge_distances





plot_training_spacebent_all_smooth()



