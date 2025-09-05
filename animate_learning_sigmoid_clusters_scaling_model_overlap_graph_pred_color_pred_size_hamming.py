#how would you suggest to approach a problem where there are 2 million nodes with bitstrings as labels ad we want to connect 2 nodes if their hamming distance is less than a threshold, then construct a graph and get the positions of each node so that the more connected components will be closer to each other? the actual problem starts with having 60k nodes that shift labels every iteration, to a total of 2 million. each iteration we want to get the connected components and display the graph. we want to see how this graph evolves across iterations, but we'd like the connected nodes to be close to each other across iterations so that the nodes shift across iterations to nearby nodes. what is a way to achieve this considering that we can't fully represent the 2mil nodes graph and then get the nodes location on a 2d plane to then shift across them, because the edge computation is in n^2 (with the hamming dist). we can afford to lose some accuracy on the placement of the nodes


import numpy as np
import os
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import tensorflow as tf
import plotly.io as pio
import plotly.express as px
from scipy.spatial.distance import hamming
import networkx as nx
from scipy.spatial.distance import pdist, squareform
pio.renderers.default = "browser"


SHOW_TRAINING_POINTS = True

nr_neurons = 101
nr_digits = 10
activation = "sigmoid"
nr_epochs_train = 300

try_nr = f'digits_{nr_digits}_nodes_{nr_neurons}_{activation}_epoc_{nr_epochs_train}'
selected_layers = [0]

# Load model
# model = tf.keras.models.load_model(
#     f"trainings2\\training_digits_{try_nr}\\model_filename_{nr_digits}.h5"
# )
# model.summary()

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0
x_train_gray, y_train_mapped = x_train_flattened, y_train


x_train_gray = x_train_gray[::2]
y_train_mapped = y_train_mapped

y_labels = y_train_mapped[::2]

if activation == "sigmoid":
    offset_value = 0.5
elif activation == "relu":
    offset_value = 0.0

def sigmoid(z):
    #return 1 / (1 + np.exp(-z))
    return np.where(z < 0, 0, 1)

def bin_counts(x, y, labels, bins=40):
    """Return counts of labels per bin."""
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    bin_indices_x = np.digitize(x, xedges) - 1
    bin_indices_y = np.digitize(y, yedges) - 1

    counts = {}
    for bx, by, lbl in zip(bin_indices_x, bin_indices_y, labels):
        if bx < 0 or by < 0 or bx >= bins or by >= bins:
            continue
        key = (bx, by)
        if key not in counts:
            counts[key] = {}
        counts[key][lbl] = counts[key].get(lbl, 0) + 1

    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2
    bin_centers_y = (yedges[:-1] + yedges[1:]) / 2

    return counts, bin_centers_x, bin_centers_y

def normalize_size_linear(count, min_count, max_count, min_size=5, max_size=30):
    """Linearly normalize count into a visual marker size between min_size and max_size."""
    if max_count == min_count:
        return (min_size + max_size) / 2  # avoid division by zero
    return ((count - min_count) / (max_count - min_count)) * (max_size - min_size) + min_size

def get_quadrant_labels(X):
    """Assign a bitstring label to each sample based on sign(X-0.5)."""
    # X: (n_samples, n_features)
    return np.packbits((X > offset_value).astype(np.uint8), axis=1, bitorder='little').flatten()

def bitstring_labels(X_hd, offset_value):
    """Return list of bitstrings for each sample."""
    return [''.join(str(int(b)) for b in (row > offset_value)) for row in X_hd]

def quadrant_clusters(X_hd, offset_value):
    """Return a dict mapping quadrant label to list of indices."""
    labels = bitstring_labels(X_hd, offset_value)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)
    return clusters, labels

def hamming_connected_components(clusters, max_hamming=5):
    """Group clusters whose bitstring labels are within max_hamming distance."""
    labels = list(clusters.keys())
    G = nx.Graph()
    G.add_nodes_from(labels)
    for i, l1 in enumerate(labels):
        print(i, len(labels))
        for j in range(i+1, len(labels)):
            l2 = labels[j]
            # Hamming distance between bitstrings
            dist = sum(a != b for a, b in zip(l1, l2))
            if dist <= max_hamming:
                G.add_edge(l1, l2)
    # Each connected component is a new cluster
    new_clusters = []
    for comp in nx.connected_components(G):
        indices = []
        for label in comp:
            indices.extend(clusters[label])
        new_clusters.append(indices)
    return new_clusters

def compute_overlap_accuracy(clusters, y_labels):
    """
    For each quadrant cluster, assign the majority class as the predicted label for all items in the cluster.
    Return the fraction of items where predicted == true label.
    """
    
    correct = 0
    total = 0
    for indices in clusters.values():
        cluster_labels = y_labels[indices]
        if len(cluster_labels) == 0:
            continue
        # Find majority class in this cluster
        values, counts = np.unique(cluster_labels, return_counts=True)

      #  print(values)
      #  print(counts)
     #   print()

        majority_class = values[np.argmax(counts)]
        # Count correct predictions in this cluster
        correct += np.sum(cluster_labels == majority_class)
        total += len(cluster_labels)
    if total == 0:
        return 0.0
    return correct / total

def plot_training_spacebent_all_binned():
    global try_nr

    accuracies = np.load(f"trainings2\\training_digits_{try_nr}\\accuracy.npy")[:100][::2]

    X_bent_output_layers_0 = {}
    X_bent_output_layers_1 = {}
    random_points = {}

    for layer_nr in selected_layers:
        X_bent_output_layers_0[layer_nr] = np.load(f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy")[:100]
        random_points[layer_nr] = sigmoid(np.random.uniform(-100, 100, (20000, X_bent_output_layers_0[layer_nr].shape[2])))

    # Load layer 1 outputs for predictions
    X_bent_output_layers_1[0] = np.load(f"trainings2\\training_digits_{try_nr}\\X_bent_output_layers_1.npy")[:100]

    if len(accuracies) > 1.9* len(X_bent_output_layers_0[selected_layers[0]]): # bugfix
        accuracies = accuracies[::2]

    trained_pcas = {}
    for layer_nr in selected_layers:
        pca = PCA(n_components=2).fit(X_bent_output_layers_0[layer_nr][-1][::2])
        trained_pcas[layer_nr] = pca
        random_points[layer_nr] = pca.transform(random_points[layer_nr])

    X_bent_output_layers = {}
    for layer_nr in selected_layers:
        X_bent_output_layers[layer_nr] = []
        for i in range(len(X_bent_output_layers_0[layer_nr][::2])):
            X_bent_output_layers[layer_nr].append(
                trained_pcas[layer_nr].transform(
                    sigmoid(1 * (X_bent_output_layers_0[layer_nr][i][::2] - offset_value))
                )
            )

    # --- Compute global min and max counts for marker size normalization (per predicted digit in new clusters) ---
    # all_counts = []
    # for layer_nr in selected_layers:
    #     for i in range(len(X_bent_output_layers_0[layer_nr])):
    #         # Get predicted digits for this frame
    #         layer1_outputs = X_bent_output_layers_1[0][i]
    #         predicted_digits = np.argmax(layer1_outputs, axis=1)
    #         clusters = quadrant_clusters(X_bent_output_layers_0[layer_nr][i], offset_value)
    #         new_clusters = hamming_connected_components(clusters, max_hamming=5)
    #         for indices in new_clusters:
    #             pred_digits_in_cluster = predicted_digits[indices]
    #             unique_pred_digits, pred_digit_counts = np.unique(pred_digits_in_cluster, return_counts=True)
    #             all_counts.extend(pred_digit_counts.tolist())

    # p99_count = np.percentile(all_counts, 98)
    # all_counts = [min(x, p99_count) for x in all_counts]

    min_count = 10#min(all_counts)
    max_count = 1#max(all_counts)
    p99_count = 10
    print(f"Global min count (per predicted digit in cluster): {min_count}, max count: {max_count}")

    # Plot the distribution of all circle sizes from all_counts
    # fig = px.histogram(
    #     x=all_counts,
    #     nbins=50000,
    #     title="Distribution of circle sizes (predicted digit counts in clusters)",
    #     labels={"x": "Count", "y": "Frequency"},
    # )

    # fig.update_traces(marker=dict(color="skyblue", line=dict(color="black", width=1)))
    # fig.update_layout(
    #     bargap=0,
    #     title_x=0.5
    # )

    # fig.show()

    frame_nr = 0
    nr_interp_frames = 1
    skip_smooth_interval = 1

    animation_folder_path = (
        f"animations2\\training_learning_binned_{try_nr}_{nr_interp_frames}_"
        f"{skip_smooth_interval}x10000"
    )
    os.makedirs(animation_folder_path, exist_ok=True)

    for layer_nr in selected_layers:
        os.makedirs(f"{animation_folder_path}\\layer_nr{layer_nr}_pred_color_pred_size", exist_ok=True)

    colorscales = [
        "Viridis", "Cividis", "Plasma", "Magma", "Inferno",
        "Turbo", "Blues", "Greens", "Reds", "Oranges"
    ]

    # Pre-selected colors for digits 0-9 (from rainbow colormap)
    mapped_colors = [
        "#9400D3",  # violet
        "#4B0082",  # indigo
        "#0000FF",  # blue
        "#00FF00",  # green
        "#FFFF00",  # yellow
        "#FF7F00",  # orange
        "#FF0000",  # red
        "#FF1493",  # deep pink
        "#00CED1",  # dark turquoise
        "#FFD700",  # gold
    ]




    prev_graph = None
    prev_labels = None

    # Track number of unique quadrant clusters per frame
    # unique_clusters = set()
    

    # unique_cluster_counts = []
    # increases = []
    # for ii in range(0, len(accuracies), skip_smooth_interval):
    #     print(ii, len(accuracies))

        
    #     # Track number of unique clusters for this frame
    #     clusters, labels = quadrant_clusters(X_bent_output_layers_0[layer_nr][ii], offset_value)
    #     old_leng = len(unique_clusters)
    #     unique_clusters.update(labels)

    #     increases.append(len(unique_clusters) - old_leng)
    #     unique_cluster_counts.append(len(unique_clusters))

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(y=unique_cluster_counts, mode='lines+markers'))
    # fig.add_trace(go.Scatter(y=increases, mode='lines+markers'))
    # fig.show()
    # exit()

    for ii in range(0, len(accuracies), skip_smooth_interval):
        print(ii, len(accuracies))

        for j in range(nr_interp_frames):
            fig_layer_all = {}
            for layer_nr in selected_layers:
                x_training_pca = X_bent_output_layers[layer_nr][ii]

                # Get predicted digits from layer 1 for current frame
                layer1_outputs = X_bent_output_layers_1[0][ii]  # shape: (n_samples, n_digits)
                predicted_digits = np.argmax(layer1_outputs, axis=1)  # shape: (n_samples,)

                fig_layer = go.Figure()

                x_training_latent = X_bent_output_layers_0[layer_nr][ii][::2]

                clusters, labels = quadrant_clusters(x_training_latent, offset_value)

                arr = np.array([list(l) for l in labels], dtype="U1").view(np.uint32).reshape(len(labels), -1)

                # Compute pairwise Hamming distances (fraction of differing positions)
                dists = squareform(pdist(arr, metric="hamming")) * arr.shape[1]

                # Build graph for threshold ≤ 15
                prev_graph = nx.Graph()
                prev_graph.add_nodes_from(labels)
                rows, cols = np.where((dists <= 15) & (dists > 0))
                edges = [(labels[i], labels[j]) for i, j in zip(rows, cols) if i < j]
                prev_graph.add_edges_from(edges)

                print(prev_graph)
                print("start connected components")
                new_clusters = get_connected_components(prev_graph, clusters)
                print("done connected components")
                prev_labels = labels

                # ...existing code for cluster_circles, plotting, etc...
                cluster_circles = []
                for k, indices in enumerate(new_clusters):
                    subset = x_training_pca[indices]
                    mean_coord = subset.mean(axis=0)
                    pred_digits_in_cluster = predicted_digits[indices]
                    unique_pred_digits, pred_digit_counts = np.unique(pred_digits_in_cluster, return_counts=True)
                    for pred_digit, count in zip(unique_pred_digits, pred_digit_counts):
                        count = min(count, p99_count)
                        size = normalize_size_linear(count, min_count, max_count, min_size=5, max_size=30)
                        cluster_circles.append({
                            "mean_coord": tuple(np.round(mean_coord, 8)),
                            "size": size,
                            "indices": indices,
                            "k": k,
                            "pred_digit": int(pred_digit),
                            "count": int(count),
                        })

                # Group by (mean_coord, size) for overlap detection
                from collections import defaultdict
                grouped = defaultdict(list)
                for c in cluster_circles:
                    grouped[(c["mean_coord"], c["size"])].append(c)

                # Draw circles, offsetting radii if needed (order by predicted digit)
                min_offset = 1  # minimal visible offset
                for group in grouped.values():
                    if len(group) == 1:
                        c = group[0]
                        fig_layer.add_trace(
                            go.Scatter(
                                x=[c["mean_coord"][0]],
                                y=[c["mean_coord"][1]],
                                mode="markers",
                                marker=dict(
                                    size=c["size"],
                                    color=mapped_colors[c["pred_digit"]],
                                    opacity=0.9,
                                    line=dict(width=1, color="black"),
                                    symbol="circle-open",
                                ),
                                name=f"cluster {c['k']} pred_digit {c['pred_digit']}",
                                showlegend=False,
                            )
                        )
                    else:
                        # Sort by predicted digit for consistent offset ordering
                        group_sorted = sorted(group, key=lambda x: x["pred_digit"])
                        for idx, c in enumerate(group_sorted):
                            offset = (idx - (len(group_sorted) - 1) / 2) * min_offset
                            fig_layer.add_trace(
                                go.Scatter(
                                    x=[c["mean_coord"][0]],
                                    y=[c["mean_coord"][1]],
                                    mode="markers",
                                    marker=dict(
                                        size=c["size"] + offset,
                                        color=mapped_colors[c["pred_digit"]],
                                        opacity=0.9,
                                        line=dict(width=1, color="black"),
                                        symbol="circle-open",
                                    ),
                                    name=f"cluster {c['k']} pred_digit {c['pred_digit']}",
                                    showlegend=False,
                                )
                            )

                # Compute overlap_accuracy for this frame/layer (optional: can be removed if not needed)
                overlap_acc = compute_overlap_accuracy(clusters, y_labels)

                if SHOW_TRAINING_POINTS:
                    #1. Background random points (gray)
                    fig_layer.add_trace(
                        go.Scatter(
                            x=random_points[layer_nr][:, 0],
                            y=random_points[layer_nr][:, 1],
                            mode="markers",
                            marker=dict(size=1.5, opacity=0.4, color="gray"),
                            name="random space",
                            showlegend=False,
                        )
                    )

                fig_layer.update_layout(
                    title = f"Frame {ii}",
                    template="plotly_dark",
                    width=1600,
                    height=1500,
                    xaxis_title="PC1",
                    yaxis_title="PC2",
                )
                fig_layer_all[layer_nr] = fig_layer

            for layer_nr in selected_layers:
                out_path = (
                    f"{animation_folder_path}\\layer_nr{layer_nr}_pred_color_pred_size\\image_{frame_nr:05}.png"
                )
                fig_layer_all[layer_nr].write_image(out_path)
                print(f"saved to {out_path}")

            frame_nr += 1


def str_to_int_array(labels):
    """Convert list of binary strings into a NumPy array of integers."""
    return np.array([int(l, 2) for l in labels], dtype=np.uint64)

def update_hamming_graph(prev_graph, prev_labels, new_labels, max_hamming=15):
    """
    Incrementally update graph using vectorized bitwise XOR + popcount.
    """
    new_labels = set(new_labels)
    prev_labels = set(prev_labels)

    # Remove old nodes
    to_remove = prev_labels - new_labels
    prev_graph.remove_nodes_from(to_remove)

    # Add new nodes
    to_add = new_labels - prev_labels
    prev_graph.add_nodes_from(to_add)

    if not to_add:
        return prev_graph

    # Map labels → int
    all_nodes = list(prev_graph.nodes)
    all_ints = str_to_int_array(all_nodes)

    to_add_nodes = list(to_add)
    to_add_ints = str_to_int_array(to_add_nodes)

    # Vectorized distances:
    # XOR broadcast (len(to_add), len(all_nodes))
    xor_matrix = np.bitwise_xor(to_add_ints[:, None], all_ints[None, :])

    # Popcount across matrix
    dists = np.vectorize(int.bit_count)(xor_matrix)

    # Mask distances
    mask = (dists > 0) & (dists <= max_hamming)

    # Build edges
    rows, cols = np.where(mask)
    edges = [(to_add_nodes[i], all_nodes[j]) for i, j in zip(rows, cols)]
    prev_graph.add_edges_from(edges)

    return prev_graph

def get_connected_components(graph, clusters):
    """Return list of indices for each connected component."""


    new_clusters = []
    for comp in nx.connected_components(graph):
        indices = []
        for label in comp:
            indices.extend(clusters[label])
        new_clusters.append(indices)
    return new_clusters

plot_training_spacebent_all_binned()
