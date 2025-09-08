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
pio.renderers.default = "browser"



hamming_threshold = 5

nr_neurons = 101
nr_digits = 10
activation = "sigmoid"
nr_epochs_train = 300

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


def plot_training_spacebent_all_smooth():
    
    global try_nr

    losses = np.load(f'trainings2\\training_digits_{try_nr}\\loss.npy')
    
    nr_layers = 4
    
  

    

    layer_nr = selected_layer
    predict_layer = 1

    X_latent = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy')
    X_pred = np.load(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{predict_layer}.npy')
    accuracies = np.load(f'trainings2\\training_digits_{try_nr}\\accuracy.npy')

    random_points = sigmoid_force_quad(np.random.uniform(-15, 15 ,  (100000, X_latent[0].shape[1])))

    print("total_saves", len(X_latent))
    print("total acc", len(accuracies))

    if len(X_latent) != len(accuracies):
        print("lengths not equal")
        exit()

    pca = PCA(n_components=3)
    pca = pca.fit(random_points) # X_latent[-1]-0.5
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
        
        
    for ii in range(2, len(accuracies), skip_smooth_interval):
        print(ii)


        angle_ = frame_nr * degrees_per_iter * np.pi/180
    

        X_latent_ii = X_latent[ii][::20]
        X_pred_ii = X_pred[ii][::20]

        if activation == "sigmoid":
            X_latent_ii = sigmoid_force_quad(X_latent_ii-0.5)
        elif activation == "relu":
            X_latent_ii = sigmoid_force_quad(X_latent_ii)




        X_latent_pca = trained_pca.transform(X_latent_ii)
        X_pred_label = np.argmax(X_pred_ii, axis=1)
        


        clusters, labels = quadrant_clusters(X_latent_ii)
        new_clusters = hamming_connected_components(clusters, max_hamming=hamming_threshold)

        fig_layer = go.Figure()

        # Separate indices for clusters >=3 and <=3
        large_cluster_indices = []
        small_cluster_indices = []
        for comp in new_clusters:
            if len(comp) >= 3:
                large_cluster_indices.extend(comp)
            else:
                small_cluster_indices.extend(comp)

        # Plot large clusters (size 2, opacity 0.9)
        if large_cluster_indices:
            fig_layer.add_trace(go.Scatter3d(
                name="large clusters",
                x=X_latent_pca[large_cluster_indices, 0],
                y=X_latent_pca[large_cluster_indices, 1],
                z=X_latent_pca[large_cluster_indices, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.7,
                    color=np.array(X_pred_label)[large_cluster_indices],
                    colorscale='rainbow'
                ),
                showlegend=False
            ))

        # Plot small clusters (size 1, opacity 0.3)
        if small_cluster_indices:
            fig_layer.add_trace(go.Scatter3d(
                name="small clusters",
                x=X_latent_pca[small_cluster_indices, 0],
                y=X_latent_pca[small_cluster_indices, 1],
                z=X_latent_pca[small_cluster_indices, 2],
                mode="markers",
                marker=dict(
                    size=5,
                    opacity=0.4,
                    color=np.array(X_pred_label)[small_cluster_indices],
                    colorscale='rainbow'
                ),
                showlegend=False
            ))

       
        #add lines here




        angle = angle_
        angle = angle - 90*np.pi/180
        vertical_angle = 45*np.pi/180
        r = 1.5


        #fig_layer.add_trace(go.Scatter3d(name = "sigmoid space", x=X_latent_pca[:,0], y=X_latent_pca[:,1],z = X_latent_pca[:,2], mode = "markers", marker = dict(opacity=0.7, size = 1, color=X_pred_label,colorscale='rainbow')))


        fig_layer.update_layout(scene_camera=dict(eye=dict(x=r*np.cos(vertical_angle)*np.cos(angle), y=r*np.cos(vertical_angle)*np.sin(angle), z=r*np.sin(vertical_angle))))
        #fig_layer.update_layout(scene_camera=dict(eye=dict(x=1.6*np.cos(angle)), y=1.6*np.sin(angle)), z=0.3)))
        fig_layer.update_layout( template = "plotly_dark")#, width = 1600, height = 1500)
        fig_layer.update_coloraxes(showscale=False)


        fig_layer.write_image(f"{animation_folder_path}\\layer_nr{layer_nr}\\image_{frame_nr:05}.png")

        
        print("framenr", frame_nr)
        fig_layer.show()
        exit()
        frame_nr+=1





plot_training_spacebent_all_smooth()



