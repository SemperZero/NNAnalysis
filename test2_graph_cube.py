import numpy as np
import networkx as nx
from collections import defaultdict
from scipy.special import expit  # Sigmoid function
import ast
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def analyze_nd_cube(points, tol=1e-5):
    """
    Analyzes points in an n-dimensional cube:
    - Assigns labels to points on vertices/edges.
    - Counts total vertices and edges.
    - Returns mappings for vertices/edges.
    """
    n_dim = points.shape[1]
    labels = np.full(len(points), -1, dtype=int)
    label_counter = 0
    label_to_description = {}
    label_to_description_vertex = {}
    label_to_description_edge = {}
    vertex_key_to_label = {}
    edge_key_to_label = {}
    vertex_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    points_map = {}

    for idx, point in enumerate(points):
        vertex_key = []
        is_vertex = True
        for coord in point:
            if abs(coord - 0) < tol:
                vertex_key.append(0)
            elif abs(coord - 1) < tol:
                vertex_key.append(1)
            else:
                is_vertex = False
                break
        
        if is_vertex:
            vertex_key = tuple(vertex_key)
            if vertex_key not in vertex_key_to_label:
                vertex_key_to_label[vertex_key] = label_counter
                label_to_description[label_counter] = vertex_key
                label_to_description_vertex[label_counter] = vertex_key
                label_counter += 1
            labels[idx] = vertex_key_to_label[vertex_key]
            vertex_counts[vertex_key] += 1
            continue

        n_free = 0
        edge_key = []
        for coord in point:
            if abs(coord - 0) < tol:
                edge_key.append(0)
            elif abs(coord - 1) < tol:
                edge_key.append(1)
            else:
                n_free += 1
                edge_key.append(None)
        
        if n_free == 1:
            edge_key = tuple(edge_key)
            if edge_key not in edge_key_to_label:
                edge_key_to_label[edge_key] = label_counter
                label_to_description[label_counter] = edge_key
                label_to_description_edge[label_counter] = edge_key
                label_counter += 1
            labels[idx] = edge_key_to_label[edge_key]
            edge_counts[edge_key] += 1
    
    return labels, label_to_description, label_to_description_vertex, label_to_description_edge, vertex_counts, edge_counts, vertex_key_to_label, edge_key_to_label

# def construct_graph(label_to_description): # if two edges/vertices share a common coordinate in space we consider them connected. this is very loose in small dimensionality
#     """Construct a graph where nodes are vertices/edges and edges are based on shared components."""
#     G = nx.Graph()
#     for label, desc in label_to_description.items():
#         G.add_node(label, description=desc)
    
#     for label1, desc1 in label_to_description.items():
#         for label2, desc2 in label_to_description.items():
#             if label1 != label2: #and any( a== b for a, b in zip(ast.literal_eval(desc1.split('_')[1]), ast.literal_eval(desc2.split('_')[1])) ):
#                 nr_coords = nr_dim
#                 for a, b in zip(ast.literal_eval(desc1.split('_')[1]), ast.literal_eval(desc2.split('_')[1])) :
#                     if a != b:
#                         nr_coords -= 1    
                    
                
#                 if nr_coords == nr_dim-1:
#                     G.add_edge(label1, label2)
                        
    
#     #print(G)
#    # exit()
#     return G

def construct_graph(label_to_description, label_to_description_vertex, label_to_description_edge, nr_dim, vertex_key_to_label, edge_key_to_label, vertex_counts): # two vertices need to have an edge between them for the 3 components to be connected
    """Construct a graph where nodes are vertices/edges and edges are based on shared components."""
    G = nx.Graph()
    for label, desc in label_to_description_vertex.items():
        if vertex_counts[label] >=100:
            G.add_node(label, description=desc)
    
    MIN_NR_POINTS_ON_EDGE = 100

    for label1, desc1 in label_to_description_vertex.items():
        for label2, desc2 in label_to_description_vertex.items():
            if label1 != label2: #and any( a== b for a, b in zip(ast.literal_eval(desc1.split('_')[1]), ast.literal_eval(desc2.split('_')[1])) ):
                nr_coords = nr_dim
                has_common_edge = False
                for i, (a, b) in enumerate(zip(desc1, desc2)) :
                    if a != b:
                        nr_coords -= 1
                        edge_key = list(desc1)
                        edge_key[i] = None
                        edge_key = tuple(edge_key)
                       # print(edge_key)
                        if edge_key in edge_key_to_label:
                            if edge_counts[edge_key] >= MIN_NR_POINTS_ON_EDGE:
                                has_common_edge = True
                                edge_label = edge_key_to_label[edge_key]
                    
                    
                    if nr_coords < nr_dim-1: # we encountered a second difference
                        break
                    
                
                if nr_coords == nr_dim-1 and has_common_edge:
                    G.add_edge(label1, label2)
                    G.add_edge(label1, edge_label)
                    G.add_edge(label2, edge_label)

                    
                        
    

    return G

def get_connected_components(G):
    """Return labels of connected components in the graph."""
    return [list(component) for component in nx.connected_components(G)]

def find_index_list(lst, num):
    for i, sublist in enumerate(lst):
        if num in sublist:
            return i  # Return the index of the first matching sublist
    return -1  # Return -1 if the number is not found

# Example usage
# nr_dim = 10  # Dimension
# points = expit(np.random.uniform(-15, 15 ,  (100000, nr_dim)))

try_nr = 105
nr_digits = 10

model = tf.keras.models.load_model(f'training_digits_{try_nr}\\model_filename_{nr_digits}.h5')
model.summary()

print(model.layers)

selected_layer = 1

# nr_dim = model.get_weights()[1].shape[1]
# print(nr_dim)
# exit()


model_1 = models.Sequential(model.layers[:selected_layer+1])
model_2 = models.Sequential(model.layers[selected_layer+1:])

random_points_full = np.random.uniform(-10, 10 ,  (1000000, 28*28))

random_points = model_1.predict(random_points_full)

nr_dim = random_points.shape[1]


# points = np.array([
#     [0,0,0],
#     [0,0,0.5],
#     [0,0,1],


#     [1,0,0],
#     [1,0,0.5],
#     [1,0,1],

#     [0,1,0],
#     [0.5,1,0],
#     [1,1,0],

#     [0,1,1],
#     [0.5,1,1],
#     [1,1,1],
# ])
tol = 0.01  # Tolerance for classification

labels, label_to_description, label_to_description_vertex, label_to_description_edge, vertex_counts, edge_counts, vertex_key_to_label, edge_key_to_label = analyze_nd_cube(random_points, tol)
print("nr_vertex", sum(vertex_counts.values()))
print("nr_edge", sum(edge_counts.values()))
print("other", len(labels) - sum(vertex_counts.values()) - sum(edge_counts.values()) )
G = construct_graph(label_to_description, label_to_description_vertex, label_to_description_edge, nr_dim, vertex_key_to_label, edge_key_to_label, vertex_counts)
components = get_connected_components(G)

#print("Connected Components:", components)


connected_labels_map = {}
connected_labels_map[-1] = -1 
for label in label_to_description:
    connected_labels_map[label] = find_index_list(components, label)




connected_labels = [connected_labels_map[x] for x in labels]

counter = Counter(connected_labels)

MIN_NR_POINTS_PER_CLUSTER = 100
for value, count in counter.items():
    if count < MIN_NR_POINTS_PER_CLUSTER:
       connected_labels = [-2 if x == value else x for x in connected_labels]

counter = Counter(connected_labels)
for value, count in counter.items():
    print(f"{value}: {count}")
   
connected_labels = np.array(connected_labels)


pca = PCA(n_components=3)
pca_result_full = pca.fit_transform(random_points)#[:100_000])




# fig = go.Figure()

# fig.add_trace(
#     go.Scatter3d(
#         name = f"{label}: {counter[label]}",
#         x=pca_result_full[:, 0], 
#         y=pca_result_full[:, 1], 
#         z=pca_result_full[:, 2],
#         mode='markers',
#         marker=dict(size=1, opacity=1, color = connected_labels, colorscale = "rainbow"  )
# ))









fig = go.Figure()

for label in list(np.unique(connected_labels)):
    if counter[label] < 1000:
        continue
    print(label)

    indices = np.where(connected_labels == label)[0]
    print(indices)
    
    pca_result = pca_result_full[indices]
    print(len(pca_result), len(indices))

    

    fig.add_trace(
        go.Scatter3d(
            name = f"{label}: {counter[label]}",
            x=pca_result[:, 0], 
            y=pca_result[:, 1], 
            z=pca_result[:, 2],
            mode='markers',
            marker=dict(size=1, opacity=1 )
        # marker=dict(size=1, opacity=1, color = connected_labels, colorscale = "rainbow"  )
    ))

fig.update_layout(template = "plotly_dark", title="3D PCA Projection of Sigmoid-Transformed Data")
fig.show()


for label in list(np.unique(connected_labels)):
    print(label)


    
    
    class_indices = np.where(connected_labels == label)[0]
    
    average_vector = np.mean(random_points_full[class_indices], axis=0) # TODO:  check this thoroughly

    
    reconstructed_image = average_vector.reshape(28, 28)
    dpi = 100
    width_inch = 1000 / dpi
    height_inch = 1000 / dpi

    # Set the figure size in inches and DPI
    plt.figure(figsize=(width_inch, height_inch), dpi=dpi)

    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed Image for Predicted Class {label} : {len(class_indices)}")
    plt.axis("off")  # Remove axes for a cleaner image
    plt.show()