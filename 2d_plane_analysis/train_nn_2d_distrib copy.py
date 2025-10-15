import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import Callback
import os
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tensorflow.keras import Model, Input, Sequential, layers
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, AdamW


import plotly.io as pio
pio.renderers.default = "browser"

FIT_MODEL = True
USE_CALLBACK = True


nr_digits = 10

nr_neurons = 300
nr_inner_neurons = nr_neurons
activation = "relu"
nr_epochs_train = 70006
try_nr = f'plane_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'


np.random.seed(40)
tf.random.set_seed(40)


# Example usage


# === Input function to generate data ===
def generate_dataset(n_zones=10, points_per_zone=1, n_classes=3, min_distance=0.05, nr_dim = 2):
    def generate_non_overlapping_zones(existing_centers):
        while True:
            candidate = np.random.uniform(-1, 1, size=nr_dim)
            if all(distance.euclidean(candidate, c) >= min_distance for c in existing_centers):
                return candidate

    def generate_class_points_no_overlap(label, existing_centers):
        points = []
        for _ in range(n_zones):
            center = generate_non_overlapping_zones(existing_centers)
            existing_centers.append(center)
            spread = np.random.uniform(0.002, 0.005)
            zone_points = np.random.normal(loc=center, scale=spread, size=(points_per_zone, nr_dim))
            points.append(zone_points)
        points = np.vstack(points)
        labels = np.full((points.shape[0],), label)
        return points, labels, existing_centers

    existing_centers = []
    all_points, all_labels = [], []
    for label in range(n_classes):
        class_points, class_labels, existing_centers = generate_class_points_no_overlap(label, existing_centers)
        all_points.append(class_points)
        all_labels.append(class_labels)

    X = np.vstack(all_points).astype(np.float32)
    y = np.hstack(all_labels).astype(np.int64)
    return X, y


def get_model_no_final_activation(model):
    # copy all layers except last
    new_layers = []
    for layer in model.layers[:-1]:
        new_layers.append(layer.__class__.from_config(layer.get_config()))
    
    # final Dense layer without activation
    last_layer = model.layers[-1]
    last_config = last_layer.get_config()
    last_config['activation'] = None
    new_layers.append(last_layer.__class__.from_config(last_config))
    
    # build new model
    model_no_final = Sequential(new_layers)
    model_no_final.build(model.input_shape)
    
    # copy weights
    for l_new, l_old in zip(model_no_final.layers, model.layers):
        l_new.set_weights(l_old.get_weights())
    
    return model_no_final


def get_model_without_final_activation(model, layer_index=-1):
    orig_layers = model.layers
    n = len(orig_layers)

    # normalize negative index
    if layer_index < 0:
        layer_index = n + layer_index
    if not (0 <= layer_index < n):
        raise IndexError(f"layer_index {layer_index} out of range for model with {n} layers")

    new_layers = []
    for i in range(layer_index + 1):
        old = orig_layers[i]
        cfg = old.get_config()
        if i == layer_index and 'activation' in cfg:
            cfg['activation'] = None
        new_layer = old.__class__.from_config(cfg)
        new_layers.append(new_layer)

    new_model = Sequential(new_layers)
    new_model.build(model.input_shape)

    # copy weights only for layers that actually have weights
    for new_l, old_l in zip(new_model.layers, orig_layers[: layer_index + 1]):
        w = old_l.get_weights()
        if w:
            new_l.set_weights(w)

    return new_model

def generate_spiral(n_points_per_class=100, noise=0.02, n_turns=3):
    n = np.arange(0, n_points_per_class)
    theta = np.linspace(0, n_turns * np.pi, n_points_per_class)
    r = np.linspace(0.0, 1.0, n_points_per_class) ** 0.5

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
        np.zeros(n_points_per_class),
        np.ones(n_points_per_class)
    ])

    # Add Gaussian noise
    data += np.random.randn(*data.shape) * noise

    return data, labels


def generate_spiral2(n_points_per_class=100, base_noise=0.02, n_turns=3):
    theta = np.linspace(0, n_turns * np.pi, n_points_per_class)
    r = np.linspace(0.0, 1.0, n_points_per_class)

    x1 = r * np.sin(theta)
    y1 = r * np.cos(theta)
    x2 = -r * np.sin(theta)
    y2 = -r * np.cos(theta)

    data = np.vstack([
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1)
    ])
    labels = np.hstack([
        np.zeros(n_points_per_class),
        np.ones(n_points_per_class)
    ])

    # Gradual noise: scaled by radius
    radii = np.tile(r, 2).reshape(-1, 1)
    noise = np.random.randn(*data.shape) * (base_noise * (0.3 + 0.7 * radii))
    data += noise

    return data, labels

# === Generate Data ===
#x_train, y_train = generate_dataset(n_zones=10, points_per_zone=1, n_classes=3, min_distance=0.05, nr_dim = 2)


x_train, y_train = generate_spiral2(n_points_per_class=1000, base_noise=0.02, n_turns=8)


print("len(x_train)", len(x_train))

print(y_train)

fig = go.Figure()

fig.add_trace(go.Scatter(x = x_train[:,0], y = x_train[:,1], mode = "markers", marker = dict(size = 10, color = y_train, colorscale = "Rainbow")))
fig.update_layout(template = "plotly_dark", width = 1500, height = 1500)
fig.show()
#exit()

# === TensorFlow Dataset ===
X_tensor = tf.convert_to_tensor(x_train)
y_tensor = tf.convert_to_tensor(y_train)

nr_classes = 2
y_onehot = tf.keras.utils.to_categorical(y_train, num_classes=nr_classes)



model = models.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(nr_inner_neurons, activation=activation),
    layers.Dense(2, activation='softmax')
])





def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)


batch_global = -1

u = np.linspace(-1, 1, 200)
v = np.linspace(-1, 1, 200)
U, V = np.meshgrid(u, v)
points_grid = np.stack([U.ravel(), V.ravel()], axis=1)  # shape (N, 2)
    

class PredictionCallback(Callback):
    def __init__(self, X_data, X_full_space, nr_layers):
        super().__init__()
        self.X_data = X_data
        self.predictions_batch = []
        self.predictions_epoch = []
        self.weights_batch = []
        self.weights_epoch = []
        self.loss_epoch = []
        self.loss_batch = []
        self.accuracy_epoch = []  # new
        self.accuracy_batch = []  # new
        self.space_bent_output_layers = {}
        self.X_bent_output_layers = {}
        self.layer_weights = {}
        for layer_nr in range(nr_layers):
            self.space_bent_output_layers[layer_nr] = []
            self.X_bent_output_layers[layer_nr] = []
            self.layer_weights[layer_nr] = []

        self.save_every = 1000
        self.X_full_space = X_full_space

        



    def save_data(self):
        print("saving data")
        if not os.path.isdir(f'trainings2\\training_digits_{try_nr}'):
            os.makedirs(f'trainings2\\training_digits_{try_nr}')

        np.save(f'trainings2\\training_digits_{try_nr}\\loss.npy', self.loss_batch)
        np.save(f'trainings2\\training_digits_{try_nr}\\accuracy.npy', self.accuracy_batch)  # save batch accuracy

        for i in [0, 1]:  # for each selected layer
            np.save(f'trainings2\\training_digits_{try_nr}\\X_bent_output_layers_{i}.npy', self.X_bent_output_layers[i])
            np.save(f'trainings2\\training_digits_{try_nr}\\space_bent_output_layers_{i}.npy', self.space_bent_output_layers[i])

    def on_batch_end(self, batch, logs=None):
        global batch_global
        batch_global += 1
        
        # if batch_global < 10:
        #     self.save_every = 1
        # else:
        #     self.save_every = 10

        self.loss_batch.append(logs.get('loss'))
        self.accuracy_batch.append(logs.get('accuracy'))  # save batch accuracy

        if batch_global % self.save_every <= 30:
            for selected_layer in [0, 1]:
              #  print("selected_layer", selected_layer)

               # penultimate_layer_model = models.Sequential(self.model.layers[:selected_layer + 1])
                
                penultimate_layer_model = get_model_without_final_activation(self.model, selected_layer)
                
                X_bent_output = penultimate_layer_model.predict(self.X_data, verbose = False)
                self.X_bent_output_layers[selected_layer].append(X_bent_output)

                X_full_space_bent_output = penultimate_layer_model.predict(self.X_full_space, verbose = False)
                self.space_bent_output_layers[selected_layer].append(X_full_space_bent_output)


        if batch_global % (10 *self.save_every)  == 0:
            self.save_data()








prediction_callback = PredictionCallback(X_tensor, points_grid, len(model.layers))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

if FIT_MODEL:
    if USE_CALLBACK:
        model.fit(
            X_tensor,
            y_onehot,  # use one-hot labels
            epochs=nr_epochs_train,
            batch_size=32,
            validation_split=0.1,
            callbacks=[prediction_callback]
        )
    else:
        model.fit(
            X_tensor,
            y_onehot,  # use one-hot labels
            epochs=nr_epochs_train,
            batch_size=32 ,
            validation_split=0.1
        )




        #7001 is 1 ; 7002 is 32, 