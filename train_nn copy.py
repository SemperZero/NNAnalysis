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

FIT_MODEL = True
USE_CALLBACK = True
nr_digits = 10  # Change this to 3 if needed
try_nr = 96
# 16 is with relu, 17 is full sigmoid, 18 is full relu

#for sig_dim in [6, 8, 12, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]:
    #try_nr+=1

# 94 has 15 on relu, 95 has 100


nr_epochs_train = 300

model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(15, activation='relu'),
    layers.Dense(7, activation='sigmoid'),
    #layers.Dense(7, activation='relu'),
    layers.Dense(nr_digits, activation='softmax')
])

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)


def shuffle_dataset(x,y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return x[indices], y[indices]


def sample_equal_digits(x, y, digits, samples_per_digit):
    filtered_x, filtered_y = [], []
    for digit in digits:
        indices = np.where(y == digit)[0][:samples_per_digit]  # Take first N samples per digit
        filtered_x.append(x[indices])
        filtered_y.append(y[indices])
    return np.concatenate(filtered_x), np.concatenate(filtered_y)

#nr_samples_per_class = 10000

# if nr_digits == 3:
#     selected_digits = [9, 5, 8]
#     x_train, y_train = sample_equal_digits(x_train, y_train, selected_digits, nr_samples_per_class)
#     x_test, y_test = sample_equal_digits(x_test, y_test, selected_digits, nr_samples_per_class)
#     y_train = np.where(y_train == 9, 0, np.where(y_train == 5, 1, 2))  # 9->0, 5->1, 8->2
#     y_test = np.where(y_test == 9, 0, np.where(y_test == 5, 1, 2))      # Same mapping for test set
# elif nr_digits == 10:
#     selected_digits = list(range(10))  # All digits 0-9
#     x_train, y_train = sample_equal_digits(x_train, y_train, selected_digits, nr_samples_per_class)
#     x_test, y_test = sample_equal_digits(x_test, y_test, selected_digits, nr_samples_per_class)

x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0

x_train_gray, y_train_mapped = x_train_flattened, y_train
x_test_gray, y_test_mapped = x_test_flattened, y_test



def apply_grayshift(x_train_gray, y_train_mapped):
    x = []
    y = []
    for i in np.linspace(0, 1, 11):
        x.append(i * x_train_gray)
        y.append(y_train_mapped)
        

    x = np.vstack(x) 
    y = np.concatenate(y)


    return x, y
    

#x_train_gray, y_train_mapped = apply_grayshift(x_train_gray, y_train_mapped)

#x_train_gray, y_train_mapped = shuffle_dataset(x_train_gray, y_train_mapped)

batch_global = -1

def fibonacci_sphere(samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # Radius at y

        theta = phi * i  # Golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def get_interpolated_from_origin_trainingset(original_vectors):
    n_samples = 1000
    selected_vectors = original_vectors[np.random.choice(original_vectors.shape[0], n_samples, replace=False)]

    n_interpolations = 100
    origin = np.zeros(selected_vectors.shape[1])  # Origin point in the same dimension as the vectors

    interpolated_vectors = []

    for vector in selected_vectors:
        interpolations = np.linspace(origin, vector, n_interpolations)
        interpolated_vectors.extend(interpolations)

    interpolated_vectors = np.array(interpolated_vectors)
    return interpolated_vectors

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
        self.loss_epoch = []
        self.space_bent_output_layers = {}
        self.space_bent_output_labels_layers = {}
        self.X_bent_output_layers = {}
        self.layer_weights = {}
        for layer_nr in range(nr_layers):
            self.space_bent_output_layers[layer_nr] = []
            self.space_bent_output_labels_layers[layer_nr] = []
            self.X_bent_output_layers[layer_nr] = []
            self.layer_weights[layer_nr] = []
            
        self.save_every = 1
        self.X_full_space = X_full_space
        

    def on_epoch_end(self, epoch, logs=None):
        return
        if epoch % self.save_every == 0:
            
            current_predictions = self.model.predict(self.X_data)
            self.predictions_epoch.append(current_predictions)

            self.loss_epoch.append(logs.get('loss'))

            weights = []
            for layer in self.model.get_weights():
                #print(layer.shape)
                weights.append(layer.flatten())
                
            self.weights_epoch.append(np.concatenate(weights))
            
            penultimate_layer_model = models.Sequential(self.model.layers[:-1])
            space_bent_output = penultimate_layer_model.predict(self.X_data)
            self.space_bent_output.append(space_bent_output)

    def save_data(self):
        if not os.path.isdir(f'training_digits_{try_nr}'):
            os.makedirs(f'training_digits_{try_nr}')
       # np.save(f'training_digits_{try_nr}\\predictions.npy', self.predictions_batch)
        np.save(f'training_digits_{try_nr}\\loss.npy', self.loss_batch)


        for i in [1]:#range(len(self.model.layers)):
        #i=2
            np.save(f'training_digits_{try_nr}\\X_bent_output_layers_{i}.npy', self.X_bent_output_layers[i])
            np.save(f'training_digits_{try_nr}\\space_bent_output_{i}.npy', self.space_bent_output_layers[i])
            np.save(f'training_digits_{try_nr}\\space_bent_output_labels_{i}.npy', self.space_bent_output_labels_layers[i])

            
      #  np.save(f'training_digits_{try_nr}\\layer_weights_{i}.npy', self.layer_weights[i])
            
    def on_batch_end(self, batch, logs=None):
        global batch_global
        batch_global += 1
        save_every = 100

        # if batch_global < 30:
        #     save_every = 1
        # elif batch_global < 100:
        #     save_every = 3
        # elif batch_global < 300:
        #     save_every = 10
        # elif batch_global < 600:
        #     save_every = 30
        # elif batch_global < 1000:
        #     save_every = 90
        # else:
        #     save_every = 270
            
        if batch_global % save_every == 0:
            
            for selected_layer in [1]:
                self.loss_batch.append(logs.get('loss'))
                
                
                
                
                
                penultimate_layer_model = models.Sequential(model.layers[:selected_layer+1])
                rest_model = models.Sequential(model.layers[selected_layer+1:])
                
                #space_bent_output = penultimate_layer_model.predict(self.X_full_space)
                space_bent_output = self.X_full_space
                space_bent_output_labels = rest_model.predict(space_bent_output).argmax(axis = 1)
                X_bent_output = penultimate_layer_model.predict(self.X_data)

                

                self.space_bent_output_layers[selected_layer].append(space_bent_output)
                self.X_bent_output_layers[selected_layer].append(X_bent_output)
                self.space_bent_output_labels_layers[selected_layer].append(space_bent_output_labels)

                
                
                # if batch_global > 9000:
                #     random_points = np.random.uniform(-15, 15 ,  (200000, X_bent_output.shape[1])) # do 500000
                #     random_points = sigmoid(random_points)
                #     pca = PCA(n_components=3)
                #     pca = pca.fit(random_points)
                #     random_points = pca.transform(random_points)

                #     for ii in range(3):
                #         X_bent_output = self.X_bent_output_layers[selected_layer][ii]
                        
                #         color_space_labels = "gray"

                        
                        
                #         x_train_gray = pca.transform(X_bent_output)
                        

                #         fig = go.Figure()
                #         fig.add_trace(go.Scatter3d( x=random_points[:,0],  y=random_points[:,1],  z=random_points[:,2], mode='markers', marker = dict(opacity=0.3, size = 1, color=color_space_labels,colorscale='rainbow'), name='random_points'))
                #         fig.add_trace(go.Scatter3d( x=x_train_gray[:,0],  y=x_train_gray[:,1],  z=x_train_gray[:,2], mode='markers', marker = dict(opacity=1, size = 0.5, color=y_train_mapped,colorscale='rainbow'), name='digit_train_set'))
                #         fig.update_layout(template = "plotly_dark")
                #         fig.show()
                #     exit()

                    

        # save_every = save_every*10
        
        if batch_global % (save_every*10) == 0:
            self.save_data()
            






# high_dim_points_normalized = []
# unit_vectors = np.eye(28*28)
# for k in np.linspace(0, 1, 150)**2 * 100:
#     high_dim_points_normalized.append(k*unit_vectors)
# high_dim_points_normalized = np.vstack(high_dim_points_normalized)
#high_dim_points_normalized[high_dim_points_normalized == 0] = -1e10



#high_dim_points_normalized = get_interpolated_from_origin_trainingset(x_test_gray)



#random_points = np.random.uniform(-10, 10 ,  (100000, x_train_gray.shape[1]))
random_points = np.random.uniform(-15, 15 ,  (300000, 7))

prediction_callback = PredictionCallback(x_train_gray, random_points, len(model.layers))

# Compile the model
model.compile(optimizer=Adam(learning_rate= 0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


if FIT_MODEL:
    if USE_CALLBACK:
        model.fit(x_train_gray, y_train_mapped, epochs=nr_epochs_train, batch_size=32, validation_split=0.1, callbacks=[prediction_callback])
    else:
        model.fit(x_train_gray, y_train_mapped, epochs=nr_epochs_train, batch_size=32, validation_split=0.1)


    model.save(f'training_digits_{try_nr}\\model_filename_{nr_digits}.h5')

#exit()

model = tf.keras.models.load_model(f'training_digits_{try_nr}\\model_filename_{nr_digits}.h5')

#exit()



#print(input_shape)
#exit()

def select_random_quadrants(k, n):
    quadrants = np.random.choice([-1, 1], size=(k, n))
    return quadrants


def filter_points_full(points, quadrants):
    # Determine the sign of each point (1 for positive, -1 for negative)
    point_signs = np.sign(points)
    
    # Create a mask where each row (point) matches any of the selected quadrants
    # Broadcasting the comparison of point signs with each quadrant
    quadrant_matches = np.all(point_signs[:, np.newaxis, :] == quadrants, axis=2)
    
    # Filter the points that match any of the quadrants
    matching_points = points[np.any(quadrant_matches, axis=1)]
    
    return matching_points

def filter_points(points, quadrants, match_threshold=0.99):
    """
    Filters points to only those that have at least 95% of their dimensions within the selected quadrants.
    """
    # Determine the sign of each point (1 for positive, -1 for negative)
    point_signs = np.sign(points)
    
    # Create a mask where each row (point) matches each quadrant in all dimensions
    quadrant_matches = (point_signs[:, np.newaxis, :] == quadrants)
    
    # Count how many dimensions match for each point with each quadrant
    match_count = np.sum(quadrant_matches, axis=2)
    
    # Calculate the minimum number of matching dimensions required (95% of n)
    min_matches_required = match_threshold * points.shape[1]
    
    # Filter the points based on the 95% matching condition with any of the selected quadrants
    matching_points = points[np.any(match_count >= min_matches_required, axis=1)]
    
    return matching_points



selected_layer = 2

input_shape = model.get_weights()[selected_layer].shape[0]

#random_points = np.random.uniform(-150, 150 ,  (100000, input_shape))


# selected_quadrants = select_random_quadrants(300, input_shape)

# random_points = np.empty((0, input_shape))

# while len(random_points) < 100_000:
#     r_points = np.random.uniform(-15, 15 ,  (1_000_000, input_shape))
#     r_points = filter_points(r_points, selected_quadrants)
#     random_points = np.vstack([random_points, r_points])
#     print(len(random_points))

# print(random_points)
# exit()


#random_points = relu(random_points)

model.summary()

model_1 = models.Sequential(model.layers[:selected_layer])
model_2 = models.Sequential(model.layers[selected_layer:])

 



random_points = np.random.uniform(-10, 10 ,  (500000, x_train_gray.shape[1]))

random_points = model_1.predict(random_points)
x_train_gray = model_1.predict(x_train_gray)
print(x_train_gray.shape)


color_space_labels = model_2.predict(random_points).argmax(axis = 1)
#color_space_labels = "gray"













pca = PCA(n_components=3)
pca = pca.fit(random_points)
x_train_gray = pca.transform(x_train_gray)
random_points = pca.transform(random_points)

# axes = fibonacci_sphere(7)
# x_train_gray = np.dot(axes.T, x_train_gray.T).T
# random_points = np.dot(axes.T, random_points.T).T



fig = go.Figure()
#fig.add_trace(go.Scatter( x=x_train_gray[:,0],  y=x_train_gray[:,1], mode='markers', marker = dict(opacity=1, size = 1.5, color=y_train_mapped,colorscale='rainbow'), name='digit_train_set'))
#fig.add_trace(go.Scatter( x=random_points[:,0],  y=random_points[:,1], mode='markers', marker = dict(opacity=0.5, size = 0.5, color=color_space_labels,colorscale='rainbow'), name='digit_train_set'))


fig.add_trace(go.Scatter3d( x=random_points[:,0],  y=random_points[:,1],  z=random_points[:,2], mode='markers', marker = dict(opacity=0.7, size = 1, color=color_space_labels,colorscale='rainbow'), name='random_points'))
fig.add_trace(go.Scatter3d( x=x_train_gray[:,0],  y=x_train_gray[:,1],  z=x_train_gray[:,2], mode='markers', marker = dict(opacity=1, size = 1, color=y_train_mapped,colorscale='rainbow'), name='digit_train_set'))


fig.update_layout(template = "plotly_dark")

# Show the interactive plot
fig.show()



# does the 2d space increased to 700 dimensions cover the entire range of at least one dimension? what about 3d?


