import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

nr_digits = 10
nr_inner_neurons = 8
activation = "sigmoid"
nr_epochs_train = 300
try_nr = f'digits_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'


model = tf.keras.models.load_model(f'trainings\\training_digits_{try_nr}\\model_filename_{nr_digits}.h5')

model.summary()

selected_layer = 0

M = model.get_weights()[selected_layer*2].T


model_1 = models.Sequential(model.layers[:selected_layer+1])
model_2 = models.Sequential(model.layers[selected_layer+1:])


fig = go.Figure()

random_points_base = sigmoid(np.random.uniform(-20, 20, (100000, M.shape[0])))

pca = PCA(n_components = 3)
pca = pca.fit(random_points_base)
random_points_base_pca = pca.transform(random_points_base)

fig.add_trace(go.Scatter3d(
    x=random_points_base_pca[:, 0], 
    y=random_points_base_pca[:, 1], 
    z=random_points_base_pca[:, 2],
    mode='markers',
    marker=dict(size=1, opacity=0.4, color = "gray" ),
    name = "base_space"
))


for target_class in range(0, 3):


    valid_inputs =  np.load(f'trainings\\training_digits_{try_nr}\\{nr_digits}_class_regenerated_{target_class}.npy')
    valid_points_proj_sig = model_1.predict(valid_inputs)
    


   
    valid_points_proj_sig_pca = pca.transform(valid_points_proj_sig)





    fig.add_trace(go.Scatter3d(
        x=valid_points_proj_sig_pca[:, 0], 
        y=valid_points_proj_sig_pca[:, 1], 
        z=valid_points_proj_sig_pca[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.7),
        name = f"class_{target_class}"
    ))

    
fig.update_layout(template= "plotly_dark")
fig.write_html(f'trainings\\training_digits_{try_nr}\\{nr_digits}_classes_regenerated.html')
#fig.show()
