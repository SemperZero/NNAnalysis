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
import tensorflow as tf




selected_layers = [1]


def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if min_val!=max_val:
        normalized = (arr - min_val) / (max_val - min_val)
        scaled = 2 * normalized - 1
    else:
        scaled = arr
    
    return scaled


        
def get_interpolated_matrix(current_layer, next_layer, nr_interp_frames):
    interp_matrix = np.zeros((nr_interp_frames, current_layer.shape[0], current_layer.shape[1]))
    for i in range(current_layer.shape[0]):
        for j in range(current_layer.shape[1]):
            point_trace = np.linspace(current_layer[i][j], next_layer[i][j], nr_interp_frames)
            interp_matrix[:, i, j] = point_trace
 
   # for k in range(interp_matrix.shape[0]):
    #    print(interp_matrix[k][333][2])
        
   # print(current_layer[333][2])
    
   # print(next_layer[333][2])
    #exit()

    return interp_matrix
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def plot_training_spacebent_all_smooth():
    
    global try_nr

    



    losses = np.load(f'training_digits_{try_nr}\\loss.npy')
    
   # predictions = np.load(f'training_digits_{try_nr}\\predictions.npy')

    nr_layers = 4
    
  
    X_bent_output_layers_0 = {}
    space_bent_output_all_0 = {}
    space_bent_output_all_0_labels = {}
    random_points = {}

    for layer_nr in selected_layers:
        X_bent_output_layers_0[layer_nr] = np.load(f'training_digits_{try_nr}\\X_bent_output_layers_{layer_nr}.npy')
        space_bent_output_all_0[layer_nr] = np.load(f'training_digits_{try_nr}\\space_bent_output_{layer_nr}.npy')
        space_bent_output_all_0_labels[layer_nr] = np.load(f'training_digits_{try_nr}\\space_bent_output_labels_{layer_nr}.npy')
      #  layer_weights.append(np.load(f'training_digits_{try_nr}\\layer_weights_{layer_nr}.npy'))
    
    

        random_points[layer_nr] = sigmoid(np.random.uniform(-15, 15 ,  (500000, X_bent_output_layers_0[layer_nr].shape[2])))

    print("total_saves", len(X_bent_output_layers_0[selected_layers[0]]))

    trained_pcas = {}
    for layer_nr in selected_layers:
       # space_bent_output = space_bent_output_all_0[layer_nr][-1]
        #if sigmoid_random_points.shape[1] > 3:
        pca = PCA(n_components=3)
        #pca = pca.fit(space_bent_output_all_0[layer_nr][-1])
        pca = pca.fit(random_points[layer_nr])
        trained_pcas[layer_nr] = pca

        
            
    X_bent_output_layers = {}
    space_bent_output_all = {}

    for layer_nr in selected_layers:
        X_bent_output_layers[layer_nr] = []
        space_bent_output_all[layer_nr] = []

        


        for i in range(len(X_bent_output_layers_0[layer_nr])):
            X_bent_output_layers[layer_nr].append(trained_pcas[layer_nr].transform(X_bent_output_layers_0[layer_nr][i]))

            space_bent_output_all_0[layer_nr][i] = sigmoid(space_bent_output_all_0[layer_nr][i])
            space_bent_output_all[layer_nr].append(trained_pcas[layer_nr].transform(space_bent_output_all_0[layer_nr][i]))
           # print(X_bent_output_layers_0[layer_nr][i].shape)
           # print(X_bent_output_layers[layer_nr][i].shape)
           # exit()
            



    
    

    degrees_per_sec = 10
    degrees_per_iter = degrees_per_sec / 60
   # print("total descents", len(X_bent_output_layers[0]), len(space_bent_output_all[0]),len(layer_weights), len(predictions), X_bent_output_layers[0][0].shape, space_bent_output_all[0][0].shape)
   # exit()
   
    frame_nr = 0
    skip_smooth_interval = 1
    
    animation_folder_path = f"animations\\training_learning_smooth_{try_nr}_{skip_smooth_interval}"# nr_interp_frames, skip_step
    if not os.path.isdir(animation_folder_path):
        os.makedirs(animation_folder_path)

    for layer_nr in selected_layers:
        if not os.path.isdir(f"{animation_folder_path}\\layer_nr{layer_nr}"):
            os.makedirs(f"{animation_folder_path}\\layer_nr{layer_nr}")
        
        
    for ii in range(0, len(losses), skip_smooth_interval):
        print(ii)

        
        

    
        
        #interp_xtrain = {}

        

        
        fig_layer_all = {}
        for layer_nr in selected_layers:
            angle_ = frame_nr * degrees_per_iter * np.pi/180
            
            
          #  x_training = interp_xtrain[layer_nr][j]
            
            #print( "val", x_training[333][2])
            
            # for k in range(0, x_training.shape[1]):#3 dimensions
                #   x_training[:,k] = min_max_normalize(x_training[:,k])
            
        
            # print(space_bent_output_all[layer_nr].shape)
            
            
            
            

            fig_layer = go.Figure()
        #    prediction = np.argmax(predictions[ii], axis = 1)
            #   print("output lengs", len(prediction), len(y_train_mapped))
    
            
            fig_layer.add_trace(go.Scatter3d(name = "sigmoid space", x=space_bent_output_all[layer_nr][ii][:,0], y=space_bent_output_all[layer_nr][ii][:,1],z = space_bent_output_all[layer_nr][ii][:,2], mode = "markers", marker = dict(opacity=0.7, size = 1, color=space_bent_output_all_0_labels[layer_nr][ii],colorscale='rainbow')))
          #  fig_layer.add_trace(go.Scatter3d(name = "training data", x=X_bent_output_layers[layer_nr][ii][:,0], y=X_bent_output_layers[layer_nr][ii][:,1],z = X_bent_output_layers[layer_nr][ii][:,2], mode = "markers", marker = dict(opacity=1, size = 1, color=y_train_mapped,colorscale='rainbow')))

            angle = angle_
            
                    
            
            
    
            angle = angle - 90*np.pi/180
            vertical_angle = 45*np.pi/180
            r = 1.5
            fig_layer.update_layout(scene_camera=dict(eye=dict(x=r*np.cos(vertical_angle)*np.cos(angle), y=r*np.cos(vertical_angle)*np.sin(angle), z=r*np.sin(vertical_angle))))

            #fig_layer.update_layout(scene_camera=dict(eye=dict(x=1.6*np.cos(angle)), y=1.6*np.sin(angle)), z=0.3)))
            fig_layer.update_layout( template = "plotly_dark", width = 1600, height = 1500)
            fig_layer.update_coloraxes(showscale=False)

        # fig_layer.show()
        # exit()
            
            fig_layer_all[layer_nr] = fig_layer
           # fig_layer.show()
           # exit()
        
    # if i > 300:
        # fig_weights.show()
        # exit()

        #fig_weights.write_image(f"training{try_nr}\\animations\\approx_{i:05}.png")
        

        
        # fig_layer_all[0].show()
       # exit()
        for layer_nr in selected_layers:
            fig_layer_all[layer_nr].write_image(f"{animation_folder_path}\\layer_nr{layer_nr}\\image_{frame_nr:05}.png")

        
        print("framenr", frame_nr)
        frame_nr+=1

try_nr = 96


(x_train, y_train_mapped), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plot_training_spacebent_all_smooth()



# in a relu-sig arch do the clusters in the sigmoid layer get chosen from the very start dradient step, mapping then to the softmax in the next layer? and never change after???
