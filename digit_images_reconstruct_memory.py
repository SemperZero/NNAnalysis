import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA

nr_digits = 10
nr_inner_neurons = 8
activation = "sigmoid"
nr_epochs_train = 300
try_nr = f'digits_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'


model = tf.keras.models.load_model(f'trainings\\training_digits_{try_nr}\\model_filename_{nr_digits}.h5')

model.summary()

W1, b1 = model.layers[0].get_weights()  # First Dense layer (ReLU)
W2, b2 = model.layers[1].get_weights()  # Second Dense layer (Sigmoid)

nr_digits = W2.shape[1]  # Number of output classes
print(nr_digits)

def generate_valid_inputs(A, b, num_samples):
        """
        Generate random valid inputs that satisfy Ax > b.
        """
        valid_samples = []
        
        while len(valid_samples) < num_samples:
            x = np.random.uniform(-1, 1, size=(A.shape[1],))  # Adjust range if needed
            if np.all(A @ x > b):  # Check constraints
                valid_samples.append(x)
        
        return np.array(valid_samples)

def generate_high_certainty_samples(model, target_class, num_samples):
    collected_samples = []
    batch_size = 100000  # Generate and test 100k vectors at a time

    while len(collected_samples) < num_samples:
        random_inputs = np.random.uniform(-1, 1, size=(batch_size, model.input_shape[1]))
        predictions = model.predict(random_inputs)
        certainties = predictions.max(axis=1)
        predicted_classes = predictions.argmax(axis=1)
        
        high_certainty_indices = np.where((certainties > 0.9999) & (predicted_classes == target_class))[0]
         
        if len(high_certainty_indices) > 0:
            collected_samples.extend(random_inputs[high_certainty_indices])
        print(len(collected_samples))
            
        if len(collected_samples) > num_samples:
            collected_samples = collected_samples[:num_samples]

    return np.array(collected_samples)

def compute_A_b(target_class):
    A_list = []
    b_list = []

    for j in range(nr_digits):
        if j == target_class:
            continue  # Skip self-comparison

        delta_W2 = (W2[:, target_class] - W2[:, j]).reshape(1, -1)  # Shape (1, 16)
        A_j = delta_W2 @ W1.T  # Shape (1, input_dim)
        b_j = (b2[j] - b2[target_class]) - delta_W2 @ b1  # Corrected bias computation

        A_list.append(A_j)
        b_list.append(b_j)

    # Convert lists to numpy arrays
    A = np.vstack(A_list)  # Shape: (nr_digits-1, input_dim)
    b = np.hstack(b_list)  # Shape: (nr_digits-1,)
    return A, b


full_X_reconstruct = []
full_Y_reconstruct = []

for target_class in range(0, nr_digits):
    A, b = compute_A_b(target_class)

    print("A shape:", A.shape)
    print("b shape:", b.shape)

    

    # Example use (A, b need to be computed from the trained model)
    #valid_inputs = generate_valid_inputs(A, b, num_samples=10000)
    valid_inputs = generate_high_certainty_samples(model, target_class, 1_000)
    print(valid_inputs.shape)



    average_vector = np.mean(valid_inputs, axis=0)

    # Reconstruct and display images
    
    reconstructed_image = average_vector.reshape(28, 28)
    dpi = 100
    width_inch = 300 / dpi
    height_inch = 300 / dpi

    # Set the figure size in inches and DPI
    plt.figure(figsize=(width_inch, height_inch), dpi=dpi)

    plt.imshow(reconstructed_image, cmap="gray")
    plt.title(f"Reconstructed Image for Predicted Class {target_class}")
    plt.axis("off")  # Remove axes for a cleaner image
   # plt.show()
    plt.savefig(f'trainings\\training_digits_{try_nr}\\{nr_digits}_class_regenerated_{target_class}.png', bbox_inches='tight', pad_inches=0.1)

   
    np.save(f'trainings\\training_digits_{try_nr}\\{nr_digits}_class_regenerated_{target_class}.npy', valid_inputs)

   # plt.close()  # Close the plot to avoid displaying

   # del valid_inputs
