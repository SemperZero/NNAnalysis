import numpy as np
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from itertools import combinations
from collections import defaultdict, Counter
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from scipy.spatial import distance
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from scipy.linalg import null_space
from scipy.optimize import lsq_linear, minimize, LinearConstraint
import os
from scipy.optimize import linprog
from scipy.linalg import svd

import numpy as np
from scipy.optimize import linprog

def generate_x_samples(M, B, Y_ranges, num_samples=1000):
    n = M.shape[0]
    #print(M.shape)
    #exit()
    burn_in_steps = 100 #* n # Number of steps before collecting samples (for mixing)
    thinning_steps = 10 #* n # Number of steps between collected samples

    # --- Tolerance for strict inequalities ---
    # For Y > min_y and Y < max_y, we use Y >= min_y + tolerance and Y <= max_y - tolerance
    TOL_Y = 1e-6 # A small positive number
    # For X > 0, we use X >= X_MIN_BOUND (can be 0 or small positive)
    X_MIN_BOUND = 1e-6 # To ensure x_i are strictly positive, or just >=0 if 0 is allowed.
                    # Problem statement says x > 0, so 1e-6 is appropriate.


    # --- 1. Formulate the Problem as a System of Linear Inequalities for X ---

    # The constraints are:
    # 1. X @ M + B > Y_min_j  => -(X @ M) <= B - (Y_min_j + TOL_Y)
    # 2. X @ M + B < Y_max_j  => X @ M <= Y_max_j - B - TOL_Y
    # 3. X > 0              => -X <= -X_MIN_BOUND

    # Let X be a 1D array of shape (n,)
    # Convert to standard LP inequality form: A_ub @ X <= b_ub

    # Constraint 1: X @ M + B > Y_min  => -(X @ M) <= B - (Y_min + TOL_Y)
    A_ub1 = -M.T # Transpose M to align with X as a 1D vector (n,)
    b_ub1 = B - (Y_ranges[:, 0] + TOL_Y)

    # Constraint 2: X @ M + B < Y_max  => X @ M <= Y_max - B - TOL_Y
    A_ub2 = M.T
    b_ub2 = Y_ranges[:, 1] - B - TOL_Y

    # Constraint 3: X > 0 => -X <= -X_MIN_BOUND
    A_ub3 = -np.eye(n)
    b_ub3 = -np.full(n, X_MIN_BOUND) # All elements must be >= X_MIN_BOUND

    # Combine all inequality constraints
    A_ub = np.vstack((A_ub1, A_ub2, A_ub3))
    b_ub = np.concatenate((b_ub1, b_ub2, b_ub3))

    # --- 2. Find an Initial Feasible Point using Linear Programming ---
    c = np.zeros(n) # Dummy objective function coefficients for feasibility

    # No explicit bounds needed as X >= X_MIN_BOUND is handled by A_ub3
    bounds = [(None, None)] * n

    print("Attempting to find an initial feasible point...")
    # Set maximum iterations and tolerance for linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', options={'maxiter': 10000, 'tol': 1e-7})

    if res.success:
        initial_x = res.x
        print(f"Initial feasible point found (first 5 elements):\n{initial_x[:5]}...")
        # Verify constraints for the initial point
        calculated_Y = initial_x @ M + B
        print(f"Calculated Y for initial point:\n{calculated_Y}")
        print(f"Y ranges:\n{Y_ranges.T}")
        print(f"All x_i > {X_MIN_BOUND}: {np.all(initial_x >= X_MIN_BOUND - 1e-9)}") # Allow slight tolerance for check
        print(f"Y strictly within ranges: {np.all(calculated_Y > Y_ranges[:,0] + 1e-9) and np.all(calculated_Y < Y_ranges[:,1] - 1e-9)}")
        # Further check to be sure:
        print(f"Linprog slack (should be negative for inequalities): {A_ub @ initial_x - b_ub}")


    else:
        print("Could not find an initial feasible point.")
        print(f"Linprog status: {res.message}")
        print("This might mean the feasible region is empty given the strict inequality constraints.")
        print("Consider loosening the Y_ranges or X_MIN_BOUND if this is unexpected.")
        exit("Exiting because no feasible region exists or could be found.")


    # --- 3. Implement the Hit-and-Run Sampler ---

    # Helper function to check if a point is within the feasible region (with tolerance)
    def is_feasible(x_point, A_matrix, b_vector, tol=1e-9):
        # Check A_ub @ x <= b_ub. Add tolerance for numerical stability
        return np.all(A_matrix @ x_point <= b_vector + tol)

    # Function to find intersection points along a direction
    def find_step_bounds(current_x, direction, A_matrix, b_vector, tol=1e-12):
        # We want to find alpha_min, alpha_max such that:
        # A @ (current_x + alpha * direction) <= b
        # A @ current_x + alpha * (A @ direction) <= b
        # alpha * (A @ direction) <= b - (A @ current_x)

        denominators = A_matrix @ direction
        numerators = b_vector - (A_matrix @ current_x)

        alpha_candidates_upper = []
        alpha_candidates_lower = []

        # Iterate through each inequality constraint
        for i in range(A_matrix.shape[0]):
            denom = denominators[i]
            num = numerators[i]

            if denom > tol:  # direction points "into" the half-space, imposes upper bound
                alpha_candidates_upper.append(num / denom)
            elif denom < -tol: # direction points "out of" the half-space, imposes lower bound
                alpha_candidates_lower.append(num / denom)
            else: # denom is close to zero (direction is parallel to the hyperplane)
                # Check if current_x satisfies this constraint with current alpha=0
                if num < -tol: # current_x is already outside this parallel constraint
                    return -np.inf, np.inf # Indicate infeasible direction or current point issue

        alpha_max = np.inf
        if alpha_candidates_upper:
            alpha_max = np.min(alpha_candidates_upper)

        alpha_min = -np.inf
        if alpha_candidates_lower:
            alpha_min = np.max(alpha_candidates_lower)

        return alpha_min, alpha_max

    # Hit-and-Run sampling loop
    samples_X = []
    current_x = initial_x.copy()

    print(f"\nStarting Hit-and-Run sampling ({num_samples} samples)...")
    print(f"Burn-in steps: {burn_in_steps}, Thinning steps: {thinning_steps}")

    successful_steps = 0
    attempts = 0 # To prevent infinite loops if the region is very thin or tricky

    while len(samples_X) < num_samples:
        attempts += 1
        # Safety break: If too many attempts fail to find a valid move, something might be wrong
        if attempts > (num_samples * thinning_steps + burn_in_steps) * 5:
            print("Warning: Too many attempts without collecting enough samples. Exiting sampling loop.")
            break

        # 1. Choose a random direction (unit vector)
        direction = np.random.randn(n)
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-12: # Avoid zero direction
            continue
        direction = direction / direction_norm

        # 2. Find the bounds for alpha
        alpha_min, alpha_max = find_step_bounds(current_x, direction, A_ub, b_ub)

        # 3. Check for valid interval
        if alpha_min >= alpha_max - 1e-9: # Add a small tolerance for comparison
            # This can happen if the current point is very close to the boundary,
            # and the random direction points slightly outwards, or if the region is degenerate.
            # We need to ensure alpha_min < alpha_max for sampling.
            # print(f"  Debug: Alpha bounds not valid ({alpha_min:.2e}, {alpha_max:.2e}). Retrying direction.")
            continue # Try a new random direction

        # 4. Sample a new alpha uniformly within the bounds
        alpha = np.random.uniform(alpha_min, alpha_max)

        # 5. Move to the new point
        next_x = current_x + alpha * direction

        # A more robust check for the newly proposed point being feasible (optional but good for debugging)
        if not is_feasible(next_x, A_ub, b_ub):
            # This should ideally not happen if find_step_bounds is perfect and current_x is perfectly feasible,
            # but floating point arithmetic can cause tiny deviations.
            # print(f"  Debug: Proposed next_x is not feasible. alpha_min={alpha_min:.2e}, alpha_max={alpha_max:.2e}, alpha_chosen={alpha:.2e}. Retrying.")
            continue # Skip this step and try again

        current_x = next_x # Update current_x only if the new point is feasible

        successful_steps += 1

        # 6. Store the sample after burn-in and thinning
        if successful_steps > burn_in_steps and (successful_steps - burn_in_steps) % thinning_steps == 0:
            samples_X.append(current_x.copy())
            if len(samples_X) % (num_samples // 10 or 1) == 0:
                print(f"Collected {len(samples_X)} samples...")

    print(f"Finished sampling. Collected {len(samples_X)} samples out of {num_samples} desired.")
    print(f"Total attempts: {attempts}, Successful moves: {successful_steps}")

    return samples_X


def nullspace(A, rtol=1e-5):
    """Compute orthonormal basis for nullspace of A."""
    u, s, vh = svd(A)
    rank = (s > rtol * s[0]).sum()
    ns = vh[rank:].conj().T
    return ns

def find_particular_solution(A, d, lower, upper):
    """Find one feasible particular solution for A x = d with bounds."""
    n_vars = A.shape[1]
    c = np.zeros(n_vars)
    bounds = [(l, u) for l, u in zip(lower, upper)]
    res = linprog(c, A_eq=A, b_eq=d, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        return None

def bounded_nullspace_walk(x0, ns_basis, lower, upper, n_samples, step_size=1.0):
    """
    Generate samples by walking along nullspace directions inside bounds.

    Parameters:
    - x0: starting feasible point
    - ns_basis: nullspace basis vectors (n_vars x nullity)
    - lower, upper: variable bounds
    - n_samples: number of samples to generate
    - step_size: max step size along direction (scaled)

    Returns:
    - samples: list of feasible points
    """
    n_vars, nullity = ns_basis.shape
    samples = [x0.copy()]
    current = x0.copy()

    for _ in range(n_samples - 1):
        # Pick random direction in nullspace unit ball
        direction_coeffs = np.random.randn(nullity)
        direction_coeffs /= np.linalg.norm(direction_coeffs)
        direction = ns_basis @ direction_coeffs

        # Compute max step sizes in positive and negative directions to stay in bounds
        with np.errstate(divide='ignore', invalid='ignore'):
            pos_steps = np.where(direction > 1e-12,
                                 (upper - current) / direction,
                                 np.inf)
            neg_steps = np.where(direction < -1e-12,
                                 (lower - current) / direction,
                                 np.inf)

        max_pos_step = np.min(pos_steps)
        max_neg_step = np.min(neg_steps)

        # Clamp max steps to step_size for controlled walk
        max_pos_step = min(max_pos_step, step_size)
        max_neg_step = max(max_neg_step, -step_size)

        # Sample step size uniformly in feasible range
        step = np.random.uniform(max_neg_step, max_pos_step)

        # Move and append sample
        current = current + step * direction
        samples.append(current.copy())

    return samples


def find_particular_solution_unbounded(A, d):
    """Find a particular solution to A x = d (unconstrained)."""
    x_particular, *_ = np.linalg.lstsq(A, d, rcond=None)
    return x_particular

def generate_solutions_unbounded(A, d_min, d_max, k, num_samples=10, scale=1.0):
    """
    Generate multiple solutions to A x = d with no bounds on x.
    Each solution is: x = x₀ + N z, where z is random and N spans null(A).
    """
    for _ in range(k):
        d = np.random.uniform(np.minimum(d_min, d_max), np.maximum(d_min, d_max))

        x0 = find_particular_solution_unbounded(A, d)
        N = nullspace(A)
        n_null = N.shape[1]

        solutions = []
        for _ in range(num_samples):
            z = np.random.randn(n_null) * scale
            x = x0 + N @ z
            solutions.append(x)
    return solutions

def generate_solutions_with_d_ranges(A, d_min, d_max, lower, upper, k=5, samples_per_d=5, step_size=1.0):
    """
    Generate solutions for sampled d in [d_min, d_max] using bounded nullspace walk.

    Returns list of (d_sample, [solutions]) tuples.
    """

    n_eqs = A.shape[0]
    n_vars = A.shape[1]

    results = []

    for sample_i in range(k):
        d_sample = np.random.uniform(d_min, d_max)
      #  print(d_sample)

        x0 = find_particular_solution(A, d_sample, lower, upper)
        results.append(x0)
        # if x0 is None:
        #    # print(f"No feasible solution for sampled d (index {sample_i}). Skipping.")
        #     continue

        # ns_basis = nullspace(A)
        # if ns_basis.size == 0:
        #     # Unique solution only
        #     results.append((d_sample, [x0]))
        #     continue

        # sols = bounded_nullspace_walk(x0, ns_basis, lower, upper, samples_per_d, step_size=step_size)
        # results.extend( sols)

    return results



def analyze_nd_cube(points, tol=1e-1):
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

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)


def shuffle_dataset(x,y):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    return x[indices], y[indices]


def sample_equal_digits(x, y, digits):
    filtered_x, filtered_y = [], []
    for digit in digits:
        indices = np.where(y == digit)[0]
        filtered_x.append(x[indices])
        filtered_y.append(y[indices])
    return np.concatenate(filtered_x), np.concatenate(filtered_y)




x_train_flattened = x_train.reshape(-1, 28 * 28) / 255.0
x_test_flattened = x_test.reshape(-1, 28 * 28) / 255.0

x_train, y_train = x_train_flattened, y_train
x_test, y_test = x_test_flattened, y_test


def get_model_no_final_activation(model, nr_final_layer = -1):
    layers_without_last = model.layers[:nr_final_layer]
    

    # Create a new model without the final activation
    input_layer = Input(shape=model.input_shape[1:], name='new_input')  # Unique name for input layer
    x = input_layer
    for i, layer in enumerate(layers_without_last):
        x = layer(x)  # Reuse the original layers

    # Add the final layer without activation
    output_no_activation = Dense(model.layers[nr_final_layer].units, name='new_output')(x)  # Unique name for output layer

    # Create the new model
    model_no_softmax = Model(inputs=input_layer, outputs=output_no_activation)

    # Copy the weights from the original model to the new model
    for i, layer in enumerate(model_no_softmax.layers[1:]):  # Skip the Input layer (index 0)
        layer.set_weights(model.layers[i].get_weights())

    
    return model_no_softmax


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def get_vector_line(v1, color = "white", width = 3):
    return go.Scatter3d(x=[0, v1[0]], y=[0, v1[1]], z=[0,v1[2]], mode='lines', line=dict(color=color, width=width))
def get_vector_line2d(v1, color = "white", width = 3):
    return go.Scatter(x=[0, v1[0]], y=[0, v1[1]], mode='lines', line=dict(color=color, width=width))


def get_quadrant_labels(X_relu):
    cluster_mask_to_label = {}
    current_label = 0
    quadrant_labels = []
    for point in X_relu:
        cluster_id = 0
        
        for i, value in enumerate(point):
            if value > 0:
                cluster_id |= (1 << i)
        #cluster_counts[cluster_id] += 1

        if cluster_id not in cluster_mask_to_label:
            cluster_mask_to_label[cluster_id] = current_label
            current_label+=1

        quadrant_labels.append(cluster_mask_to_label[cluster_id])
    return quadrant_labels

def get_sigmoid_color_labels(arr, tol=1e-1):
    arr = (arr-0.5)*2
    norms = []
    for row in arr:
        mask = ~(np.isclose(row, -1, atol=tol) | np.isclose(row, 1, atol=tol))
        filtered = row[mask]
       # print(filtered)
        if len(filtered):
         norms.append(np.min(np.abs(filtered)))
        else:
            norms.append(0)
    return np.array(norms)

#np.random.seed(42)


nr_digits =10
activation = "sigmoid"
nr_epochs_train = 300

if nr_digits == 2:
    selected_digits = [5, 8]
    x_train, y_train = sample_equal_digits(x_train, y_train, selected_digits)
    x_test, y_test = sample_equal_digits(x_test, y_test, selected_digits)

    y_train = np.where(y_train == 5, 0, 1)  # 5->0, 8->1
    y_test = np.where(y_test == 5, 0, 1)   

for nr_inner_neurons in range(8, 30):
#for nr_inner_neurons in [10]:

    try_nr = f'digits_{nr_digits}_nodes_{nr_inner_neurons}_{activation}_epoc_{nr_epochs_train}'

    model = tf.keras.models.load_model(f'trainings\\training_digits_{try_nr}\\model_filename_{nr_digits}.h5')

    selected_layer = 0


    input_shape = model.get_weights()[selected_layer*2].shape[0]
    lowerdim = model.get_weights()[(selected_layer+1)*2].shape[0]



    model.summary()

    model_1 = models.Sequential(model.layers[:selected_layer+1])
    model_2 = models.Sequential(model.layers[selected_layer+1:])
    model_no_act = get_model_no_final_activation(model)





    M = model.get_weights()[selected_layer*2].T
    B = model.get_weights()[selected_layer*2+1].T


    #random_points = np.random.uniform(-1, 1, (100000, input_shape))  # 1000 points in 10D space
    random_points = x_train

    random_points_proj = np.dot(M, random_points.T).T + B.T
    X_input_proj = np.dot(M, x_train.T).T + B.T




    #quadrant_labels_higherdim = get_quadrant_labels(random_points)

    quadrant_labels_lowerdim = get_quadrant_labels(random_points_proj)

    sigmoid_color_labels = get_sigmoid_color_labels(random_points_proj)



    pca = PCA(n_components=3)
    pca = pca.fit(x_train)

    #random_points = relu(random_points)

    labels_cube, label_to_description, label_to_description_vertex, label_to_description_edge, vertex_counts, edge_counts, vertex_key_to_label, edge_key_to_label = analyze_nd_cube(sigmoid(random_points_proj))

  
    #print(labels_cube)
    #exit()


    random_points_pca = pca.transform(random_points)
    x_train_pca = pca.transform(x_train)


    colors = ['blue', 'red', 'green']

    pred_probs = model(random_points)
    predicted_class = tf.argmax(pred_probs, axis=1).numpy()

    # random_points_proj_in_higherdim_pca = random_points_proj_in_higherdim


    #exit()


    ############## ANALYSIS ON CLUSTERS #####################


    random_points_proj_sig = sigmoid(random_points_proj)
    X_input_proj_sig = sigmoid(X_input_proj)
    random_points_proj_sig_true = model_1.predict(random_points)

    random_points_base = sigmoid(np.random.uniform(-20, 20, (100000, M.shape[0])))

    labels_cube_train, label_to_description_train, label_to_description_vertex_train, label_to_description_edge_train, vertex_counts_train, edge_counts_train, vertex_key_to_label_train, edge_key_to_label_train = analyze_nd_cube(X_input_proj_sig)


    print(labels_cube_train)
    import plotly.express as px

    fig = px.histogram(
        x=labels_cube_train,
        nbins=len(np.unique(labels_cube_train)),  # One bin per unique label
        title='Distribution of Labels',
        labels={'x': 'Label Value', 'y': 'Count'},
        color_discrete_sequence=['blue']  # Optional: set bar color
    )

    # Customize layout
    fig.update_layout(
        bargap=0.1,  # Gap between bars
        xaxis=dict(tickmode='linear')  # Ensure all labels are shown on x-axis
    )





    #fig.show()

    SELECTED_INDEX_CLUSTER = list(label_to_description_vertex_train.keys()) #manually after looking at graph

    #index_corner = np.where(labels_cube_train == SELECTED_INDEX_CLUSTER)[0]#[0]
    index_corner = np.where(np.isin(labels_cube_train, SELECTED_INDEX_CLUSTER))[0]

    print(index_corner)

    def custom_round(arr, epsilon=1e-2, decimals=2):
        arr = np.asarray(arr)
        # Force values close to 0 or 1 to become 0 or 1
        rounded = np.where(np.abs(arr - 0) <= epsilon, 0, arr)
        rounded = np.where(np.abs(rounded - 1) <= epsilon, 1, rounded)
        # For all others, round to the specified number of decimal places
        final = np.round(rounded, decimals=decimals)
        return final


    #rounded_arr = custom_round(selected_x, epsilon= 0.1, decimals=2)
    #print(rounded_arr)
    #print(rounded_arr.shape)










    #exit()

    import matplotlib.pyplot as plt

    # SAVE INPUT DIGITS

    # for selected_cluster in SELECTED_INDEX_CLUSTER:

    #     index_corner = np.where(labels_cube_train == selected_cluster)[0]


    #     selected_x = x_train[index_corner]
    #     selected_y = y_train[index_corner]

    #     average_vector = np.mean(selected_x, axis=0)

    #     # Reconstruct and display images

    #     reconstructed_image = average_vector.reshape(28, 28)
    #     dpi = 100
    #     width_inch = 1000 / dpi
    #     height_inch = 1000 / dpi

    #     # plt.figure(figsize=(width_inch, height_inch), dpi=dpi)

    #     # plt.imshow(reconstructed_image, cmap="gray")
    #     # plt.title(f"Reconstructed Image for Predicted Class ")
    #     # plt.axis("off")  # Remove axes for a cleaner image
    #     # plt.show()

    #     plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
    #     plt.imshow(reconstructed_image, cmap="gray")
    #     plt.title(f"Reconstructed Image for Predicted Class {selected_y[0]}")
    #     plt.axis("off")  # Remove axes for a cleaner image

    #     save_folder = f'trainings\\training_digits_{try_nr}\\images_cluster_trainset\\class_{selected_y[0]}'
    #     if not os.path.isdir(save_folder):
    #         os.makedirs(save_folder)
    #     plt.savefig(f'{save_folder}\\nr_points_{len(selected_x)}_vertex_{label_to_description_vertex_train[selected_cluster]}.png', bbox_inches='tight', pad_inches=0)
    #     plt.close() 




    found_points = {}
    found_labels = {}

    def get_points_in_700d_projected(A, b, pinv, num_samples):

        # Step 1: Compute particular solution
        x_particular = pinv @ b  # shape (700,)

        # Step 2: Compute null space
        N = null_space(A)  # shape (700, k), where k ≈ 700 - rank(A)

        # Step 3: Sample random coefficients
        # For example, sample 10 random solutions

        rand_coeffs = np.random.randn(N.shape[1], num_samples)# * 10 # shape (k, 10)

        # Step 4: Generate full solutions
        X_samples = x_particular[:, np.newaxis] + N @ rand_coeffs 
        return X_samples


    def get_points_in_700d_projected1(P, y0, pinv, num_samples):
        def nullspace(A, atol=1e-13, rtol=0):
            u, s, vh = np.linalg.svd(A)
            tol = max(atol, rtol * s[0])
            rank = (s > tol).sum()
            return vh[rank:].T



        # Step 2: Find base feasible point x0
        result = lsq_linear(P, y0, bounds=(0, 1))
        if not result.success:
            raise ValueError("No feasible base solution found.")

        x0 = result.x
        N = nullspace(P)  # shape (100, 98)
        b = -x0

        dim = N.shape[1]

        # Step 3: Sampling feasible points by solving constrained QP in nullspace coords
        def sample_feasible_points(n_points):
            points = []
            lincon = LinearConstraint(N, b, 1)
            for i in range(n_points * 3):  # oversample attempts
                c_target = np.random.randn(dim)

                def obj(c):
                    return np.sum((c - c_target) ** 2)

                res = minimize(obj, np.zeros(dim), constraints=[lincon], method='trust-constr',
                            options={'maxiter': 50, 'verbose': 0})

                if res.success:
                    c = res.x
                    x = x0 + N @ c
                    if np.all(x >= 0):
                        points.append(x)
                        if len(points) >= n_points:
                            break

            
                print(len(points))
            if len(points) < n_points:
                print(f"Warning: only found {len(points)} feasible points out of requested {n_points}")
            return np.array(points)


        return sample_feasible_points(num_samples)
  


    for selected_cluster in SELECTED_INDEX_CLUSTER[:100]:

        print(selected_cluster)
        index_corner = np.where(labels_cube_train == selected_cluster)[0]


        selected_x = x_train[index_corner]
        selected_y = y_train[index_corner]
        print("Y", selected_y)

        


        selected_x_projected = X_input_proj[index_corner]


        # fig = go.Figure()

        # fig.add_trace(go.Scatter(
        #         x=list(range(len(selected_x_projected[0]))),
        #         y=[5]* len(selected_x_projected[0]),
        #         mode='lines',
        #         line = dict(color= "red")
        #     ))
    
        # fig.add_trace(go.Scatter(
        #         x=list(range(len(selected_x_projected[0]))),
        #         y=[-5]* len(selected_x_projected[0]),
        #         mode='lines',
        #         line = dict(color= "red")
        #     ))
        

        # for vector in selected_x_projected:

        #     fig.add_trace(go.Scatter(
        #         x=list(range(len(vector))),
        #         y=vector,
        #         mode='markers+lines',
        #         marker=dict(size=10),
        #         line=dict(width=0.75),
        #         opacity = 0.5
        #     ))

        # # Update layout to use Plotly dark theme
        # fig.update_layout(
        #     template='plotly_dark', 
        #     xaxis_title='Index',
        #     yaxis_title='Value',
        #  #   showlegend=False
        # )

        # folder_plots = f"plots\\visualize_inner_number_distrib_cluster\\nr_inner_neurons_{nr_inner_neurons}"
        
        # if not os.path.isdir(folder_plots):
        #     os.makedirs(folder_plots)

        # fig.write_html(f"{folder_plots}\\selected_inner_cluster_{selected_cluster}.html")

        minimums = np.min(selected_x_projected, axis = 0)
        maximums = np.max(selected_x_projected, axis = 0)
      #  print(minimums)
      #  print(maximums)

    #    continue


        Y_ranges = np.column_stack([minimums, maximums])
       # print(Y_ranges)
       # exit()
       # print(M.shape)
       # exit()

        x_lower = np.full(M.shape[1], 0)
        x_upper = np.full(M.shape[1], 1.0)
        k = 1000
        samples_per_d = 100
       # full_samples = generate_solutions_with_d_ranges(M, minimums, maximums, x_lower, x_upper, k=k, samples_per_d=samples_per_d, step_size=0.5)
        full_samples = generate_solutions_unbounded(M, minimums, maximums, k, samples_per_d)

        generate_solutions_unbounded
      #  full_samples = generate_x_samples(M, B.T, Y_ranges, num_samples=1000)
        print(full_samples[0].shape)
      #  exit()
     #   exit()
        
        full_samples = np.vstack(full_samples)

        print(full_samples.shape)

        average_vector = np.mean(full_samples, axis=0)
        print(average_vector.shape)

        x_min = np.min(average_vector)
        x_max = np.max(average_vector)
        x_normalized = 255 * (average_vector - x_min) / (x_max - x_min)

        # Optional: Convert to uint8 for image compatibility
        x_normalized_uint8 = x_normalized.astype(np.uint8)

        print(x_normalized_uint8)
        print(x_normalized_uint8.shape)
        #exit()


        reconstructed_image = x_normalized_uint8.reshape( 28, 28, -1) 
        dpi = 100
        width_inch = 1000 / dpi
        height_inch = 1000 / dpi

    #  Set the figure size in inches and DPI
        plt.figure(figsize=(width_inch, height_inch), dpi=dpi)

        plt.imshow(reconstructed_image, cmap="gray")
     #   plt.title(f"Reconstructed Image for Predicted Class {found_labels[selected_cluster]} ")
        plt.axis("off")  # Remove axes for a cleaner image

        save_folder = f'trainings\\training_digits_{try_nr}\\images_cluster_trainset\\class_{selected_y[0]}_reconstructed'
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        plt.savefig(f'{save_folder}\\nr_points_{len(selected_x)}_vertex_{label_to_description_vertex_train[selected_cluster]}.png', bbox_inches='tight', pad_inches=0)


    continue
    index_corner = np.where(np.isin(labels_cube_train, SELECTED_INDEX_CLUSTER))[0]
    selected_x = X_input_proj_sig[index_corner]
    selected_y = y_train[index_corner]

    pca = PCA(n_components = 3)
    pca = pca.fit(selected_x)
    selected_x_pca = pca.transform(selected_x)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=selected_x_pca[:, 0], 
        y=selected_x_pca[:, 1], 
        z=selected_x_pca[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.7, color=[colors[i] for i in selected_y] ),
        name = "selected_clusters with sigmoid"
    ))

    for selected_cluster in SELECTED_INDEX_CLUSTER:
        
        found_points_pca = pca.transform(sigmoid(found_points[selected_cluster].T))

        fig.add_trace(go.Scatter3d(
            x=found_points_pca[:, 0], 
            y=found_points_pca[:, 1], 
            z=found_points_pca[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.7, color=colors[found_labels[selected_cluster]]),
            name = f"generated points in {selected_cluster} cluster"
            ))


    fig.update_layout(template= "plotly_dark")

    fig.show()



    selected_x = X_input_proj[index_corner]
    pca = PCA(n_components = 3)
    pca = pca.fit(selected_x)
    selected_x_pca = pca.transform(selected_x)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=selected_x_pca[:, 0], 
        y=selected_x_pca[:, 1], 
        z=selected_x_pca[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.7, color=[colors[i] for i in selected_y]  ),
        name = "selected_clusters no sigmoid",
    ))

    for selected_cluster in SELECTED_INDEX_CLUSTER:
        
        found_points_pca = pca.transform(found_points[selected_cluster].T)

        fig.add_trace(go.Scatter3d(
            x=found_points_pca[:, 0], 
            y=found_points_pca[:, 1], 
            z=found_points_pca[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.7, color=colors[found_labels[selected_cluster]] ),
            name = f"generated points in {selected_cluster} cluster"
            ))


    fig.update_layout(template= "plotly_dark")

    fig.show()








    ############## ANALYSIS ON CLUSTERS #####################






    print(X_input_proj_sig)

    #exit()



    pca = PCA(n_components = 3)
    pca = pca.fit(random_points_base)
    random_points_proj_sig_pca = pca.transform(random_points_proj_sig)
    random_points_base_pca = pca.transform(random_points_base)
    X_input_proj_sig_pca = pca.transform(X_input_proj_sig)
    random_points_proj_sig_true_pca = pca.transform(random_points_proj_sig_true)

    # random_points_proj_sig_pca = random_points_proj_sig
    # random_points_base_pca = random_points_base
    # X_input_proj_sig_pca = X_input_proj_sig
    # random_points_proj_sig_true_pca = random_points_proj_sig_true


    # fig.add_trace(go.Scatter3d(
    #     x=random_points_proj[:, 0], 
    #     y=random_points_proj[:, 1], 
    #     z=random_points_proj[:, 2],
    #     mode='markers',
    #     marker=dict(size=1, opacity=0.4, color = quadrant_labels_lowerdim ),
    #     name = "base_space"
    # ))
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=random_points_base_pca[:, 0], 
        y=random_points_base_pca[:, 1], 
        z=random_points_base_pca[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.4, color = "gray" ),
        name = "base_space"
    ))



    fig.add_trace(go.Scatter3d(
        x=random_points_proj_sig_pca[:, 0], 
        y=random_points_proj_sig_pca[:, 1], 
        z=random_points_proj_sig_pca[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.7, color = quadrant_labels_lowerdim, colorscale = "rainbow"  ),
        name = "quadrant_labels_lowerdim"
    ))

    # fig.add_trace(go.Scatter3d(
    #     x=random_points_proj_sig_true_pca[:, 0], 
    #     y=random_points_proj_sig_true_pca[:, 1], 
    #     z=random_points_proj_sig_true_pca[:, 2],
    #     mode='markers',
    #     marker=dict(size=1, opacity=0.7, color = quadrant_labels_lowerdim, colorscale = "rainbow"  ),
    #     name = "random_points_proj_sig_true_pca"
    # ))


    fig.add_trace(
        go.Scatter3d(
        x=random_points_proj_sig_pca[:, 0], 
        y=random_points_proj_sig_pca[:, 1], 
        z=random_points_proj_sig_pca[:, 2],
        mode='markers',
        marker=dict(color=[colors[i] for i in predicted_class], size=1, opacity=0.7),
        name = "predicted_class"
    ))


    # fig.add_trace(
    #     go.Scatter3d(
    #     x=random_points_proj_sig_pca[:, 0], 
    #     y=random_points_proj_sig_pca[:, 1], 
    #     z=random_points_proj_sig_pca[:, 2],
    #     mode='markers',
    #     marker=dict(size=1, opacity=0.7, color = labels_cube, colorscale = "rainbow"  ),
    #     name = "labels_cube"
    # ))


    fig.add_trace(
        go.Scatter3d(
        x=X_input_proj_sig_pca[:, 0], 
        y=X_input_proj_sig_pca[:, 1], 
        z=X_input_proj_sig_pca[:, 2],
        mode='markers',
        hovertext=labels_cube_train,
        marker=dict(size=1, opacity=0.8, color=[colors[i] for i in y_train], colorscale = "rainbow"  ),
        name = "y_train"
    )
    )

    for selected_cluster in SELECTED_INDEX_CLUSTER:
        
        found_points_pca = pca.transform(sigmoid(found_points[selected_cluster].T))

        fig.add_trace(go.Scatter3d(
            x=found_points_pca[:, 0], 
            y=found_points_pca[:, 1], 
            z=found_points_pca[:, 2],
            mode='markers',
            marker=dict(size=2, opacity=0.7, color=colors[found_labels[selected_cluster]]),
            name = f"generated points in {selected_cluster} cluster"
            ))


    fig.update_layout(template= "plotly_dark")











    # grid = np.linspace(-5, 5, 20)
    # x, y = np.meshgrid(grid, grid)

    # # XY plane: z = 0
    # fig.add_trace(go.Surface(
    #     x=x, y=y, z=np.zeros_like(x),
    #     colorscale='Blues', opacity=0.5,
    #     showscale=False,
    #     name='XY Plane'
    # ))

    # # YZ plane: x = 0
    # fig.add_trace(go.Surface(
    #     x=np.zeros_like(x), y=x, z=y,
    #     colorscale='Greens', opacity=0.5,
    #     showscale=False,
    #     name='YZ Plane'
    # ))

    # # XZ plane: y = 0
    # fig.add_trace( go.Surface(
    #     x=x, y=np.zeros_like(y), z=y,
    #     colorscale='Reds', opacity=0.5,
    #     showscale=False,
    #     name='XZ Plane'
    # ))



















    fig.show()

    #exit()
    #_______________________________



    selected_layer = 1
    M = model.get_weights()[selected_layer*2].T
    B = model.get_weights()[selected_layer*2+1].T

    #random_points_proj_sig = relu(random_points_proj)

    random_points_proj_sig = np.dot(M, random_points_proj_sig.T).T + B.T
    random_points_base = np.dot(M, random_points_base.T).T + B.T
    X_input_proj_sig = np.dot(M, X_input_proj_sig.T).T + B.T



    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=random_points_base[:, 0], 
        y=random_points_base[:, 1], 
        z=random_points_base[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.4, color = "gray" ),
        name = "base_space"
    ))



    fig.add_trace(go.Scatter3d(
        x=random_points_proj_sig[:, 0], 
        y=random_points_proj_sig[:, 1], 
        z=random_points_proj_sig[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.7, color = quadrant_labels_lowerdim, colorscale = "rainbow"  ),
        name = "quadrant_labels_lowerdim"
    ))


    fig.add_trace(
        go.Scatter3d(
        x=random_points_proj_sig[:, 0], 
        y=random_points_proj_sig[:, 1], 
        z=random_points_proj_sig[:, 2],
        mode='markers',
        marker=dict(color=[colors[i] for i in predicted_class], size=1, opacity=0.7),
        name = "predicted_class"
    ))


    # fig.add_trace(
    #     go.Scatter3d(
    #     x=random_points_proj_sig[:, 0], 
    #     y=random_points_proj_sig[:, 1], 
    #     z=random_points_proj_sig[:, 2],
    #     mode='markers',
    #     marker=dict(size=1, opacity=0.7, color = labels_cube, colorscale = "rainbow"  ),
    #     name = "labels_cube"
    # ))


    fig.add_trace(
        go.Scatter3d(
        x=X_input_proj_sig[:, 0], 
        y=X_input_proj_sig[:, 1], 
        z=X_input_proj_sig[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.8, color=[colors[i] for i in y_train], colorscale = "rainbow"  ),
        name = "y_train"
    )
    )

    fig.update_layout(template= "plotly_dark")


    fig.show()


    #ANALYSIS

    #why is it that adding the random points from 0 to 1 spreads them on a line but from -1 to 1 spans the entire space
    # plot the fully randomly generated points which just pass the threshold and see where they land, as their "contour" images were really good
