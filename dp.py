import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
import random
import math
from sklearn.metrics.pairwise import cosine_similarity

# Privacy-Preserving Mechanisms
# Function for Dynamic Privacy Budget Allocation (DPBA)
def compute_dpba(t, epsilon_max, beta):
    """Computes dynamic privacy budget at round t."""
    epsilon_t = epsilon_max * (1 - np.exp(-beta * t))
    return epsilon_t

# Function for Gradient Sparsification with Differential Privacy (GSDP)
def gradient_sparsification(gradients, k):
    """Sparsify gradients, keeping only top-k largest values."""
    # Flatten the gradients to ensure compatibility with np.argsort
    flat_gradients = [tf.reshape(grad, [-1]).numpy() for grad in gradients]

    sparsified_gradients = []

    for grad in flat_gradients:
        # Sort by absolute gradient value
        sorted_indices = np.argsort(np.abs(grad))  # Sort by absolute gradient value
        top_k_indices = sorted_indices[-k:]  # Select top-k indices

        # Create a zero array for sparsified gradients and retain only top-k values
        sparsified_grad = np.zeros_like(grad)
        sparsified_grad[top_k_indices] = grad[top_k_indices]

        sparsified_gradients.append(sparsified_grad)

    # Convert the sparsified gradients back to tensors
    sparsified_gradients = [tf.convert_to_tensor(grad) for grad in sparsified_gradients]

    return sparsified_gradients

# Function for Adaptive Noise Scaling (ANS)

# def adaptive_noise_scaling(gradients, epsilon_t, delta, N):
#     """Computes the adaptive noise based on gradient magnitude."""
#     # Flatten gradients before computing the norm
#     flat_gradients = [tf.reshape(grad, [-1]).numpy() for grad in gradients]

#     # Ensure each gradient is a 2D array before computing the norm
#     delta_f_t = np.mean([np.linalg.norm(grad, ord=2) for grad in flat_gradients if grad.ndim > 0])

#     # Compute noise scale
#     sigma_t_square = (2 * (delta_f_t ** 2) * np.log(1.25 / delta)) / (epsilon_t ** 2)
#     return np.sqrt(sigma_t_square)

def adaptive_noise_scaling(gradients, epsilon_t, delta, N):
    """Computes the adaptive noise based on gradient magnitude using TensorFlow operations."""
    # Compute L2 norms without converting to NumPy
    delta_f_t = tf.reduce_mean([tf.norm(grad, ord=2) for grad in gradients if grad is not None])

    # Compute noise scale
    sigma_t_square = (2 * (delta_f_t ** 2) * np.log(1.25 / delta)) / (epsilon_t ** 2)
    return tf.sqrt(sigma_t_square)

# Function for Secure Aggregation with Partial Differential Privacy (SAP-DP)
# Helper function to compute gradient similarity
def compute_similarity(g_i, g_j):
    """Compute cosine similarity between two gradient updates."""

    return cosine_similarity(g_i.reshape(1, -1), g_j.reshape(1, -1))[0][0]


# Function to perform client grouping based on gradient similarity
def group_clients_based_on_similarity(client_gradients, tau):
    """
    Group clients based on the gradient similarity threshold (tau).
    """
    N = len(client_gradients)
    clusters = []
    visited = [False] * N

    for i in range(N):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        for j in range(i + 1, N):
            if not visited[j] and compute_similarity(client_gradients[i], client_gradients[j]) > tau:
                cluster.append(j)
                visited[j] = True
        clusters.append(cluster)
    return clusters

# Function to apply Differential Privacy selectively to client updates
def apply_dp_selectively(cluster, client_gradients, alpha, sigma):
    """
    Apply DP selectively to a fraction alpha of clients in the cluster.
    """
    noisy_gradients = []

    for i in cluster:
        if random.random() < alpha:  # Apply DP noise with probability alpha
            noise = np.random.normal(0, sigma, client_gradients[i].shape)
            noisy_gradients.append(client_gradients[i] + noise)
        else:
            noisy_gradients.append(client_gradients[i])  # No noise applied

    return noisy_gradients

# Function to securely aggregate client gradients with noise reduction
def secure_aggregate(client_gradients, clusters, sigma):
    """
    Securely aggregate the gradients by averaging and adding noise.
    """
    aggregated_gradients = []

    for cluster in clusters:
        cluster_gradients = np.array([client_gradients[i] for i in cluster])
        cluster_mean = np.mean(cluster_gradients, axis=0)  # Average gradient for the cluster
        noise = np.random.normal(0, sigma, cluster_mean.shape)  # Add noise to the aggregated gradient
        aggregated_gradients.append(cluster_mean + noise)

    return aggregated_gradients

# Main function for Secure Aggregation with Partial DP (SAP-DP)
def sap_dp_training(client_gradients, tau, alpha, sigma):
    """
    Perform Federated Learning with Secure Aggregation and Partial Differential Privacy (SAP-DP).
    """
    # Step 1: Group clients based on gradient similarity
    clusters = group_clients_based_on_similarity(client_gradients, tau)

    # Step 2: Apply DP selectively to client updates within each cluster
    all_noisy_gradients = []
    for cluster in clusters:
        noisy_gradients = apply_dp_selectively(cluster, client_gradients, alpha, sigma)
        all_noisy_gradients.extend(noisy_gradients)

    # Step 3: Securely aggregate the gradients from all clusters
    aggregated_gradients = secure_aggregate(all_noisy_gradients, clusters, sigma)

    return aggregated_gradients

# Load EMNIST-10 Dataset
def load_emnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Split data into clients
def create_clients(x, y, num_clients=10, batch_size=32):
    client_data = []
    data_per_client = len(x) // num_clients
    for i in range(num_clients):
        start = i * data_per_client
        end = start + data_per_client
        dataset = tf.data.Dataset.from_tensor_slices((x[start:end], y[start:end])).batch(batch_size)
        client_data.append(dataset)
    return client_data


# Define CNN Model
def create_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Federated Training with Privacy-Preserving Mechanisms
def client_update(model, dataset, round_num, epsilon_max, beta, delta, k):
    for batch in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch[0], training=True)
            loss = tf.keras.losses.categorical_crossentropy(batch[1], predictions)
        gradients = tape.gradient(loss, model.trainable_variables)

        # Apply Privacy-Preserving Mechanisms
        epsilon_t = compute_dpba(round_num, epsilon_max, beta)

        # Apply Gradient Sparsification with DP
        sparsified_gradients = gradient_sparsification(gradients, k)

        # Process gradients layer-wise
        processed_gradients = []
        for grad in sparsified_gradients:
            grad = grad.numpy() if isinstance(grad, tf.Tensor) else grad
            noise_scale = adaptive_noise_scaling(grad, epsilon_t, delta, len(dataset))
            noisy_gradient = grad + np.random.normal(0, noise_scale, grad.shape)
            processed_gradients.append(noisy_gradient)

    return processed_gradients  # Return processed gradients, NOT aggregated ones

def federated_train(global_model, clients_data, epsilon_max, beta, delta, alpha, k, rounds=10):
    history = {'loss': [], 'accuracy': []}
    communication_cost = 0
    start_time = time.time()

    for r in range(rounds):
        client_gradients = []  # Store all client gradients per round

        for dataset in clients_data:
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            gradients = client_update(local_model, dataset, r, epsilon_max, beta, delta, k)
            client_gradients.append(gradients)
            communication_cost += sum([w.nbytes for w in gradients])

        # Perform Secure Aggregation with Partial DP (SAP-DP)
        tau = 0.9  # Similarity threshold
        alpha = 0.5  # Fraction of clients applying DP
        sigma = 0.1  # Noise standard deviation
        aggregated_gradients = sap_dp_training(client_gradients, tau, alpha, sigma)

        # Apply aggregated gradients to the global model
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(aggregated_gradients, global_model.trainable_variables))

        # Evaluate model
        loss, accuracy = global_model.evaluate(x_test_emnist, y_test_emnist, verbose=0)
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        print(f'Round {r+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}')

    end_time = time.time()

    convergence_time = end_time - start_time
    print(f'Total Convergence Time: {convergence_time:.2f} seconds')
    print(f'Final Communication Overhead: {communication_cost / (1024 * 1024):.2f} MB')

    return global_model, history

# Load datasets
(x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = load_emnist()

# Create clients
clients_data_emnist = create_clients(x_train_emnist, y_train_emnist)

# Initialize Global Model
global_model = create_model()

# Train on EMNIST
print("Training on EMNIST")
global_model, history = federated_train(global_model, clients_data_emnist, epsilon_max=5.0, beta=0.1, delta=1e-5, alpha=0.5, k=10, rounds=20)

# Evaluate Model
loss, accuracy = global_model.evaluate(x_test_emnist, y_test_emnist)
print(f'Test Accuracy: {accuracy}')

y_pred = global_model.predict(x_test_emnist)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_emnist, axis=1)

# Print Precision, Recall, F1-Score
print("Classification Report:")
print(classification_report(y_true, y_pred_classes))

# Print Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Plot Training Results
def plot_training_history(history, dataset_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {dataset_name}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy - {dataset_name}')
    plt.legend()

    plt.show()

plot_training_history(history, "EMNIST")

# # Plot Noise Impact on Gradients
# def plot_noise_impact(gradient_norms):
#     original_norms, noisy_norms = zip(*gradient_norms)
#     plt.figure(figsize=(10, 5))
#     plt.plot([np.mean(norms) for norms in original_norms], label='Original Gradient Norm')
#     plt.plot([np.mean(norms) for norms in noisy_norms], label='Noisy Gradient Norm')
#     plt.xlabel('Rounds')
#     plt.ylabel('Gradient Norm')
#     plt.title('Noise Impact on Gradient Magnitude')
#     plt.legend()
#     plt.show()

# plot_noise_impact(gradient_norms)
