# demonstration of ridge wavelet neural network.
# it uses the Wavelet function as the activation function.
# here we will be using the mexican hat Wavelet as the activation function and binary cross entropy as the loss function.
# Also as we are not getting any binary output from the function as the output we will be using sigmoid function and threshold for conversion 
# of values into binary values

# importing libraries
import pandas as pd 
import math as m
import random as rd

# data ingestion 
try:
    df = pd.read_csv("/home/Zeus/Documents/PD1/Data/dataset.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Using generated dummy data for testing.")
    # Fallback dummy data so the script runs if the file is missing or path changes
    df = pd.DataFrame([[rd.random() for _ in range(4)] + [rd.choice([0, 1])] for _ in range(200)], 
                      columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'label'])

X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].values.tolist()
Y = df['label'].tolist()

# train test split
X_train = X[:160]
X_test = X[160:]

Y_train = Y[:160]
Y_test = Y[160:]

# initialization 
w = []
for i in range(4):
    w.append(rd.uniform(-0.5, 0.5)) 

# Smart Initialization for 'a' (Dilation) based on dataset standard deviation
all_train_vals = [val for row in X_train for val in row]
mean_val = sum(all_train_vals) / len(all_train_vals)
variance = sum((val - mean_val)**2 for val in all_train_vals) / len(all_train_vals)
std_dev = m.sqrt(variance) if variance > 0 else 1.0

a = std_dev                 # Dilation (Spread)
b = 0.0                     # Translation (Shift)
c = rd.uniform(-0.5, 0.5)   # Scaling Factor (Output weight)

# forward driver functions
def forward_pass(x, w, a, b, c):
    # 1. Linear combination
    z = sum(w[i] * x[i] for i in range(len(x)))
    
    # 2. Wavelet argument (shift and dilate)
    t = (z - b) / a
    
    # 3. Mexican Hat activation (dropping the normalization constant)
    psi = (1 - t**2) * m.exp(-0.5 * t**2)
    
    # 4. Scaling
    v = c * psi
    
    # 5. Sigmoid conversion (with overflow protection)
    if v < -100:
        y_hat = 0.0
    else:
        y_hat = 1 / (1 + m.exp(-v))
        
    return z, t, psi, v, y_hat

# loss function
def calculate_bce_loss(y_true, y_pred):
    # Epsilon prevents math.log(0) errors
    eps = 1e-15
    y_pred = max(min(y_pred, 1 - eps), eps)
    return -(y_true * m.log(y_pred) + (1 - y_true) * m.log(1 - y_pred))

# change calculation and updation (Backpropagation)
def backpropagate(x, y_true, y_pred, t, psi, c, a, b, w, lr):
    # 1. Base error from BCE + Sigmoid derivative simplification
    delta_v = y_pred - y_true
    
    # 2. Gradient for scaling factor 'c'
    grad_c = delta_v * psi
    
    # 3. Derivative of Mexican Hat and intermediate error
    psi_prime = t * (t**2 - 3) * m.exp(-0.5 * t**2)
    delta_t = delta_v * c * psi_prime
    
    # 4. Gradients for a, b, and w
    grad_a = delta_t * (-t / a)
    grad_b = delta_t * (-1 / a)
    grad_w = [delta_t * (x[i] / a) for i in range(len(x))]
    
    # 5. Apply updates using gradient descent
    c_new = c - lr * grad_c
    a_new = a - lr * grad_a
    b_new = b - lr * grad_b
    w_new = [w[i] - lr * grad_w[i] for i in range(len(w))]
    
    return c_new, a_new, b_new, w_new

# main function (Training Loop)
def train_network(epochs, learning_rate):
    global w, a, b, c 
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0 
        
        # Iterating through the training dataset sample by sample
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = Y_train[i]
            
            # Forward pass
            z, t, psi, v, y_pred = forward_pass(x, w, a, b, c)
            
            # Check accuracy for this specific sample
            predicted_class = 1 if y_pred >= 0.5 else 0
            if predicted_class == y_true:
                correct_predictions += 1
            
            # Calculate loss
            total_loss += calculate_bce_loss(y_true, y_pred)
            
            # Backpropagation & Update
            c, a, b, w = backpropagate(x, y_true, y_pred, t, psi, c, a, b, w, learning_rate)
            
        # Print progress and accuracy every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(X_train)
            train_accuracy = (correct_predictions / len(X_train)) * 100
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

# test
def evaluate_network():
    print("\n--- Testing Model ---")
    correct_predictions = 0
    total_samples = len(X_test)
    
    for i in range(total_samples):
        x = X_test[i]
        y_true = Y_test[i]
        
        # We only need the final prediction for testing
        _, _, _, _, y_pred = forward_pass(x, w, a, b, c)
        
        # Apply threshold (0.5) for binary classification
        predicted_class = 1 if y_pred >= 0.5 else 0
        
        if predicted_class == y_true:
            correct_predictions += 1
            
    accuracy = (correct_predictions / total_samples) * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples} correct)")

# execution
epochs = 100
learning_rate = 0.05

train_network(epochs, learning_rate)
evaluate_network()