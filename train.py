import numpy as np
import copy
import math
import json

# ================================
# 1. Load dataset
# ================================
def load_data(file_path="data/data.txt"):
    
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    x = data[:, 0]  # population
    y = data[:, 1]  # profit
    return x, y

x_train, y_train = load_data()

# ================================
# 2. Compute Cost Function
# ================================
def compute_cost(x, y, w, b):
    
    m = x.shape[0]
    total_cost = 0
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        cost_sum += cost
    total_cost = cost_sum / (2 * m)
    return total_cost

# ================================
# 3. Compute Gradient
# ================================
def compute_gradient(x, y, w, b):
    
    m = x.shape[0]
    dj_dw_sum = 0
    dj_db_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_sum += (f_wb - y[i]) * x[i]
        dj_db_sum += (f_wb - y[i])
    dj_dw = dj_dw_sum / m
    dj_db = dj_db_sum / m
    return dj_dw, dj_db

# ================================
# 4. Gradient Descent
# ================================
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

# ================================
# 5. Train the Model
# ================================
initial_w = 0.
initial_b = 0.
iterations = 2000
alpha = 0.01

w, b, J_history, w_history = gradient_descent(
    x_train, y_train, initial_w, initial_b,
    compute_cost, compute_gradient, alpha, iterations
)

print("\nTraining completed.")
print(f"w = {w}, b = {b}")

# ================================
# 6. Save model parameters
# ================================
model = {
    "theta0": b,
    "theta1": w,
    "learning_rate": alpha,
    "iterations": iterations
}

# Make sure folder 'model' exists
import os
if not os.path.exists("model"):
    os.makedirs("model")

with open("model/model.json", "w") as f:
    json.dump(model, f, indent=4)

print("Model saved to model/model.json")