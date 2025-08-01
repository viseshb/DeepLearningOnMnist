import numpy as np
import os
import matplotlib.pyplot as plt 
save_dir = os.path.dirname(os.path.abspath(__file__))

# ========== Data ==========
x_train = np.array([100, 1000])  # sq ft
y_train = np.array([20, 150])    # in $1000s
m = len(x_train)

# ========== Z-score Normalization ==========
def z_score_norm(x):
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        sigma = 1  # avoid division by zero
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

x_train_norm, x_mu, x_sigma = z_score_norm(x_train)

# ========== Cost Function with L2 Regularization ==========
def compute_cost(x, y, w, b, lambda_):
    total_cost = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        total_cost += (f_wb_i - y[i]) ** 2
    mse = total_cost / (2 * m)
    reg = (lambda_ / (2 * m)) * (w ** 2)  # L2 penalty
    return mse + reg

# ========== Gradient with L2 Regularization ==========
def compute_gradient(x, y, w, b, lambda_):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        dj_dw += (f_wb_i - y[i]) * x[i]
        dj_db += f_wb_i - y[i]
    dj_dw = dj_dw / m + (lambda_ / m) * w  # L2 gradient term
    dj_db = dj_db / m
    return dj_dw, dj_db

# ========== Gradient Descent ==========
def gradient_descent(x, y, w, b, alpha, num_iters, lambda_):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b, lambda_)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

    return w, b, cost_history

# ========== Training ==========
initial_w = 0
initial_b = 0
alpha = 0.01
iterations = 1000
lambda_ = 0.1

final_w, final_b, cost_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b, alpha, iterations, lambda_)

# ========== Prediction Function ==========
def predict(w, x, b, mu, sigma):
    x_norm = (x - mu) / sigma
    return w * x_norm + b

predicted_price = predict(final_w, 300, final_b, x_mu, x_sigma)
print(f"\nPrice of 300 sqft house: ${predicted_price * 1000:.2f}")

# ======== Visualization Directory ========
current_dir = os.path.dirname(__file__)
save_dir = os.path.abspath(os.path.join(current_dir, "..","images/Linear_regression"))
os.makedirs(save_dir, exist_ok=True)


plt.scatter(x_train, y_train, color='blue', label='Data Points')
x_line = np.linspace(min(x_train), max(x_train), 100)
y_line = predict(final_w, x_line, final_b, x_mu, x_sigma)
plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($1000s)")
plt.title("Regularized Linear Regression")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "Linear_regression_line.png"))
plt.show()

# ========== Plot Cost Function ==========
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence (with L2 Regularization)")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "Linear_Regression_cost_convergence.png"))
plt.show()
