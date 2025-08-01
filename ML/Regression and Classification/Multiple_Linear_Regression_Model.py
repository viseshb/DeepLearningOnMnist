import numpy as np
import matplotlib.pyplot as plt
import os

# ======== Setup ========
save_dir = os.path.join(os.path.dirname(__file__), "images/Multiple_Linear_regression")
os.makedirs(save_dir, exist_ok=True)

# ======== Input Data ========
x_train = np.array([
    [2104, 5, 1, 45],
    [1416, 3, 2, 40],
    [852,  2, 1, 35]
], dtype=float)

y_train = np.array([460, 232, 178], dtype=float)

m, n = x_train.shape
features = ['Sqft', 'Bedrooms', 'Bathrooms', 'Age']  # ✅ Added missing feature names

# ======== Z-score Normalization ========
def z_score_norm(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1  # avoid division by zero
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

x_train_norm, x_mu, x_sigma = z_score_norm(x_train)

# ======== Cost Function with L2 Regularization ========
def compute_cost(x, y, w, b, lambda_):
    total_cost = 0
    for i in range(m):
        prediction = np.dot(w, x[i]) + b
        total_cost += (prediction - y[i])**2
    mse = total_cost / (2 * m)
    reg_term = (lambda_ / (2 * m)) * np.sum(w**2)
    return mse + reg_term

# ======== Gradient Function with L2 Regularization ========
def compute_gradient(x, y, w, b, lambda_):
    dj_dw = np.zeros_like(w)
    dj_db = 0
    for i in range(m):
        error = np.dot(w, x[i]) + b - y[i]
        dj_dw += error * x[i]
        dj_db += error
    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_dw, dj_db

# ======== Gradient Descent ========
def gradient_descent(x, y, w, b, alpha, number_iterations, lambda_):
    cost_history = []
    for i in range(number_iterations):
        dw, db = compute_gradient(x, y, w, b, lambda_)
        w -= alpha * dw
        b -= alpha * db
        cost_history.append(compute_cost(x, y, w, b, lambda_))
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]:.4f}, w = {w}, b = {b:.4f}")
    return w, b, cost_history

# ======== Training ========
initial_w = np.zeros(n)
initial_b = 0
alpha = 0.01
iterations = 1000
lambda_ = 1  # L2 regularization strength

final_w, final_b, cost_history = gradient_descent(
    x_train_norm, y_train, initial_w, initial_b, alpha, iterations, lambda_)

# ======== Prediction Function ========
def predict(x, w, b, mu, sigma):
    x = np.array(x)
    x_norm = (x - mu) / sigma
    return np.dot(x_norm, w) + b

# ======== Test Predictions ========
print("\nTest Predictions:")
test_data = [
    [300, 2, 1, 30],
    [1000, 3, 2, 20],
    [1500, 4, 2, 5]
]

for house in test_data:
    price = predict(house, final_w, final_b, x_mu, x_sigma)
    print(f"House {house} => Predicted price: ${price * 1000:.2f}")

# ======== Final Parameters ========
print("\nFinal learned parameters:")
for i, w_i in enumerate(final_w):
    print(f"{features[i]} = {w_i:.6f}")
print(f"Bias = {final_b:.6f}")

# ======== Cost Plot ========
plt.figure(figsize=(8, 5))
plt.plot(range(iterations), cost_history, linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE + L2 Penalty)")
plt.title("Cost Function Convergence with L2 Regularization")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Multiple_Linear_Regression_cost_convergence.png"))
plt.show()

# ======== Feature Weight Bar Plot ========
plt.figure(figsize=(6, 4))
plt.bar(features, final_w)
plt.title("Feature Weights with Regularization")
plt.ylabel("Weight")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Multiple_linear_regression_feature_weights.png"))
plt.show()
