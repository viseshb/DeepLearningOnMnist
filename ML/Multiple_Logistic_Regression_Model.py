import numpy as np
import matplotlib.pyplot as plt
import os

# === Create Save Directory ===
save_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(save_dir, exist_ok=True)

# === Input Data ===
x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5],
                    [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
m, n = x_train.shape

# === Z-score Normalization ===
def z_score_norm(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

x_train_norm, x_mu, x_sigma = z_score_norm(x_train)

# === Sigmoid ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# === Cost Function with L2 ===
def compute_cost(x, y, w, b, lambda_):
    total_cost = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        total_cost += -(y[i]*np.log(f_wb_i) + (1 - y[i])*np.log(1 - f_wb_i))
    reg_term = (lambda_ / (2 * m)) * np.sum(w**2)
    return total_cost / m + reg_term

# === Gradient with L2 ===
def compute_gradient(x, y, w, b, lambda_):
    dj_dw = np.zeros_like(w)
    dj_db = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        dj_dw += (f_wb_i - y[i]) * x[i]
        dj_db += f_wb_i - y[i]
    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_dw, dj_db

# === Gradient Descent ===
def gradient_descent(x, y, w, b, alpha, num_iters, lambda_):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b, lambda_)
        cost_history.append(cost)
        if i % 10 == 0:
            print(f"Iteration {i}: Cost={cost:.4f}, w={w}, b={b:.4f}")
    return w, b, cost_history

# === Initialize Parameters ===
initial_w = np.zeros(n)
initial_b = 0
iterations = 100
alpha = 1
lambda_ = 1  # L2 regularization strength

# === Train the model ===
final_w, final_b, cost_history = gradient_descent(
    x_train_norm, y_train, initial_w, initial_b, alpha, iterations, lambda_
)

# === Predict Function with Z3 ===
def compute_model_output(x, w, b, mu, sigma):
    x = np.array(x)
    x_norm = (x - mu) / sigma
    return sigmoid(np.dot(w, x_norm.T) + b)

# === Custom Predictions ===
print("\nPredictions for custom inputs:")
custom_points = [[1, 1], [2, 2], [2.5, 1.5], [3, 1], [4, 1]]
for x_val in custom_points:
    pred_prob = compute_model_output(x_val, final_w, final_b, x_mu, x_sigma)
    pred_class = int(pred_prob >= 0.5)
    print(f"x = {x_val}, predicted probability = {pred_prob:.4f}, predicted class = {pred_class}")

# === Training Accuracy ===
train_probs = compute_model_output(x_train, final_w, final_b, x_mu, x_sigma)
predicted_classes = train_probs >= 0.5
accuracy = np.mean(predicted_classes == y_train)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# === Decision Boundary Plot ===
x1_vals = np.linspace(0, 4, 100)
x2_vals = np.linspace(0, 4, 100)
xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = compute_model_output(grid, final_w, final_b, x_mu, x_sigma).reshape(xx1.shape)

# ======== Visualization Directory ========
save_dir = os.path.join(os.path.dirname(__file__), "images/Multiple_Logistic_Regression")
os.makedirs(save_dir, exist_ok=True)

plt.figure()
plt.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], alpha=0.6, colors=["lightcoral", "lightblue"])
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
plt.title("Logistic Regression Decision Boundary (Z3 + L2)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Multiple_Logistic_Regression.png"))
plt.show()

# === Cost Plot ===
plt.figure()
plt.plot(range(iterations), cost_history)
plt.title("Cost Function Convergence with L2")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Multiple_Logistic_Regression_Cost_Convergence.png"))
plt.show()
