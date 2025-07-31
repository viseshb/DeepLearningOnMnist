import numpy as np
import matplotlib.pyplot as plt
import os

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
m = len(x_train)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def compute_cost(x, y, w, b):
    total_cost = 0    
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        total_cost += -(y[i]*np.log(f_wb_i) + (1-y[i])*np.log(1 - f_wb_i))
    return total_cost/m     

def compute_gradient(x,y,w,b):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        dj_dw += (f_wb_i - y[i])* x[i]
        dj_db += (f_wb_i -y[i])
    return dj_dw/m, dj_db/m

def gradient_descent(x,y,w,b, alpha, num_iters):
    dw = 0
    db = 0
    cost_history = []
    for i in range(num_iters):
        dw, db = compute_gradient(x,y,w,b)
        w = w - alpha * dw
        b = b - alpha* db
        cost_history.append(compute_cost(x,y,w,b))

        if i % 10 == 0:
            print(f"Iterations: {i}, Cost: {cost_history[-1]:.4f}, w:{w:.2f}, b:{b:.2f}")

    return w, b, cost_history

initial_w = 0
initial_b = 0
iterations = 100
alpha = 2

final_w, final_b, cost_history = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations)

def compute_model_output(x, w, b):
    g = np.dot(w, x) + b
    return sigmoid(g)

# Predict for training data
final_prediction = compute_model_output(x_train, final_w, final_b)
print(f"Prediction for x = 2: {compute_model_output(2, final_w, final_b):.4f}")

# === Custom predictions ===
print("\nPredictions for custom inputs:")
for x_val in [1, 2, 2.5, 3, 4]:
    pred_prob = compute_model_output(x_val, final_w, final_b)
    pred_class = int(pred_prob >= 0.5)
    print(f"x = {x_val}, predicted probability = {pred_prob:.4f}, predicted class = {pred_class}")

# === Training Accuracy ===
predicted_classes = final_prediction >= 0.5
accuracy = np.mean(predicted_classes == y_train)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# === Plot model prediction with decision boundary ===
plt.plot(x_train, final_prediction, c='b', label='Model Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Labels')

# Decision boundary
decision_boundary = -final_b / final_w
plt.axvline(x=decision_boundary, color='gray', linestyle='--', label=f'Decision Boundary ≈ {decision_boundary:.2f}')

# Labels and legend
plt.title("Logistic Regression: Model vs Actual")
plt.xlabel("Feature (x)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
save_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "Logistic_Regression.png"))
plt.show()

# === Plot cost convergence ===
plt.figure()
plt.plot(range(iterations), cost_history, label="Cost")
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Logistic_Regression_Cost_Convergence.png"))
plt.show()
