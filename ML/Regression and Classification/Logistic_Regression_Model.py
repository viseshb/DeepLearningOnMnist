import numpy as np
import matplotlib.pyplot as plt 
import os

x_train = np.array([0, 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])
m = len(x_train)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def z_score_norm(x):
    mu = np.mean(x, axis = 0)
    sigma = np.std(x, axis =0)
    if sigma == 0:
        sigma = 1
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma
x_train_norm, x_mu, x_sigma = z_score_norm(x_train)


def compute_cost(x, y, w, b, lambda_):
    total_cost = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        total_cost += -(y[i]* np.log(f_wb_i) + (1- y[i])*np.log(1 - f_wb_i))
    regularization = (lambda_/(2*m))*(w**2)
    return (total_cost/m) + regularization

def compute_gradient(x, y, w, b, lambda_):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_wb_i = sigmoid(z_i)
        dj_dw += (f_wb_i - y[i])* x[i]
        dj_db += f_wb_i - y[i]
    dj_dw = dj_dw/m + (lambda_/(m))*w
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(x, y, w, b, lambda_, alpha, iterations):
    dw = 0
    db = 0
    cost_history = []
    for i in range(iterations):
        # z_i = np.dot(w, x[i]) + b
        # f_wb_i = sigmoid(z_i)
        dw, db = compute_gradient(x, y, w, b, lambda_)
        w = w - alpha*dw
        b = b - alpha*db  
        cost_history.append(compute_cost(x, y, w, b, lambda_))
        if i % 100 == 0:
            print(f"Iteration: {i}, Cost: {cost_history[-1]}, w:{w}, b:{b}")

    return w, b, cost_history

initial_w = 0
initial_b = 0
iterations = 1000
lambda_ = 0.01
alpha = 0.5

final_w, final_b, cost_history = gradient_descent(x_train_norm, y_train, initial_w, initial_b, lambda_, alpha, iterations)

def predict(x, w, b, mu, sigma):
    x_norm = (x - mu)/ sigma
    return sigmoid(np.dot(w, x_norm) + b)

final_prediction = predict(x_train, final_w, final_b, x_mu, x_sigma)
print(f"Prediction for x = 2: {predict(2, final_w, final_b, x_mu, x_sigma):.4f}")

# Custom predictions
print("\nPredictions for custom inputs:")
for x_val in [1, 2, 2.5, 3, 4]:
    pred_prob = predict(x_val, final_w, final_b, x_mu, x_sigma)
    pred_class = int(pred_prob >= 0.5)
    print(f"x = {x_val}, predicted probability = {pred_prob:.4f}, predicted class = {pred_class}")

# Training Accuracy
predicted_classes = final_prediction >= 0.5
accuracy = np.mean(predicted_classes == y_train)
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# Plot predictions and decision boundary
plt.plot(x_train, final_prediction, c='b', label='Model Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Labels')
decision_boundary_normalized = -final_b / final_w
decision_boundary_original = decision_boundary_normalized * x_sigma + x_mu
plt.axvline(x=decision_boundary_original, color='gray', linestyle='--', label=f'Decision Boundary ≈ {decision_boundary_original:.2f}')
plt.title("Logistic Regression with L2 Regularization")
plt.xlabel("Feature (x)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ======== Visualization Directory ========
current_dir = os.path.dirname(__file__)
save_dir = os.path.abspath(os.path.join(current_dir, "..","images/Logistic_Regression"))
os.makedirs(save_dir, exist_ok=True)

plt.show()

# Plot cost convergence
plt.figure()
plt.plot(range(iterations), cost_history, label="Cost")
plt.title("Cost Function Convergence with Regularization")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Logistic_Regression_Cost_Convergence.png"))
plt.show()

