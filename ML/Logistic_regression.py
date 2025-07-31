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

def compute_model_output(x, w, b):
    g = w * x + b
    return sigmoid(g)

w = 2
b = -5
tmp_f_wb = compute_model_output(x_train, w, b)
cost = compute_cost(x_train, y_train, w, b)
print(f"Cost at w={w}, b={b}: {cost:.4f}")

# Plot model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Model Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Labels')
plt.title("Logistic Regression: Model vs Actual")
plt.xlabel("Feature (x)")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "Logistic_Regression.png"))
plt.show()
