import numpy as np
import matplotlib.pyplot as plt
import os
save_dir = os.path.dirname(os.path.abspath(__file__))

x_train = np.array([[2104, 5, 1, 45],
                   [1416, 3, 2, 40],
                   [852, 2, 1, 35]],dtype=float)
y_train = np.array([460, 232, 178], dtype=float)

m,n = x_train.shape

def compute_cost(x, y, w, b):
    total_cost = 0
    for i in range(m):
        prediction = np.dot(w, x[i]) + b
        total_cost += (prediction - y[i])**2
    return total_cost/(2*m)

def compute_gradient(x, y, w, b):
    dj_dw =0
    dj_db = 0
    for i in range(m):
        dj_dw += (np.dot(w, x[i]) + b - y[i])*x[i]
        dj_db += (np.dot(w, x[i]) + b - y[i])
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w, b, alpha, number_iterations):
    dw =0
    db =0
    cost_history = []
    for i in range(number_iterations):
        dw, db = compute_gradient(x, y, w, b)
        w = w - alpha *dw
        b = b - alpha * db
        cost_history.append(compute_cost(x,y, w, b)) 

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]:.4f}, w = {w}, b = {b:.4f}")

    return w, b, cost_history

initial_w = np.zeros(n)
initial_b = 0
alpha = 0.00000001
iterations = 1000

final_w, final_b, cost_history = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations)

def predict(x, w, b):
    return np.dot(x, w) + b

final_prediction = predict([300, 2, 1, 30], final_w, final_b)
print(f"Predicted price of 300 sq ft house: ${final_prediction * 1000:.2f}")



test_data = [
    [300, 2, 1, 30],
    [1000, 3, 2, 20],
    [1500, 4, 2, 5]
]
for house in test_data:
    price = predict(house, final_w, final_b)
    print(f"House {house} => Predicted price: ${price * 1000:.2f}")


print("\nFinal learned parameters:")
for i, w_i in enumerate(final_w):
    print(f"w{i+1} = {w_i:.6f}")
print(f"bias = {final_b:.6f}")

plt.figure(figsize=(8,5))
plt.plot(range(iterations), cost_history, linewidth=2)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "cost_convergence_Multiple_Regression.png"))
plt.show()