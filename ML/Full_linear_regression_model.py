import numpy as np
import os
import matplotlib.pyplot as plt 
save_dir = os.path.dirname(os.path.abspath(__file__))

x_train = np.array([100, 1000]) # in sqfeet
y_train = np.array([20, 150]) # in kdollars
m = x_train.shape[0]

def compute_cost(x,y, w,b):
    total_cost = 0
    for i in range(m):
        total_cost += (w*x[i] + b - y[i])**2
    return total_cost/(2*m)

def compute_gradient(x, y, w, b):
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        dj_dw += (w*x[i] + b - y[i])*x[i]
        dj_db += w*x[i] + b - y[i]
    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w, b, alpha, num_iters):
    dw =0
    db =0
    cost_history = []
    for i in range(num_iters):
        dw, db = compute_gradient(x, y, w, b)
        w = w - alpha*dw
        b = b - alpha*db
        cost_history.append(compute_cost(x,y, w, b))

        if i%10 == 0:
            print(f"Iteration {i}: Cost = {cost_history[-1]:.4f}, w = {w:.2f}, b = {b:.2f}")

    return w, b, cost_history

initial_w = 0
initial_b = 0
alpha = 0.000001
iterations = 100

final_w, final_b, cost_history = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations)

def predict(w, x, b):
    return w*x + b

predicted_price = predict(final_w, 300, final_b)
print(f"\nPrice of 300 sqft house: ${predicted_price * 1000:.2f}")


# Plot the data and regression line
plt.scatter(x_train, y_train, color='blue', label='Data Points')
plt.plot(x_train, predict(final_w,x_train, final_b), color='red', label='Regression Line')
plt.xlabel("Size (sq ft)")
plt.ylabel("price ($1000s)")
plt.title("Linear Regression Model")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "regression_line.png"))
plt.show()


# Plot cost vs iteration
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.savefig(os.path.join(save_dir, "cost_convergence.png"))
plt.show()
