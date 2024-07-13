import numpy as np

# Input and output data
x = np.array([1., 3., 10., 13., 7.])
y = np.array([2., 6., 20., 26., 14.])

# Initialize the parameter 'a' randomly
a = np.random.rand()

# Hyperparameters
learning_rate = 0.001
iterations = 100
N = x.size

for i in range(iterations):
    # Compute the predictions
    y_pred = a * x

    # Calculate the loss (Mean Squared Error)
    loss = (1/N) * np.sum((y_pred - y) ** 2)

    # Calculate the gradient
    gradient = (2/N) * np.sum(x * (y_pred - y))

    # Update the parameter 'a'
    a = a - learning_rate * gradient

    # Print the loss and parameter every 10 iterations
    if (i+1) % 10 == 0:
        print(f"Iteration {i+1}: Loss = {loss:.4f}, 'a' = {a:.4f}")

print(f"Optimal parameter 'a': {a:.4f}")
