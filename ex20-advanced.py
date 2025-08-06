import math
import numpy as np

x = np.array([4, 3, 0])
c1 = np.array([-0.5, 0.1, 0.08])
c2 = np.array([-0.2, 0.2, 0.31])
c3 = np.array([0.5, -0.1, 2.53])


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Calculate liner combination and apply sigmoid for each coefficient set
print("Input vector x: ", x)
print("\nCalculating sigmoid outputs for each coefficient set:")
print("=" * 50)

# Coefficient set 1
z1 = np.dot(x, c1)
output1 = sigmoid(z1)
print(f"c1 = {c1}")
print(f"z1 = x . c1 = {x} . {c1} = {z1:.4f}")
print(f"sigmoid(z1) = {output1:.6f}")
print()

# Coefficient set 2
z2 = np.dot(x, c2)
output2 = sigmoid(z2)
print(f"c2 = {c2}")
print(f"z2 = x . c2 = {x} . {c2} = {z2:.4f}")
print(f"sigmoid(z2) = {output2:.6f}")
print()

# Coefficient set 3
z3 = np.dot(x, c3)
output3 = sigmoid(z3)
print(f"c3 = {c3}")
print(f"z3 = x . c3 = {x} . {c3} = {z3:.4f}")
print(f"sigmoid(z3) = {output3:.6f}")
print()

# Find which coefficient set yields the highest output
outputs = [output1, output2, output3]
max_output = max(outputs)
best_coeff_index = outputs.index(max_output) + 1
print("=" * 50)
print("RESULTS SUMMARY:")
print(f"c1 output: {output1:.6f}")
print(f"c2 output: {output2:.6f}")
print(f"c3 output: {output3:.6f}")
print()
print(f"The highest sigmoid output is {max_output:.6f}")
print(f"This comes from coefficient set c{best_coeff_index}")
