import numpy as np

w0 = np.array(
    [
        [1.19627687e01, 2.60163283e-01],
        [4.48832507e-01, 4.00666119e-01],
        [-2.75768443e-01, 3.43724167e-01],
        [2.29138536e01, 3.91783025e-01],
        [-1.22397711e-02, -1.03029800e00],
    ]
)

w1 = np.array([[11.5631751, 11.87043684], [-0.85735419, 0.27114237]])

w2 = np.array([[11.04122165], [10.44637262]])

b0 = np.array([-4.21310294, -0.52664488])
b1 = np.array([-4.84067881, -4.53335139])
b2 = np.array([-7.52942418])

x = np.array(
    [
        [111, 13, 12, 1, 161],
        [125, 13, 66, 1, 468],
        [46, 6, 127, 2, 961],
        [80, 9, 80, 2, 816],
        [33, 10, 18, 2, 297],
        [85, 9, 111, 3, 601],
        [24, 10, 105, 2, 1072],
        [31, 4, 66, 1, 417],
        [56, 3, 60, 1, 36],
        [49, 3, 147, 2, 179],
    ]
)
y = np.array(
    [
        335800.0,
        379100.0,
        118950.0,
        247200.0,
        107950.0,
        266550.0,
        75850.0,
        93300.0,
        170650.0,
        149000.0,
    ]
)


def hidden_activation(z):
    # ReLu activation: returns max(0, z) for each element
    return np.maximum(0, z)


def output_activation(z):
    # Idnetify (linear) activation: returns input unchanged
    return z


# Test cases
x_test = [[72, 2, 25, 3, 450], [60, 3, 15, 1, 300], [74, 5, 10, 2, 100]]

print("Neural Network Forward Pass Results:")
print("=", 40)

for i, item in enumerate(x_test):
    print(f"Input {i+1}: {item}")

    # First hidden layer: Linear combination + bias + ReLu activation
    h1_in = np.dot(item, w0) + b0  # Added bias term b0
    h1_out = hidden_activation(h1_in)  # ReLu activation
    print(f" Hidden layer 1 input: {h1_in}")
    print(f" Hidden layer 1 output: {h1_out}")

    # Second hidden layer: Linear combination + bias + ReLu activation
    h2_in = np.dot(h1_out, w1) + b1
    h2_out = hidden_activation(h2_in)  # ReLu activation
    print(f" Hidden layer 2 input: {h2_in}")
    print(f" Hidden layer 2 output: {h2_out}")

    # Output layer: Linear combination + bias + linear activation
    out_in = np.dot(h2_out, w2) + b2  # Added bias term b2
    out = output_activation(out_in)  # Linear (identity) activation
    print(f" Output layer input: {out_in}")
    print(f" Final output (price): ${out[0]:.2f}")
    print()

# Prediction for [74, 5, 10, 2, 100]
print("=" * 40)
print("ANSWER: For cabin with features [ 74, 5, 10, 2, 100]:")
item = [74, 5, 10, 2, 100]
h1_in = np.dot(item, w0) + b0
h1_out = hidden_activation(h1_in)
h2_in = np.dot(h1_out, w1) + b1
h2_out = hidden_activation(h2_in)
out_in = np.dot(h2_out, w2) + b2
out = output_activation(out_in)
print(f"Predicted cabin price: ${out[0]:.2f}")
