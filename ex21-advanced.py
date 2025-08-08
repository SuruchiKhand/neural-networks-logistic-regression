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
    # ReLu activation: returns max(0,z) for each element
    return np.maximum(0, z)


def output_activation(z):
    # Identity (linear) activation: return input unchanged
    return z


x_test = [[82, 2, 65, 3, 516]]

for item in x_test:
    print(f"Input: {item}")

    # First hidden layer
    h1_in = np.dot(item, w0) + b0  # Linear combination of inputs and weights + bias
    h1_out = hidden_activation(h1_in)  # Apply ReLu activation function
    print(f"First hidden layer output: {h1_out}")

    # Second hidden layer
    h2_in = (
        np.dot(h1_out, w1) + b1
    )  # Linear combination of h1_out and weights w1 + bias
    h2_out = hidden_activation(h2_in)
    print(f"Second hidden layer output: {h2_out}")

    # Output layer
    out_in = (
        np.dot(h2_out, w2) + b2
    )  # Linear combination of h2_out and weights w2 + bias b2
    price_prediction = output_activation(out_in)

    print(f"Predicted cabin price: ${price_prediction[0]:.2f}")
