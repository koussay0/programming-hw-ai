
import numpy as np


def create_random_matrix(n_rows, n_cols, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=(n_rows, n_cols))


def threshold_fire(Vin, tau):
    Vin = np.asarray(Vin)
    return (Vin > tau).astype(float)


def process_hidden_layers(SM, bvh, bvout, V0_out, tau):
    V = V0_out
    # SM: list of matrices for hidden layers, last one maps to output
    for k, Mk in enumerate(SM[:-1]):
        V_in = Mk @ V + bvh
        V = threshold_fire(V_in, tau)
    # output layer
    V_in_out = SM[-1] @ V + bvout
    V_out = threshold_fire(V_in_out, tau)
    return V_out


def feedforward_neural_network(Sin, Sout, Seqtr, tau_fire=0.5, bias=0.0, cycles=10,
                               n_hidden_layers=1, hidden_size=4, learning_rate=0.1):
    rng = np.random.default_rng(0)
    sin = Sin
    sout = Sout

    # unpack training data
    inputs = [np.asarray(p[0], dtype=float).reshape(sin, 1) for p in Seqtr]
    targets = [np.asarray(p[1], dtype=float).reshape(sout, 1) for p in Seqtr]

    # initialize weights
    M0 = create_random_matrix(hidden_size, sin, rng)
    hidden_mats = [create_random_matrix(hidden_size, hidden_size, rng) for _ in range(max(n_hidden_layers - 1, 0))]
    Mout = create_random_matrix(sout, hidden_size, rng)
    SM = [M0] + hidden_mats + [Mout]

    bvin = np.full((sin, 1), bias)
    bvh = np.full((hidden_size, 1), bias)
    bvout = np.full((sout, 1), bias)

    for cycle in range(cycles):
        print(f"Cycle {cycle+1}")
        for x, y in zip(inputs, targets):
            Vin0 = x + bvin
            V0_out = threshold_fire(Vin0, tau_fire)
            V_pred = process_hidden_layers(SM, bvh, bvout, V0_out, tau_fire)

            diff = y - V_pred
            error = 0.5 * np.sum(diff ** 2)
            print("Input:", x.ravel(), "Target:", y.ravel(), "Output:", V_pred.ravel(), "Error:", error)
            # Simple weight nudging toward target on output layer only (not full algorithm)
            grad = diff
            SM[-1] += learning_rate * grad @ V0_out.T

    return SM


if __name__ == "__main__":
    print("Multi-layer Perceptron (Simplified)")
    sin = int(input("Enter input size m: "))
    sout = int(input("Enter output size n: "))
    n_hidden_layers = int(input("Enter number of hidden layers: "))
    hidden_size = int(input("Enter number of perceptrons per hidden layer: "))
    cycles = int(input("Enter number of training cycles: "))

    # simple generated data: identity mapping on {0,1}^sin
    from itertools import product
    patterns = list(product([0.0, 1.0], repeat=sin))
    Seqtr = []
    for p in patterns:
        x = np.array(p[:sin]).reshape(sin, 1)
        y = x[:sout].copy()
        Seqtr.append((x, y))

    feedforward_neural_network(sin, sout, Seqtr, tau_fire=0.5, bias=0.0,
                               cycles=cycles, n_hidden_layers=n_hidden_layers,
                               hidden_size=hidden_size, learning_rate=0.1)
