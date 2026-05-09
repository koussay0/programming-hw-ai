
from regression import regression_analysis
from kmeans import kmeans_clustering
from hmm import hmm_path
from mlp import feedforward_neural_network


def main():
    print("AI Programming Homework Demo")
    print("1. Linear Regression")
    print("2. K-means Clustering")
    print("3. Hidden Markov Model")
    print("4. Multi-layer Neural Network")

    choice = input("Choose an option (1-4): ").strip()

    if choice == '1':
        from regression import regression_analysis
        n = int(input("Enter number of points: "))
        pts = []
        for i in range(n):
            x_str = input(f"Point {i+1} - enter x: ")
            y_str = input(f"Point {i+1} - enter y: ")
            pts.append((float(x_str), float(y_str)))
        slope, intercept = regression_analysis(pts)
        print(f"Slope: {slope}, Intercept: {intercept}")

    elif choice == '2':
        from kmeans import kmeans_clustering
        n = int(input("Enter number of points to generate: "))
        nc = int(input("Enter number of clusters: "))
        import numpy as np
        rng = np.random.default_rng(0)
        centers = rng.uniform(-10, 10, size=(nc, 2))
        points = []
        for i in range(n):
            c = centers[i % nc]
            p = rng.normal(loc=c, scale=1.0, size=2)
            points.append(p)
        centroids, clusters, outliers = kmeans_clustering(points, nc)

    elif choice == '3':
        from hmm import hmm_path
        import numpy as np
        print("Hidden Markov Model - setup")
        n_states = int(input("Enter number of states: "))
        n_emissions = int(input("Enter number of emissions: "))
        states = [input(f"State {i} name: ") for i in range(n_states)]
        emissions = [input(f"Emission {j} symbol: ") for j in range(n_emissions)]
        print("Enter transition matrix row by row (space-separated, rows sum to 1):")
        Mtr = np.zeros((n_states, n_states), dtype=float)
        for i in range(n_states):
            row = list(map(float, input(f"Row {i} ({states[i]} -> *): ").split()))
            Mtr[i, :] = row
        print("Enter emission matrix row by row (state x emission):")
        Mem = np.zeros((n_states, n_emissions), dtype=float)
        for i in range(n_states):
            row = list(map(float, input(f"Row {i} (state {states[i]}): ").split()))
            Mem[i, :] = row
        print("Enter initial state probability vector (space-separated, length = n_states):")
        Vin = np.array(list(map(float, input().split())), dtype=float)
        hmm_path(states, emissions, Mtr, Mem, Vin)

    elif choice == '4':
        from mlp import feedforward_neural_network
        import numpy as np
        sin = int(input("Enter input size m: "))
        sout = int(input("Enter output size n: "))
        n_hidden_layers = int(input("Enter number of hidden layers: "))
        hidden_size = int(input("Enter number of perceptrons per hidden layer: "))
        cycles = int(input("Enter number of training cycles: "))
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

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
