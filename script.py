import textwrap

files = {}

files['regression.py'] = textwrap.dedent('''
import numpy as np


def regression_analysis(points):
    """Compute slope and intercept for simple linear regression.

    points: list of (x, y)
    returns (slope, intercept)
    """
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)

    n = len(points)
    if n == 0:
        raise ValueError("No points provided")

    x_avg = xs.mean()
    y_avg = ys.mean()

    num = np.sum((xs - x_avg) * (ys - y_avg))
    den = np.sum((xs - x_avg) ** 2)
    if den == 0:
        raise ValueError("All x values are identical; cannot compute slope")

    slope = num / den
    intercept = y_avg - slope * x_avg
    return slope, intercept


if __name__ == "__main__":
    print("Linear Regression Analysis")
    n = int(input("Enter number of points: "))
    pts = []
    for i in range(n):
        x_str = input(f"Point {i+1} - enter x: ")
        y_str = input(f"Point {i+1} - enter y: ")
        pts.append((float(x_str), float(y_str)))

    slope, intercept = regression_analysis(pts)
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")

    while True:
        ans = input("Enter an x value to predict y (or 'exit'): ").strip()
        if ans.lower() == 'exit':
            break
        x_val = float(ans)
        y_pred = slope * x_val + intercept
        print(f"Predicted y: {y_pred}")
''')

files['kmeans.py'] = textwrap.dedent('''
import numpy as np


def generate_seed_points(points, nc, random_state=None):
    rng = np.random.default_rng(random_state)
    pts = np.array(points, dtype=float)
    xs, ys = pts[:, 0], pts[:, 1]

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()

    # divide into nc x nc macroblocks
    sizex = (maxx - minx) / nc
    sizey = (maxy - miny) / nc

    densities = []
    macro_centers = []
    for i in range(nc):
        xlow = minx + i * sizex
        xhigh = xlow + sizex
        xmid = 0.5 * (xlow + xhigh)
        for j in range(nc):
            ylow = miny + j * sizey
            yhigh = ylow + sizey
            ymid = 0.5 * (ylow + yhigh)
            mask = (
                (pts[:, 0] >= xlow)
                & (pts[:, 0] < xhigh)
                & (pts[:, 1] >= ylow)
                & (pts[:, 1] < yhigh)
            )
            count = np.sum(mask)
            densities.append(count)
            macro_centers.append((xmid, ymid))

    densities = np.array(densities)
    macro_centers = np.array(macro_centers)

    # pick nc macroblocks with highest density
    if len(densities) < nc:
        raise ValueError("Not enough macroblocks to choose seeds from")
    idx_sorted = np.argsort(-densities)
    chosen_idxs = idx_sorted[:nc]
    seeds = macro_centers[chosen_idxs]

    # compute radius as half of min pairwise distance between seeds
    min_dist2 = np.inf
    for i in range(nc):
        for j in range(i + 1, nc):
            dx = seeds[i, 0] - seeds[j, 0]
            dy = seeds[i, 1] - seeds[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < min_dist2:
                min_dist2 = d2
    radius = 0.5 * np.sqrt(min_dist2) if min_dist2 < np.inf else 1.0

    return seeds, radius


def kmeans_clustering(points, nc, max_shift=1e-3, max_loops=100, random_state=None):
    rng = np.random.default_rng(random_state)
    pts = np.array(points, dtype=float)
    n = pts.shape[0]

    centroids, radius = generate_seed_points(pts, nc, random_state)

    for loop in range(1, max_loops + 1):
        clusters = [[] for _ in range(nc)]
        outliers = []

        # assign points
        for p in pts:
            dists = np.linalg.norm(centroids - p, axis=1)
            k = int(np.argmin(dists))
            if dists[k] <= radius:
                clusters[k].append(p)
            else:
                outliers.append(p)

        new_centroids = centroids.copy()
        for i in range(nc):
            if clusters[i]:
                arr = np.array(clusters[i])
                new_centroids[i] = arr.mean(axis=0)

        # compute shift
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()

        print(f"Iteration {loop}")
        for i in range(nc):
            print(f"  Cluster {i}: centroid={new_centroids[i]}, points={len(clusters[i])}")
        print(f"  Outliers: {len(outliers)}")
        print(f"  Max centroid shift: {shift}\n")

        centroids = new_centroids
        if shift <= max_shift:
            break

    return centroids, clusters, outliers


if __name__ == "__main__":
    print("K-means Clustering")
    n = int(input("Enter number of points to generate: "))
    nc = int(input("Enter number of clusters: "))

    # generate synthetic data: nc Gaussian blobs
    rng = np.random.default_rng(0)
    centers = rng.uniform(-10, 10, size=(nc, 2))
    points = []
    for i in range(n):
        c = centers[i % nc]
        p = rng.normal(loc=c, scale=1.0, size=2)
        points.append(p)

    centroids, clusters, outliers = kmeans_clustering(points, nc)
    print("Final centroids:")
    for i, c in enumerate(centroids):
        print(f"Cluster {i}: {c}")
    print(f"Outliers: {len(outliers)}")
''')

files['hmm.py'] = textwrap.dedent('''
import numpy as np


def valid_emission(seq_em, emissions):
    return all(e in emissions for e in seq_em)


def emission_set(e, Mem, emissions, eps=1e-6):
    j = emissions.index(e)
    n_states = Mem.shape[0]
    S = []
    for i in range(n_states):
        if abs(Mem[i, j]) >= eps:
            S.append(i)
    return S


def valid_transition(state_indices, Mtr, Vin, eps=1e-6):
    if not state_indices:
        return False
    if abs(Vin[state_indices[0]]) < eps:
        return False
    for i in range(len(state_indices) - 1):
        s_from = state_indices[i]
        s_to = state_indices[i + 1]
        if abs(Mtr[s_from, s_to]) < eps:
            return False
    return True


def calculate_probability(state_indices, seq_em, states, emissions, Mtr, Mem, Vin):
    n = len(state_indices)
    acc = 1.0
    # initial factor
    row_em = state_indices[0]
    col_em = emissions.index(seq_em[0])
    acc *= Vin[row_em] * Mem[row_em, col_em]

    for m in range(1, n):
        i = state_indices[m - 1]
        j = state_indices[m]
        row_tr = i
        col_tr = j
        row_em = j
        col_em = emissions.index(seq_em[m])
        ptr = Mtr[row_tr, col_tr]
        pem = Mem[row_em, col_em]
        acc *= ptr * pem
    return acc


def hmm_path(states, emissions, Mtr, Mem, Vin):
    states = list(states)
    emissions = list(emissions)

    while True:
        seq_str = input("Enter emission sequence separated by spaces (or 'exit'): ").strip()
        if seq_str.lower() == 'exit':
            break
        seq_em = seq_str.split()
        if not valid_emission(seq_em, emissions):
            print("Invalid emission in sequence. Try again.")
            continue

        # build candidate state index sets using emission_set
        possible_states_per_pos = []
        for e in seq_em:
            S = emission_set(e, Mem, emissions)
            possible_states_per_pos.append(S)

        # Cartesian product over positions
        from itertools import product
        candidates = []
        for tup in product(*possible_states_per_pos):
            candidates.append(list(tup))

        best_prob = -1.0
        best_path = None

        print("All valid paths and probabilities:")
        for path in candidates:
            if not valid_transition(path, Mtr, Vin):
                continue
            p = calculate_probability(path, seq_em, states, emissions, Mtr, Mem, Vin)
            state_names = [states[i] for i in path]
            print(f"  Path {state_names}: probability={p}")
            if p > best_prob:
                best_prob = p
                best_path = state_names

        if best_path is None:
            print("No valid path found for this sequence.")
        else:
            print(f"Most probable path: {best_path} with probability {best_prob}\n")


if __name__ == "__main__":
    print("Hidden Markov Model - Most Probable Path")
    n_states = int(input("Enter number of states: "))
    n_emissions = int(input("Enter number of emissions: "))

    states = [input(f"State {i} name: ") for i in range(n_states)]
    emissions = [input(f"Emission {j} symbol: ") for j in range(n_emissions)]

    print("Enter transition matrix row by row (space-separated, rows sum to 1):")
    Mtr = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        row = list(map(float, input(f"Row {i} ({states[i]} -> *): ").split()))
        if len(row) != n_states:
            raise ValueError("Row length mismatch")
        Mtr[i, :] = row

    print("Enter emission matrix row by row (state x emission):")
    Mem = np.zeros((n_states, n_emissions), dtype=float)
    for i in range(n_states):
        row = list(map(float, input(f"Row {i} (state {states[i]}): ").split()))
        if len(row) != n_emissions:
            raise ValueError("Row length mismatch")
        Mem[i, :] = row

    print("Enter initial state probability vector (space-separated, length = n_states):")
    Vin = np.array(list(map(float, input().split())), dtype=float)
    if Vin.shape[0] != n_states:
        raise ValueError("Initial vector length mismatch")

    print("Transition matrix:\n", Mtr)
    print("Emission matrix:\n", Mem)
    print("Initial probabilities:", Vin)

    hmm_path(states, emissions, Mtr, Mem, Vin)
''')

files['mlp.py'] = textwrap.dedent('''
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
''')

files['main.py'] = textwrap.dedent('''
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
''')

created = []
for name, content in files.items():
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)
    created.append(name)

created