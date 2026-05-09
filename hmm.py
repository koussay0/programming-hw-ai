
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
            print(f"Most probable path: {best_path} with probability {best_prob}
")


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

    print("Transition matrix:
", Mtr)
    print("Emission matrix:
", Mem)
    print("Initial probabilities:", Vin)

    hmm_path(states, emissions, Mtr, Mem, Vin)
