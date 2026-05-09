
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
        print(f"  Max centroid shift: {shift}
")

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
