
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
