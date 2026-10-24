import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


def assign_bins(values, boundaries):
    """
    Assigns continuous values into discrete intervals following Equation (1):
    First interval is closed [d_0, d_1], subsequent intervals are half-open (d_{r-1}, d_r].
    """
    n_bins = len(boundaries) - 1
    bins = np.zeros(len(values), dtype=int)
    for r in range(n_bins):
        low, high = boundaries[r], boundaries[r + 1]
        if r == 0:
            mask = (values >= low) & (values <= high)
        else:
            mask = (values > low) & (values <= high)
        bins[mask] = r
    # Handle out-of-range test values gracefully
    bins[values < boundaries[0]] = 0
    bins[values > boundaries[-1]] = n_bins - 1
    return bins


def build_quanta_matrix(binned_values, y, n_bins, n_classes):
    """Constructs the 2D quanta matrix (Table 1) of shape (n_classes, n_bins)."""
    quanta = np.zeros((n_classes, n_bins), dtype=float)
    for i in range(n_classes):
        for r in range(n_bins):
            quanta[i, r] = np.sum((y == i) & (binned_values == r))
    return quanta


def compute_caim_score(quanta):
    """Calculates the CAIM criterion (Equation 10) with overflow-safe division."""
    n_bins = quanta.shape[1]
    col_totals = quanta.sum(axis=0)
    col_maxes = quanta.max(axis=0)

    total = 0.0
    for r in range(n_bins):
        if col_totals[r] > 0:
            # Overflow prevention: divide max_r by M_{+r} first, then multiply by max_r
            total += (col_maxes[r] / col_totals[r]) * col_maxes[r]
    return total / n_bins


def compute_cair_score(quanta):
    """Calculates the CAIR criterion (Equations 5, 7, and 8)."""
    M = quanta.sum()
    if M == 0:
        return 0.0
    p_ir = quanta / M
    p_i_plus = p_ir.sum(axis=1, keepdims=True)
    p_plus_r = p_ir.sum(axis=0, keepdims=True)

    mutual_info = 0.0
    shannon_entropy = 0.0
    for i in range(quanta.shape[0]):
        for r in range(quanta.shape[1]):
            if p_ir[i, r] > 0:
                denom = p_i_plus[i, 0] * p_plus_r[0, r]
                if denom > 0:
                    mutual_info += p_ir[i, r] * np.log2(p_ir[i, r] / denom)
                shannon_entropy += p_ir[i, r] * np.log2(1.0 / p_ir[i, r])

    return mutual_info / shannon_entropy if shannon_entropy > 0 else 0.0


def fit_caim_feature(feature, y, n_classes):
    """Runs the greedy top-down CAIM discretization algorithm on one feature."""
    unique_vals = np.unique(feature)
    candidate_midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0

    boundaries = [float(np.min(feature)), float(np.max(feature))]
    global_caim = 0.0
    k = 1

    while True:
        best_caim = -1.0
        best_boundary = None

        for b in candidate_midpoints:
            if b in boundaries:
                continue
            trial_bounds = sorted(boundaries + [float(b)])
            binned = assign_bins(feature, trial_bounds)
            quanta = build_quanta_matrix(binned, y, len(trial_bounds) - 1, n_classes)
            score = compute_caim_score(quanta)

            if score > best_caim:
                best_caim = score
                best_boundary = float(b)

        if best_boundary is not None and (best_caim > global_caim or k < n_classes):
            boundaries = sorted(boundaries + [best_boundary])
            global_caim = best_caim
            k += 1
        else:
            break

    return boundaries, global_caim


def fit_equal_width(feature, n_bins):
    """Unsupervised Equal-Width discretization."""
    return list(np.linspace(np.min(feature), np.max(feature), n_bins + 1))


def fit_equal_frequency(feature, n_bins):
    """Unsupervised Equal-Frequency discretization."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bounds = np.percentile(feature, quantiles)
    return list(np.unique(bounds))


def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    n_classes = len(np.unique(y))

    print("=== 1. FULL DATASET CAIM BOUNDARIES (150 EXAMPLES) ===")
    full_caims, full_cairs = [], []
    for j, name in enumerate(feature_names):
        bounds, caim_val = fit_caim_feature(X[:, j], y, n_classes)
        binned = assign_bins(X[:, j], bounds)
        quanta = build_quanta_matrix(binned, y, len(bounds) - 1, n_classes)
        cair_val = compute_cair_score(quanta)
        full_caims.append(caim_val)
        full_cairs.append(cair_val)
        print(
            f"{name:20s} | Intervals: {len(bounds)-1} | "
            f"Bounds: {[round(b, 2) for b in bounds]} | "
            f"CAIM: {caim_val:.2f} | CAIR: {cair_val:.3f}"
        )
    print(f"Mean Full-Dataset CAIM: {np.mean(full_caims):.2f} | Mean CAIR: {np.mean(full_cairs):.3f}\n")

    print("=== 2. 10-FOLD CROSS-VALIDATION BENCHMARK ===")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    methods = ["CAIM", "Equal-Width (4 bins)", "Equal-Frequency (4 bins)", "Built-in (Raw)"]

    metrics = {m: {"cair": [], "caim": [], "intervals": [], "time": [], "acc": [], "leaves": []} for m in methods}

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for m in methods:
            if m == "Built-in (Raw)":
                clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, ccp_alpha=0.015, random_state=42)
                clf.fit(X_train, y_train)
                metrics[m]["acc"].append(clf.score(X_test, y_test) * 100.0)
                metrics[m]["leaves"].append(clf.get_n_leaves())
                continue

            t0 = time.perf_counter()
            X_train_disc = np.zeros_like(X_train)
            X_test_disc = np.zeros_like(X_test)
            fold_cairs, fold_caims = [], []
            total_intervals = 0

            for j in range(X.shape[1]):
                if m == "CAIM":
                    bounds, _ = fit_caim_feature(X_train[:, j], y_train, n_classes)
                elif m == "Equal-Width (4 bins)":
                    bounds = fit_equal_width(X_train[:, j], 4)
                else:
                    bounds = fit_equal_frequency(X_train[:, j], 4)

                X_train_disc[:, j] = assign_bins(X_train[:, j], bounds)
                X_test_disc[:, j] = assign_bins(X_test[:, j], bounds)

                n_int = len(bounds) - 1
                total_intervals += n_int
                q_mat = build_quanta_matrix(X_train_disc[:, j], y_train, n_int, n_classes)
                fold_cairs.append(compute_cair_score(q_mat))
                fold_caims.append(compute_caim_score(q_mat))

            elapsed = time.perf_counter() - t0
            metrics[m]["cair"].append(np.mean(fold_cairs))
            metrics[m]["caim"].append(np.mean(fold_caims))
            metrics[m]["intervals"].append(total_intervals)
            metrics[m]["time"].append(elapsed)

            clf = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2, ccp_alpha=0.015, random_state=42)
            clf.fit(X_train_disc, y_train)
            metrics[m]["acc"].append(clf.score(X_test_disc, y_test) * 100.0)
            metrics[m]["leaves"].append(clf.get_n_leaves())

    for m in methods:
        if m != "Built-in (Raw)":
            print(
                f"{m:25s} | CAIR: {np.mean(metrics[m]['cair']):.3f} (±{np.std(metrics[m]['cair']):.3f}) | "
                f"Intervals: {np.mean(metrics[m]['intervals']):.1f} | "
                f"Time: {np.mean(metrics[m]['time']):.4f}s | "
                f"DT Acc: {np.mean(metrics[m]['acc']):.2f}% (±{np.std(metrics[m]['acc']):.2f}) | "
                f"Leaves: {np.mean(metrics[m]['leaves']):.1f}"
            )
        else:
            print(
                f"{m:25s} | Raw Continuous Features | "
                f"DT Acc: {np.mean(metrics[m]['acc']):.2f}% (±{np.std(metrics[m]['acc']):.2f}) | "
                f"Leaves: {np.mean(metrics[m]['leaves']):.1f}"
            )


if __name__ == "__main__":
    main()