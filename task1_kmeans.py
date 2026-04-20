"""
Task 1: K-Means Clustering — Fast Vectorized Implementation
HW3 - DM Models 2
Dataset: kmeans_data.zip (9999 samples, 784 features, 10 classes)

Distance computations use scipy.spatial.distance.cdist and vectorized NumPy.
Generalized Jaccard uses the algebraic identity:
    min(a,b) = (a+b-|a-b|)/2,  max(a,b) = (a+b+|a-b|)/2
    => Jaccard sim = (S - L1) / (S + L1)  where S = sum(x)+sum(c), L1 = ||x-c||_1
Reference: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/jaccard.htm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
from scipy.spatial.distance import cdist
from collections import Counter
import time, os, zipfile, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
#  Vectorized distance functions: returns (n, k) matrix
# ─────────────────────────────────────────────────────────

def euclidean_distances(X, centroids):
    return cdist(X, centroids, metric='euclidean')

def cosine_distances(X, centroids):
    return cdist(X, centroids, metric='cosine')

def jaccard_distances(X, centroids):
    """Generalized Jaccard via L1 identity — O(n*k*d) via cdist."""
    Xsum = X.sum(axis=1)
    Csum = centroids.sum(axis=1)
    S  = Xsum[:, None] + Csum[None, :]           # (n, k)
    L1 = cdist(X, centroids, metric='cityblock')  # (n, k)
    return 1.0 - (S - L1) / (S + L1 + 1e-10)

DIST_FNS = {
    'euclidean': euclidean_distances,
    'cosine':    cosine_distances,
    'jaccard':   jaccard_distances,
}

# ─────────────────────────────────────────────────────────
#  K-Means
# ─────────────────────────────────────────────────────────

class KMeans:
    def __init__(self, k, metric='euclidean', max_iter=500, random_state=42):
        self.k = k; self.metric = metric
        self.dist_fn = DIST_FNS[metric]
        self.max_iter = max_iter; self.random_state = random_state
        self.centroids = None; self.labels_ = None
        self.sse_history = []; self.iterations_ = 0
        self.converge_reason_ = ''

    def _assign(self, X):
        return np.argmin(self.dist_fn(X, self.centroids), axis=1)

    def _sse(self, X, labels):
        sse = 0.0
        for ki in range(self.k):
            pts = X[labels == ki]
            if len(pts):
                sse += np.sum((pts - self.centroids[ki]) ** 2)
        return sse

    def _update_centroids(self, X, labels):
        new = np.zeros_like(self.centroids)
        for ki in range(self.k):
            pts = X[labels == ki]
            new[ki] = pts.mean(axis=0) if len(pts) else self.centroids[ki]
        return new

    def fit(self, X, stop_condition='all'):
        np.random.seed(self.random_state)
        self.centroids   = X[np.random.choice(len(X), self.k, replace=False)].astype(float).copy()
        self.sse_history = []
        prev_sse         = None

        for it in range(self.max_iter):
            labels = self._assign(X)
            sse    = self._sse(X, labels)
            self.sse_history.append(sse)

            if stop_condition in ('sse_increase', 'all'):
                if prev_sse is not None and sse > prev_sse:
                    self.labels_          = labels
                    self.iterations_      = it + 1
                    self.converge_reason_ = 'SSE increased'
                    break

            new_c = self._update_centroids(X, labels)

            if stop_condition in ('no_change', 'all'):
                shift = np.max(np.linalg.norm(new_c - self.centroids, axis=1))
                if shift < 1e-6:
                    self.centroids        = new_c
                    self.labels_          = self._assign(X)
                    self.iterations_      = it + 1
                    self.converge_reason_ = 'No centroid change'
                    break

            prev_sse = sse; self.centroids = new_c
        else:
            self.labels_          = self._assign(X)
            self.iterations_      = self.max_iter
            self.converge_reason_ = 'Max iterations reached'
        return self

    def get_final_sse(self):
        return self.sse_history[-1] if self.sse_history else None

# ─────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────

def majority_vote_labels(cluster_labels, true_labels, k):
    predicted = np.zeros_like(cluster_labels)
    for ki in range(k):
        mask = cluster_labels == ki
        if mask.sum() == 0: continue
        predicted[mask] = Counter(true_labels[mask]).most_common(1)[0][0]
    return predicted

def compute_accuracy(predicted, true_labels):
    return np.mean(predicted == true_labels)

# ─────────────────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────────────────

def load_data(zip_path='kmeans_data.zip'):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        with z.open([n for n in names if n.endswith('data.csv')][0]) as f:
            X_df = pd.read_csv(f, header=0)
        with z.open([n for n in names if n.endswith('label.csv')][0]) as f:
            y_df = pd.read_csv(f, header=0)

    X = X_df.values.astype(float)
    y = y_df.iloc[:, 0].values
    n = min(len(X), len(y)); X, y = X[:n], y[:n]
    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features, "
          f"{len(np.unique(y))} classes: {sorted(np.unique(y).tolist())}")

    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X)

    unique = np.unique(y)
    y = np.array([np.where(unique == v)[0][0] for v in y])
    return X, y

# ─────────────────────────────────────────────────────────
#  Experiments
# ─────────────────────────────────────────────────────────

def run_all(X, y):
    results = {}
    k       = len(np.unique(y))
    metrics = ['euclidean', 'cosine', 'jaccard']
    print(f"K = {k}\n")

    # ── Q1 / Q2 / Q3 ──────────────────────────────────────
    print("=== Q1, Q2, Q3: SSE, Accuracy, Convergence (stop='all', max_iter=500) ===")
    q1_sse, q2_acc, q3_iters, q3_times = {}, {}, {}, {}
    for metric in metrics:
        t0 = time.time()
        km = KMeans(k=k, metric=metric, max_iter=500).fit(X, 'all')
        elapsed = time.time() - t0
        sse  = km.get_final_sse()
        pred = majority_vote_labels(km.labels_, y, k)
        acc  = compute_accuracy(pred, y)
        q1_sse[metric] = sse; q2_acc[metric] = acc
        q3_iters[metric] = km.iterations_; q3_times[metric] = elapsed
        print(f"  {metric:10s}  SSE={sse:.2f}  Acc={acc:.4f}  "
              f"Iters={km.iterations_}  Time={elapsed:.1f}s  [{km.converge_reason_}]")

    results.update(dict(q1_sse=q1_sse, q2_acc=q2_acc,
                        q3_iters=q3_iters, q3_times=q3_times))

    # ── Q4 ────────────────────────────────────────────────
    print("\n=== Q4: SSE by termination condition (max_iter=100) ===")
    scs = ['no_change', 'sse_increase', 'max_iter']
    q4  = {sc: {} for sc in scs}
    for sc in scs:
        for metric in metrics:
            km = KMeans(k=k, metric=metric, max_iter=100).fit(X, sc)
            q4[sc][metric] = km.get_final_sse()
            print(f"  [{sc:15s}][{metric:10s}]  SSE={q4[sc][metric]:.2f}  Iters={km.iterations_}")
    results['q4'] = q4

    # ── SSE curves ────────────────────────────────────────
    print("\nCollecting SSE convergence curves...")
    sse_curves = {}
    for metric in metrics:
        km = KMeans(k=k, metric=metric, max_iter=500).fit(X, 'all')
        sse_curves[metric] = km.sse_history
    results['sse_curves'] = sse_curves

    return results, k, metrics

# ─────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────

def make_plots(results, metrics):
    os.makedirs('plots', exist_ok=True)
    C = {'euclidean': '#2196F3', 'cosine': '#FF5722', 'jaccard': '#4CAF50'}

    def bar(ax, vals, title, ylabel):
        bs = ax.bar(metrics, vals, color=[C[m] for m in metrics], edgecolor='black', width=0.5)
        for b, v in zip(bs, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,
                    f'{v:.2f}', ha='center', fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel); ax.set_xlabel('Distance Metric')

    fig, ax = plt.subplots(figsize=(7,4))
    bar(ax, [results['q1_sse'][m] for m in metrics], 'Q1: Final SSE by Distance Metric', 'SSE')
    plt.tight_layout(); plt.savefig('plots/q1_sse_comparison.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7,4))
    bar(ax, [results['q2_acc'][m] for m in metrics], 'Q2: Clustering Accuracy', 'Accuracy')
    ax.set_ylim(0, 1.1)
    plt.tight_layout(); plt.savefig('plots/q2_accuracy_comparison.png', dpi=150); plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    bar(ax1, [results['q3_iters'][m] for m in metrics], 'Q3: Iterations to Converge', 'Iterations')
    bar(ax2, [results['q3_times'][m] for m in metrics], 'Q3: Wall-Clock Time (s)', 'Time (s)')
    plt.tight_layout(); plt.savefig('plots/q3_convergence.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(9,4))
    scs = ['no_change','sse_increase','max_iter']
    sc_labels = ['No Change\nin Centroid','SSE Increases','Max Iterations (100)']
    x, w = np.arange(3), 0.25
    for i, m in enumerate(metrics):
        ax.bar(x+i*w, [results['q4'][sc][m] for sc in scs], w,
               label=m.capitalize(), color=C[m], edgecolor='black')
    ax.set_xticks(x+w); ax.set_xticklabels(sc_labels)
    ax.set_title('Q4: SSE by Metric and Termination Condition', fontweight='bold')
    ax.set_ylabel('SSE'); ax.legend()
    plt.tight_layout(); plt.savefig('plots/q4_sse_by_condition.png', dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8,4))
    for m in metrics:
        curve = results['sse_curves'][m]
        ax.plot(range(1, len(curve)+1), curve, label=m.capitalize(), color=C[m], linewidth=2)
    ax.set_title('SSE Convergence Over Iterations', fontweight='bold')
    ax.set_xlabel('Iteration'); ax.set_ylabel('SSE'); ax.legend()
    plt.tight_layout(); plt.savefig('plots/sse_convergence.png', dpi=150); plt.close()
    print("All plots saved.")

# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y = load_data('kmeans_data.zip')
    results, k, metrics = run_all(X, y)
    make_plots(results, metrics)
    print("\nTask 1 complete.")
