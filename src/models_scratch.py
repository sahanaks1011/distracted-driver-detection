"""Classifiers implemented from scratch in NumPy (no ML library).

Owner: Sahana K S
  LinearSVM     multiclass hinge loss + L2           (paper eq. 1-3)
  Softmax       cross-entropy loss + L2              (paper eq. 4)
  TwoLayerNet   affine -> ReLU -> affine -> softmax  (paper section 4.5, 100 hidden units)
All are trained with mini-batch SGD (momentum for the net).
"""
import numpy as np


# ---------------------------------------------------------------- losses
def svm_loss(W, b, X, y, reg):
    """Vectorised multiclass SVM loss and gradients."""
    n = X.shape[0]
    scores = X @ W + b
    correct = scores[np.arange(n), y][:, None]
    margins = np.maximum(0, scores - correct + 1.0)
    margins[np.arange(n), y] = 0
    loss = margins.sum() / n + reg * np.sum(W * W)

    mask = (margins > 0).astype(X.dtype)
    mask[np.arange(n), y] = -mask.sum(1)
    dW = X.T @ mask / n + 2 * reg * W
    db = mask.sum(0) / n
    return loss, dW, db


def softmax_loss(W, b, X, y, reg):
    """Vectorised softmax cross-entropy loss and gradients."""
    n = X.shape[0]
    scores = X @ W + b
    scores -= scores.max(1, keepdims=True)          # numerical stability
    p = np.exp(scores)
    p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean() + reg * np.sum(W * W)
    p[np.arange(n), y] -= 1
    dW = X.T @ p / n + 2 * reg * W
    db = p.sum(0) / n
    return loss, dW, db


# ---------------------------------------------------------------- linear models
class LinearClassifier:
    loss_fn = None

    def __init__(self, num_classes=10, seed=42):
        self.C = num_classes
        self.seed = seed
        self.W = None
        self.b = None

    def train(self, X, y, lr=1e-3, reg=1e-3, num_iters=1500, batch_size=256,
              verbose=False):
        rng = np.random.default_rng(self.seed)
        D = X.shape[1]
        # Small fixed-seed initialisation: the paper found SVM very sensitive to init.
        self.W = (0.001 * rng.standard_normal((D, self.C))).astype(np.float32)
        self.b = np.zeros(self.C, dtype=np.float32)
        history = []
        for it in range(num_iters):
            idx = rng.integers(0, X.shape[0], batch_size)
            loss, dW, db = type(self).loss_fn(self.W, self.b, X[idx], y[idx], reg)
            self.W -= lr * dW
            self.b -= lr * db
            history.append(float(loss))
            if verbose and it % 250 == 0:
                print(f"    iter {it:5d}/{num_iters}  loss {loss:.4f}")
        return history

    def scores(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        return np.argmax(self.scores(X), axis=1)

    def save(self, path, mean, std):
        np.savez_compressed(path, kind=type(self).__name__, W=self.W, b=self.b,
                            mean=mean, std=std)


class LinearSVM(LinearClassifier):
    loss_fn = staticmethod(svm_loss)


class Softmax(LinearClassifier):
    loss_fn = staticmethod(softmax_loss)

    def predict_proba(self, X):
        s = self.scores(X)
        s -= s.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- two-layer net
class TwoLayerNet:
    """input (D) -> 100 ReLU units -> 10 softmax outputs."""

    def __init__(self, input_dim, hidden=100, num_classes=10, seed=42):
        rng = np.random.default_rng(seed)
        # He / Kaiming initialisation (the paper lists this as future work)
        self.p = {
            "W1": (rng.standard_normal((input_dim, hidden)) * np.sqrt(2.0 / input_dim)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": (rng.standard_normal((hidden, num_classes)) * np.sqrt(2.0 / hidden)).astype(np.float32),
            "b2": np.zeros(num_classes, dtype=np.float32),
        }
        self.seed = seed

    def forward(self, X):
        h = np.maximum(0, X @ self.p["W1"] + self.p["b1"])
        return h, h @ self.p["W2"] + self.p["b2"]

    def loss(self, X, y, reg):
        n = X.shape[0]
        h, scores = self.forward(X)
        scores = scores - scores.max(1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(1, keepdims=True)
        loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        loss += reg * (np.sum(self.p["W1"] ** 2) + np.sum(self.p["W2"] ** 2))

        d = probs
        d[np.arange(n), y] -= 1
        d /= n
        g = {"W2": h.T @ d + 2 * reg * self.p["W2"], "b2": d.sum(0)}
        dh = d @ self.p["W2"].T
        dh[h <= 0] = 0
        g["W1"] = X.T @ dh + 2 * reg * self.p["W1"]
        g["b1"] = dh.sum(0)
        return loss, g

    def train(self, X, y, X_val=None, y_val=None, lr=1e-3, reg=1e-3, num_iters=3000,
              batch_size=200, momentum=0.9, lr_decay=0.95, verbose=False):
        rng = np.random.default_rng(self.seed)
        vel = {k: np.zeros_like(v) for k, v in self.p.items()}
        iters_per_epoch = max(X.shape[0] // batch_size, 1)
        hist = {"loss": [], "train_acc": [], "val_acc": []}
        for it in range(num_iters):
            idx = rng.integers(0, X.shape[0], batch_size)
            loss, g = self.loss(X[idx], y[idx], reg)
            for k in self.p:
                vel[k] = momentum * vel[k] - lr * g[k]
                self.p[k] += vel[k]
            hist["loss"].append(float(loss))
            if (it + 1) % iters_per_epoch == 0:
                lr *= lr_decay
                sub = rng.choice(X.shape[0], min(2000, X.shape[0]), replace=False)
                hist["train_acc"].append(float((self.predict(X[sub]) == y[sub]).mean()))
                if X_val is not None:
                    hist["val_acc"].append(float((self.predict(X_val) == y_val).mean()))
                if verbose:
                    va = f"  val {hist['val_acc'][-1]:.3f}" if X_val is not None else ""
                    print(f"    iter {it + 1:5d}  loss {loss:.4f}  train {hist['train_acc'][-1]:.3f}{va}")
        return hist

    def predict_proba(self, X):
        _, s = self.forward(X)
        s = s - s.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.forward(X)[1], axis=1)

    def save(self, path, mean, std):
        np.savez_compressed(path, kind="TwoLayerNet", mean=mean, std=std, **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(d["W1"].shape[0], d["W1"].shape[1], d["W2"].shape[1])
        for k in ("W1", "b1", "W2", "b2"):
            net.p[k] = d[k]
        return net, d["mean"], d["std"]


def gradient_check(seed=0):
    """Numerical gradient check used to verify the from-scratch code (run in tests)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((5, 7))
    y = rng.integers(0, 3, 5)
    W = rng.standard_normal((7, 3)) * 0.1
    b = np.zeros(3)
    worst = 0.0
    for fn in (svm_loss, softmax_loss):
        _, dW, _ = fn(W, b, X, y, 0.1)
        num = np.zeros_like(W)
        h = 1e-5
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                Wp, Wm = W.copy(), W.copy()
                Wp[i, j] += h
                Wm[i, j] -= h
                num[i, j] = (fn(Wp, b, X, y, 0.1)[0] - fn(Wm, b, X, y, 0.1)[0]) / (2 * h)
        worst = max(worst, np.max(np.abs(num - dW)))
    net = TwoLayerNet(7, 4, 3, seed=seed)
    net.p = {k: v.astype(np.float64) for k, v in net.p.items()}
    _, g = net.loss(X, y, 0.05)
    for k in ("W1", "W2"):
        num = np.zeros_like(net.p[k])
        it = np.nditer(num, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            old = net.p[k][ix]
            net.p[k][ix] = old + 1e-5
            lp = net.loss(X, y, 0.05)[0]
            net.p[k][ix] = old - 1e-5
            lm = net.loss(X, y, 0.05)[0]
            net.p[k][ix] = old
            num[ix] = (lp - lm) / 2e-5
        worst = max(worst, np.max(np.abs(num - g[k])))
    return worst


if __name__ == "__main__":
    print("max |numerical - analytic| gradient error:", gradient_check())
