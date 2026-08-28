import time
import numpy as np
from evaluate import evaluate
from data import load, encode, FIELDS

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class RNN_BPR:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        
        # Model Parameters
        self.Emb = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W_ih = rng.normal(0, 0.1, (k, k)).astype(np.float32)
        self.W_hh = rng.normal(0, 0.1, (k, k)).astype(np.float32)
        self.b_h = np.zeros(k, dtype=np.float32)
        self.W_ho = rng.normal(0, 0.1, (k, 1)).astype(np.float32)
        self.b_o = np.zeros(1, dtype=np.float32)

        self.lr, self.l2 = lr, l2
        self.t = 0
        
        self.params = [self.Emb, self.W_ih, self.W_hh, self.b_h, self.W_ho, self.b_o]
        self.M = [np.zeros_like(p) for p in self.params]
        self.V = [np.zeros_like(p) for p in self.params]

    def logits(self, X):
        B, F = X.shape
        E = self.Emb[X]  # (B, F, k)
        H = np.zeros((B, F + 1, self.W_ih.shape[1]), dtype=np.float32)
        
        for step in range(F):
            H[:, step+1, :] = np.tanh(E[:, step, :] @ self.W_ih + H[:, step, :] @ self.W_hh + self.b_h)
            
        z = H[:, F, :] @ self.W_ho + self.b_o
        return z.squeeze(1), E, H

    def _backward(self, X, E, H, g_logits):
        """Helper to compute gradients for a given input sequence and upstream logit gradients."""
        B, F = X.shape
        
        g_W_ho = H[:, F, :].T @ g_logits[:, None]
        g_b_o = np.array([np.sum(g_logits)], dtype=np.float32)
        g_H_curr = g_logits[:, None] @ self.W_ho.T

        g_W_hh = np.zeros_like(self.W_hh)
        g_W_ih = np.zeros_like(self.W_ih)
        g_b_h = np.zeros_like(self.b_h)
        g_E = np.zeros_like(E)

        for step in reversed(range(F)):
            d_preact = g_H_curr * (1 - H[:, step+1, :]**2)
            g_b_h += np.sum(d_preact, axis=0)
            g_W_ih += E[:, step, :].T @ d_preact
            g_W_hh += H[:, step, :].T @ d_preact
            g_E[:, step, :] = d_preact @ self.W_ih.T
            g_H_curr = d_preact @ self.W_hh.T

        g_Emb = np.zeros_like(self.Emb)
        for step in range(F):
            np.add.at(g_Emb, X[:, step], g_E[:, step, :])

        return [g_Emb, g_W_ih, g_W_hh, g_b_h, g_W_ho, g_b_o]

    def step_pairwise(self, X_pos, X_neg):
        """Pairwise BPR Training Step: Optimizes P(pos > neg)"""
        B = X_pos.shape[0]
        
        # 1. Forward pass for both positive and negative items
        z_pos, E_pos, H_pos = self.logits(X_pos)
        z_neg, E_neg, H_neg = self.logits(X_neg)
        
        # 2. BPR Loss calculation
        prob = sigmoid(z_pos - z_neg)
        loss = float(-np.mean(np.log(prob + 1e-9)))

        # 3. Gradients w.r.t the logits
        g_z_pos = ((prob - 1) / B).astype(np.float32)
        g_z_neg = ((1 - prob) / B).astype(np.float32)

        # 4. Backpropagate both paths
        grads_pos = self._backward(X_pos, E_pos, H_pos, g_z_pos)
        grads_neg = self._backward(X_neg, E_neg, H_neg, g_z_neg)

        # 5. Combine gradients & Adam Update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.params)):
            P, M, Vv = self.params[i], self.M[i], self.V[i]
            
            # Combine pos and neg gradients
            G = grads_pos[i] + grads_neg[i] + (self.l2 * P) 
            
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            
            m_hat = M / (1 - b1 ** self.t)
            v_hat = Vv / (1 - b2 ** self.t)
            
            P -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        return loss

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


class DeepFM:
    """FM (shared embeddings, 2nd-order interaction + linear term) plus a
    plain MLP ("deep" component) over the same flattened embeddings, summed
    before the sigmoid -- the standard DeepFM combination. Manual forward /
    backward, no autograd, matching the rest of this file."""
    def __init__(self, dim, k=16, hidden=(64, 32), lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        n_fields = len(FIELDS)
        h1, h2 = hidden
        in_dim = n_fields * k

        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W_lin = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.W1 = (rng.normal(0, 1, (in_dim, h1)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = (rng.normal(0, 1, (h1, h2)) * np.sqrt(2.0 / h1)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.W3 = (rng.normal(0, 1, (h2, 1)) * np.sqrt(2.0 / h2)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

        self.lr, self.l2 = lr, l2
        self.t = 0
        self.params = [self.V, self.W_lin, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.M = [np.zeros_like(p) for p in self.params]
        self.Vv = [np.zeros_like(p) for p in self.params]

    def forward(self, X):
        B, F = X.shape
        E = self.V[X]                                        # (B,F,k)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_lin = self.W_lin[X].sum(1)

        flat = E.reshape(B, F * E.shape[2])
        h1_pre = flat @ self.W1 + self.b1
        h1 = np.maximum(h1_pre, 0)
        h2_pre = h1 @ self.W2 + self.b2
        h2 = np.maximum(h2_pre, 0)
        deep_out = (h2 @ self.W3 + self.b3).reshape(-1)

        z = self.b + fm_lin + fm_inter + deep_out
        return z, (E, S, flat, h1_pre, h1, h2_pre, h2)

    def step(self, X, y):
        B = len(y)
        z, (E, S, flat, h1_pre, h1, h2_pre, h2) = self.forward(X)
        p = sigmoid(z)
        g_z = ((p - y) / B).astype(np.float32)

        # deep component backward
        g_W3 = h2.T @ g_z[:, None]
        g_b3 = np.array([g_z.sum()], dtype=np.float32)
        g_h2_pre = (g_z[:, None] @ self.W3.T) * (h2_pre > 0)
        g_W2 = h1.T @ g_h2_pre
        g_b2 = g_h2_pre.sum(0)
        g_h1_pre = (g_h2_pre @ self.W2.T) * (h1_pre > 0)
        g_W1 = flat.T @ g_h1_pre
        g_b1 = g_h1_pre.sum(0)
        g_E_deep = (g_h1_pre @ self.W1.T).reshape(E.shape)

        # fm component backward (same identity as baseline.FM.step)
        g_W_lin = np.zeros_like(self.W_lin)
        np.add.at(g_W_lin, X, g_z[:, None])
        g_E_fm = g_z[:, None, None] * (S[:, None, :] - E)

        g_V = np.zeros_like(self.V)
        np.add.at(g_V, X, g_E_fm + g_E_deep)
        g_V += self.l2 * self.V
        g_W_lin += self.l2 * self.W_lin

        grads = [g_V, g_W_lin, g_W1, g_b1, g_W2, g_b2, g_W3, g_b3]
        self.t += 1
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in zip(self.params, grads, self.M, self.Vv):
            M *= b1c; M += (1 - b1c) * G
            Vv *= b2c; Vv += (1 - b2c) * (G * G)
            P -= self.lr * (M / (1 - b1c ** self.t)) / (np.sqrt(Vv / (1 - b2c ** self.t)) + eps)
        self.b -= self.lr * g_z.sum()

        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_deepfm(splits, k=16, hidden=(64, 32), lr=0.001, epochs=30, bs=8192, patience=4,
               seed=0, verbose=True, enc=None, dim=None):
    """DeepFM training, mirroring run_fm/run_bpr's shape: reuses a precomputed
    (enc, dim) when given, early-stops on valid primary, returns {'valid': ..., 'test': ...}."""
    if enc is None or dim is None:
        enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    m = DeepFM(dim, k=k, hidden=hidden, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, best_b, bad = -1, None, None, 0

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = [p.copy() for p in m.params]
            best_b = np.float32(m.b)
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        for p, b in zip(m.params, best_state):
            p[...] = b
        m.b = best_b

    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


def run_bpr(splits, k=16, lr=0.001, epochs=10, bs=8192, patience=3, seed=0, verbose=True, enc=None, dim=None):
    """Pairwise BPR training, mirroring baseline.run_fm's shape: reuses a
    precomputed (enc, dim) when given, early-stops on valid primary, and
    returns {'valid': ..., 'test': ...}."""
    if enc is None or dim is None:
        enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    # --- IN-USER PAIR GENERATION (Crucial for BPR) ---
    user_pos, user_neg = {}, {}
    for i, (ui, yi) in enumerate(zip(utr, ytr)):
        if yi == 1: user_pos.setdefault(ui, []).append(i)
        else:       user_neg.setdefault(ui, []).append(i)
    valid_users = [ui for ui in user_pos if ui in user_neg]
    if verbose:
        print(f"Found {len(valid_users)} users with both positive and negative interactions.")

    m = RNN_BPR(dim=dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()

        # Dynamically sample negatives for each positive item, within the same user
        X_pos_list, X_neg_list = [], []
        for ui in valid_users:
            for p_idx in user_pos[ui]:
                n_idx = rng.choice(user_neg[ui])  # Randomly sample one negative
                X_pos_list.append(Xtr[p_idx])
                X_neg_list.append(Xtr[n_idx])

        X_pos_arr = np.array(X_pos_list)
        X_neg_arr = np.array(X_neg_list)

        # Shuffle the paired dataset
        idx = rng.permutation(len(X_pos_arr))
        X_pos_arr = X_pos_arr[idx]
        X_neg_arr = X_neg_arr[idx]

        # Train over batches
        losses = [m.step_pairwise(X_pos_arr[i:i + bs], X_neg_arr[i:i + bs])
                  for i in range(0, len(idx), bs)]

        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = [p.copy() for p in m.params]
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break

    if best_state is not None:
        for p, b in zip(m.params, best_state):
            p[...] = b

    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == "__main__":
    print("Loading data...")
    splits = load('./KuaiRand-Pure/data')
    results = run_bpr(splits, k=16, lr=0.001, epochs=10)

    print("\n--- Final Test Set Results ---")
    r = results['test']
    print(f"GAUC:          {r['GAUC']:.4f}")
    print(f"nDCG@5:        {r['nDCG@5']:.4f}")
    print(f"Primary Score: {r['primary']:.4f}")