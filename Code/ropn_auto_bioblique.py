"""
Geometrie DOUBLEMENT oblique : projecteur de Kronecker P = P_R (x) P_L pour des entrees
matricielles X (canaux x temps), avec un projecteur oblique ferme a gauche (espace des
canaux, W_L = S_L^{-1} U_L) et un a droite (espace temporel, W_R = S_R^{-1} U_R).

    code  C = W_L^T X W_R   (r_L x r_R),   P vec(X) = vec(U_L C U_R^T)
    (P_R (x) P_L)^2 = P_R^2 (x) P_L^2 = P_R (x) P_L   -> idempotent, rang r_L r_R,
    det(I + P/L) = (1+1/L)^{r_L r_R}.

Ajustement ferme par moindres carres alternes (2-3 passes), chaque passe = deux blocs fast
(GEVD C x C) sur c et t dimensions au lieu d'une sur c*t : O(n(c^3 + t^3)) vs O(n (ct)^2 + (ct)^3),
et surtout O(c^2 + t^2) statistiques a estimer au lieu de O(c^2 t^2).
Deuxieme variante (KroneckerMetricBlock) : la METRIQUE est de Kronecker, la representation reste pleine.
Comparaison : bloc vectorise (d = c t, forme duale), les deux variantes de Kronecker et un MLP.
"""
import time
import numpy as np
from numpy.linalg import qr, cholesky
from ropn_auto_fast import fit_block_fast


def make_matrix_data(C=3, c=16, t=32, n_per=20, kappa_s=30.0, kappa_t=30.0, sep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def cov(k, d):
        eig = np.logspace(0, np.log10(k), d); eig /= eig.mean()
        Q, _ = qr(rng.standard_normal((d, d))); return Q @ np.diag(eig) @ Q.T
    Sig_s, Sig_t = cov(kappa_s, c), cov(kappa_t, t)
    Ls, Lt = cholesky(Sig_s), cholesky(Sig_t)
    # motifs de classe rang-2 spatio-temporels (moyennes matricielles)
    Ms = [sep * rng.standard_normal((c, 2)) @ rng.standard_normal((2, t)) / np.sqrt(2) for _ in range(C)]
    X = np.zeros((C * n_per, c, t)); y = np.zeros(C * n_per, dtype=int)
    for k in range(C):
        for i in range(n_per):
            X[k * n_per + i] = Ms[k] + Ls @ rng.standard_normal((c, t)) @ Lt.T   # bruit Sig_s (x) Sig_t
            y[k * n_per + i] = k
    return X, y


class BiObliqueBlock:
    """Bloc doublement oblique ferme (alternance gauche/droite)."""
    def __init__(self, n_iter=3): self.n_iter = n_iter

    def fit(self, X, y):
        n, c, t = X.shape
        Cn = len(np.unique(y))
        # init droite : bloc fast sur les moyennes temporelles (t-dim) de chaque essai
        self.W_R = np.eye(t)[:, :min(t, Cn - 1)]          # depart : premieres colonnes canoniques
        for _ in range(self.n_iter):
            # gauche : echantillons = colonnes de X_i W_R (c-dim), etiquette repetee
            XR = np.einsum("ncT,Tk->nck", X, self.W_R)      # (n, c, r_R)
            Zl = XR.transpose(0, 2, 1).reshape(-1, c); yl = np.repeat(y, XR.shape[2])
            self.U_L, self.W_L, _, _ = fit_block_fast(Zl, yl, shrink=True, rank="C1")
            # droite : echantillons = lignes de W_L^T X_i (t-dim)
            LX = np.einsum("ck,nct->nkt", self.W_L, X)      # (n, r_L, t)
            Zr = LX.reshape(-1, t); yr = np.repeat(y, LX.shape[1])
            self.U_R, self.W_R, _, _ = fit_block_fast(Zr, yr, shrink=True, rank="C1")
        F = self.code(X)
        from ropn_auto_nobp_diag import centroid_readout_weights
        self.classes, self.Wro = centroid_readout_weights(F, y)        # lecture par centroides sur le code
        return self

    def code(self, X):
        return np.einsum("ck,nct,tl->nkl", self.W_L, X, self.W_R).reshape(len(X), -1)

    def predict(self, X):
        F = self.code(X); return self.classes[(np.c_[F, np.ones(len(F))] @ self.Wro).argmax(1)]

    def n_params(self): return int(self.U_L.size + self.U_R.size + self.Wro.size)

    def check_idempotent(self):
        P_L = self.U_L @ self.W_L.T; P_R = self.U_R @ self.W_R.T
        P = np.kron(P_R, P_L)
        return np.linalg.norm(P @ P - P) / np.linalg.norm(P), np.linalg.matrix_rank(P)


# ---------------------------------------------------------------- variante 2 : METRIQUE doublement oblique
class KroneckerMetricBlock:
    """Bloc oblique ferme dont la METRIQUE de mesure est de Kronecker : M = Sig_T^{-1} (x) Sig_S^{-1}.
    Representation : U = vec(M_c - M_bar) (rang C-1 complet, aucune restriction de Kronecker) ;
    mesure : w_c = vec(Sig_S^{-1} (M_c - M_bar) Sig_T^{-1}), c'est-a-dire oblique en espace
    (canaux) ET en temps. Sig_S, Sig_T estimees par flip-flop (Dutilleul) sur les residus,
    avec ridge : O(c^2 + t^2) statistiques au lieu de O((ct)^2). Jamais de matrice ct x ct."""
    def __init__(self, n_ff=5, ridge=0.05): self.n_ff, self.ridge = n_ff, ridge

    def fit(self, X, y):
        n, c, t = X.shape
        self.classes, yi = np.unique(y, return_inverse=True); Cn = len(self.classes)
        ns = np.bincount(yi, minlength=Cn).astype(float)
        Ms = np.stack([X[yi == k].mean(0) for k in range(Cn)]); Mbar = X.mean(0)
        R = X - Ms[yi]
        # flip-flop
        Sig_T_inv = np.eye(t)
        for _ in range(self.n_ff):
            Sig_S = np.einsum("nct,tu,nvu->cv", R, Sig_T_inv, R) / (n * t)
            Sig_S += self.ridge * np.trace(Sig_S) / c * np.eye(c); Sig_S_inv = np.linalg.inv(Sig_S)
            Sig_T = np.einsum("nct,cu,nuv->tv", R, Sig_S_inv, R) / (n * c)
            Sig_T += self.ridge * np.trace(Sig_T) / t * np.eye(t); Sig_T_inv = np.linalg.inv(Sig_T)
        self.Sig_S_inv, self.Sig_T_inv = Sig_S_inv, Sig_T_inv
        # bloc fast avec solveur de Kronecker : S^{-1} vec(A) = vec(Sig_S^{-1} A Sig_T^{-1})
        Dm = np.stack([(Ms[k] - Mbar) * np.sqrt(ns[k]) for k in range(Cn)]) / np.sqrt(max(Cn - 1, 1))  # (C, c, t)
        Zm = np.einsum("cu,kut,tv->kcv", Sig_S_inv, Dm, Sig_T_inv)                                       # (C, c, t)
        D = Dm.reshape(Cn, -1).T; Z = Zm.reshape(Cn, -1).T
        M = D.T @ Z; M = 0.5 * (M + M.T)
        lam, Q = np.linalg.eigh(M); o = np.argsort(lam)[::-1]; lam, Q = lam[o], Q[:, o]
        r = Cn - 1; lr = np.clip(lam[:r], 1e-12, None)
        self.W = Z @ Q[:, :r] / np.sqrt(lr); self.U = D @ Q[:, :r] / np.sqrt(lr)
        F = X.reshape(n, -1) @ self.W
        from ropn_auto_nobp_diag import centroid_readout_weights
        _, self.Wro = centroid_readout_weights(F, yi)                   # regle LDA de covariance de Kronecker
        return self

    def predict(self, X):
        F = X.reshape(len(X), -1) @ self.W
        return self.classes[(np.c_[F, np.ones(len(F))] @ self.Wro).argmax(1)]

    def n_params(self): return int(self.U.size + self.Wro.size)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from ropn_auto_nobp_diag import readout_centroid_fit_predict
    SEEDS = 8
    print(f"=== donnees matricielles 16 x 32 (d = 512), C = 3, motifs de classe de rang 2, bruit Sig_S (x) Sig_T (kappa = 30 de chaque cote), {SEEDS} graines, 70/30 stratifie ===")
    print(f"{'separation':>10s} {'n/classe':>8s} {'n_train':>7s} | {'bloc vectorise':>15s} | {'metrique Kronecker':>18s} | {'projecteur Kronecker':>20s} | {'MLP (64)':>12s}")
    for sep, n_per in [(0.25, 20), (0.25, 60), (0.15, 60), (0.15, 200), (0.10, 200)]:
        res = {k: [] for k in ("vec", "kmet", "kproj", "mlp")}
        for sd in range(SEEDS):
            X, y = make_matrix_data(n_per=n_per, sep=sep, seed=sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
            n = len(Xtr); Vtr, Vte = Xtr.reshape(n, -1), Xte.reshape(len(Xte), -1)
            U, W, r, _ = fit_block_fast(Vtr, ytr, shrink=True, rank="C1")          # bloc vectorise (d = c t, forme duale)
            res["vec"].append((readout_centroid_fit_predict(Vtr @ W, ytr, Vte @ W) == yte).mean())
            res["kmet"].append((KroneckerMetricBlock().fit(Xtr, ytr).predict(Xte) == yte).mean())
            m = BiObliqueBlock().fit(Xtr, ytr); res["kproj"].append((m.predict(Xte) == yte).mean())
            if sd == 0 and (sep, n_per) == (0.25, 20):
                err, rk = m.check_idempotent(); idem = (err, rk, m.W_L.shape[1] * m.W_R.shape[1])
            res["mlp"].append(MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=sd).fit(Vtr, ytr).score(Vte, yte))
        f = lambda v: f"{100*np.mean(v):5.1f} +/- {100*np.std(v):4.1f}"
        print(f"{sep:10.2f} {n_per:8d} {n:7d} | {f(res['vec']):>15s} | {f(res['kmet']):>18s} | {f(res['kproj']):>20s} | {f(res['mlp']):>12s}")
    print(f"idempotence du projecteur de Kronecker P_R (x) P_L : ||P^2-P||/||P|| = {idem[0]:.1e}, rang = {idem[1]} = r_L r_R = {idem[2]}")
