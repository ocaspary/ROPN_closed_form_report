"""
Reseau en treillis de blocs ROPN-auto fast (entierement ferme, sans retropropagation).

Principe. Un bloc ferme = LDA regularisee de rang <= C-1 : empiler des blocs lineaires n'ajoute
aucune expressivite (leur composition est lineaire), et le residu (I-P)x d'un bloc a des moyennes de classe
IDENTIQUES (U engendre span(mu_c - mu)), donc un second bloc lineaire sur ce residu ne trouve rien.
La profondeur ne peut apporter quelque chose que si une NON-LINEARITE FIXE separe deux etages, et
la largeur que si les blocs d'un meme etage voient des ENTREES DIFFERENTES (transversal).

Etage 1 (transversal, parallele) : G groupes de variables (patchs / bandes / groupes aleatoires),
    un bloc fast par groupe -> codes c_g in R^{C-1}, plus un bloc global -> code c_0 et residu r_0.
Non-linearite fixe : phi(c) = [c, c^2] (expose les differences de variance) ; sur le residu,
    monomes de degre 2 des k premieres composantes principales du residu (information de
    second ordre = covariances propres a chaque classe, la ou la LDA est aveugle).
Etage 2 : un bloc fast sur la concatenation (souvent n < d : forme duale), code final -> lecture.

Tout est ferme (Cholesky/Woodbury + problemes C x C), les blocs d'un etage sont independants.
"""
import numpy as np
from ropn_auto_nobp_diag import centroid_readout_weights
from itertools import combinations_with_replacement
from ropn_auto_fast import fit_block_fast


def _quad(R):
    """Monomes de degre 2 (avec carres) des colonnes de R : d(d+1)/2 colonnes."""
    idx = list(combinations_with_replacement(range(R.shape[1]), 2))
    return np.stack([R[:, i] * R[:, j] for i, j in idx], 1)


class LatticeROPN:
    def __init__(self, n_groups=4, group_mode="contiguous", k_quad=8, use_codes_sq=True,
                 use_resid_quad=True, seed=0):
        self.G, self.mode, self.k_quad = n_groups, group_mode, k_quad
        self.use_codes_sq, self.use_resid_quad, self.seed = use_codes_sq, use_resid_quad, seed

    # ---- etage 1 ----
    def _features1(self, X, fit=False, y=None):
        d = X.shape[1]
        if fit:
            rng = np.random.default_rng(self.seed)
            idx = np.arange(d) if self.mode == "contiguous" else rng.permutation(d)
            self.groups = np.array_split(idx, self.G)
            self.blocks = [fit_block_fast(X[:, g], y, shrink=True, rank="C1")[:2] for g in self.groups]
            self.U0, self.W0, _, _ = fit_block_fast(X, y, shrink=True, rank="C1")
        codes = [X[:, g] @ W for g, (U, W) in zip(self.groups, self.blocks)]          # transversal
        c0 = X @ self.W0                                                              # global
        feats = codes + [c0]
        if self.use_codes_sq:
            feats += [c ** 2 for c in feats]
        if self.use_resid_quad:
            R = X - (X @ self.W0) @ self.U0.T                                         # residu (I-P)x
            if fit:                                                                   # ACP du residu
                Rc = R - R.mean(0); _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
                self.r_mean, self.Vk = R.mean(0), Vt[:self.k_quad].T
                Rk = (R - self.r_mean) @ self.Vk; self.r_scale = Rk.std(0) + 1e-12
            Rk = ((R - self.r_mean) @ self.Vk) / self.r_scale
            feats += [_quad(Rk)]
        Z = np.concatenate(feats, 1)
        if fit:
            self.z_mean, self.z_std = Z.mean(0), Z.std(0) + 1e-12
        return (Z - self.z_mean) / self.z_std

    # ---- etage 2 + lecture ----
    def fit(self, X, y):
        Z = self._features1(X, fit=True, y=y)
        self.U2, self.W2, self.r2, self.info2 = fit_block_fast(Z, y, shrink=True, rank="C1")
        F = Z @ self.W2
        # lecture par centroides sur le code de l'etage 2 = regle LDA sur les variables de l'etage 2
        self.classes, self.Wro = centroid_readout_weights(F, y)
        self.n_features2 = Z.shape[1]
        return self

    def predict(self, X):
        F = self._features1(X) @ self.W2
        return self.classes[(np.c_[F, np.ones(len(F))] @ self.Wro).argmax(1)]

    def n_params(self):
        n = self.U0.size + sum(U.size for U, _ in self.blocks) + self.U2.size + self.Wro.size
        if self.use_resid_quad:
            n += self.Vk.size
        return int(n)


# ---------------------------------------------------------------- selection automatique de la configuration du treillis
def fit_lattice_auto(Xtr, ytr, n_groups=4, k_grid=(0, 4, 8, 16), sq_grid=(False, True), n_folds=3, seed=0):
    """Choisit (k_quad, codes^2) par validation croisee interne (chaque ajustement est ferme et
    coute quelques ms) : c'est la boucle grow/prune de l'auto-ROPN, sans gradient.
    k_quad = 0 desactive l'etage quadratique."""
    from sklearn.model_selection import StratifiedKFold
    best, best_cfg = -1.0, None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for sq in sq_grid:
        for k in k_grid:
            cfg = dict(n_groups=n_groups, use_codes_sq=sq, use_resid_quad=(k > 0), k_quad=max(k, 1), seed=seed)
            accs = []
            for tr, va in skf.split(Xtr, ytr):
                m = LatticeROPN(**cfg).fit(Xtr[tr], ytr[tr]); accs.append((m.predict(Xtr[va]) == ytr[va]).mean())
            if np.mean(accs) > best + 1e-9:
                best, best_cfg = np.mean(accs), cfg
    return LatticeROPN(**best_cfg).fit(Xtr, ytr), best_cfg


if __name__ == "__main__":
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import load_digits, load_breast_cancer, load_wine
    from sklearn.preprocessing import StandardScaler
    from ropn_auto_nobp import make_multiclass
    from ropn_auto_finetune import make_hetero
    from ropn_auto_nobp_diag import readout_centroid_fit_predict

    def prep(X, y, sd):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
        keep = Xtr.std(0) > 1e-8; sc = StandardScaler().fit(Xtr[:, keep])
        return sc.transform(Xtr[:, keep]), sc.transform(Xte[:, keep]), ytr, yte

    datasets = {}
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=10.0, sep=0.8, seed=0); datasets["gauss k=10 n=15/cl"] = (X, y)
    X, y, *_ = make_hetero(n_classes=4, d=20, n_per=60, kappa=50.0, sep=0.8, seed=0);       datasets["hetero k=50 n=60/cl"] = (X, y)
    X, y, *_ = make_hetero(n_classes=4, d=20, n_per=300, kappa=50.0, sep=0.8, seed=0);      datasets["hetero k=50 n=300/cl"] = (X, y)
    d_ = load_digits(); datasets["digits 8x8"] = (d_.data, d_.target)
    b_ = load_breast_cancer(); datasets["breast cancer"] = (b_.data, b_.target)
    w_ = load_wine(); datasets["wine"] = (w_.data, w_.target)

    configs = {
        "bloc unique (ref.)":            None,
        "treillis: transversal seul":    dict(n_groups=4, use_codes_sq=False, use_resid_quad=False),
        "treillis: transv. + [c,c^2]":   dict(n_groups=4, use_codes_sq=True,  use_resid_quad=False),
        "treillis: + quad(residu, k=8)": dict(n_groups=4, use_codes_sq=True,  use_resid_quad=True, k_quad=8),
        "treillis: + quad(residu,k=16)": dict(n_groups=4, use_codes_sq=True,  use_resid_quad=True, k_quad=16),
        "treillis AUTO (CV interne)":    "auto",
    }
    SEEDS = 5
    for dname, (X, y) in datasets.items():
        n, d, C = X.shape[0], X.shape[1], len(np.unique(y))
        print(f"\n=== {dname} (n={n}, d={d}, C={C}, {SEEDS} decoupages 70/30 d'un meme tirage) ===")
        res = {k: [] for k in list(configs) + ["MLP (backprop)"]}; times = {k: [] for k in res}; npar = {}
        for sd in range(SEEDS):
            Xtr, Xte, ytr, yte = prep(X, y, sd)
            for k, cfg in configs.items():
                t0 = time.perf_counter()
                if cfg is None:
                    U, W, r, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")
                    pred = readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W); npar.setdefault(k, []).append(U.size + (r + 1) * C)
                elif cfg == "auto":
                    m, _ = fit_lattice_auto(Xtr, ytr, seed=sd); pred = m.predict(Xte); npar.setdefault(k, []).append(m.n_params())
                else:
                    m = LatticeROPN(seed=sd, **cfg).fit(Xtr, ytr); pred = m.predict(Xte); npar.setdefault(k, []).append(m.n_params())
                times[k].append(time.perf_counter() - t0); res[k].append((pred == yte).mean())
            t0 = time.perf_counter()
            h = int(np.clip(2 * Xtr.shape[1], 32, 256))
            mlp = MLPClassifier(hidden_layer_sizes=(h,), max_iter=500, random_state=sd).fit(Xtr, ytr)
            times["MLP (backprop)"].append(time.perf_counter() - t0)
            res["MLP (backprop)"].append(mlp.score(Xte, yte))
            npar.setdefault("MLP (backprop)", []).append(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
        for k in res:
            print(f"  {k:32s} acc = {np.mean(res[k])*100:5.1f} +/- {np.std(res[k])*100:4.1f} | "
                  f"{np.mean(times[k])*1000:7.1f} ms | {np.mean(npar[k]):7.0f} param. (moyenne sur les decoupages)")
