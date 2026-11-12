"""
Triceptron : unite a TROIS directions
    u (representation), w (mesure), s (porte lue HORS de span(W)).
    perceptron  : y = sigma(w^T x + b)                            (1 vecteur)
    oblique     : y = sigma(w^T x / w^T u + b)                    (2 vecteurs)
    triceptron  : y = sigma(s^T x + b_s) * (w^T x / w^T u)  -> lecture   (3 vecteurs)
Version rang r : code c = W^T x (r), portes a = sigma(S^T x + b) (k), sortie U (a (x) c) ou lecture.

Deux realisations :
  * TRI-CF (ferme, sans gradient) : (U, W) = bloc fast (rang C-1, W^T U = I) ; S = k premieres
    composantes principales du RESIDU (I-P)x (donc hors de span(W)) ; caracteristiques
    [c, c_i * a_j] ; lecture par moindres carres ; k choisi par validation croisee interne.
  * TRI-BP (retropropagation, gradients manuels numpy, verifies) : logits = V (a (x) c) + b0 avec
    c = W^T x + b_w, a = sigma(S^T x + b_s) : c'est une unite GLU (gated linear unit), i.e. la
    structure du FFN des LLM modernes (SwiGLU : W_up = W, W_gate = S, W_down = V).
    Comparee a un MLP ReLU a nombre de parametres voisin entraine par le MEME code (meme initialisation,
    meme Adam, meme nombre de pas), avec et sans saut lineaire pour les deux familles ; un MLP sklearn
    L-BFGS est rapporte en plus comme controle d'un autre optimiseur. Poids construits (theoreme) contre
    poids appris : section 2 du programme principal.
"""
import numpy as np
from numpy.linalg import svd
from ropn_auto_fast import fit_block_fast


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ---------------------------------------------------------------- TRI-CF
class TriceptronCF:
    def __init__(self, k_gates=None, seed=0):
        self.k, self.seed = k_gates, seed

    def _feats(self, X, fit=False, y=None):
        if fit:
            self.U, self.W, self.r, _ = fit_block_fast(X, y, shrink=True, rank="C1")
            R = X - (X @ self.W) @ self.U.T                     # residu (I-P)x : hors de span(W)
            self.r_mean = R.mean(0)
            _, _, Vt = svd(R - self.r_mean, full_matrices=False)
            k = self.r if self.k is None else self.k
            self.S = Vt[:k].T                                    # directions de porte s_j (orthonormees)
            G = (R - self.r_mean) @ self.S; self.g_scale = G.std(0) + 1e-12
        c = X @ self.W                                           # code oblique (r)
        if self.S.shape[1] == 0:
            return c
        a = sigmoid(((X - (X @ self.W) @ self.U.T - self.r_mean) @ self.S) / self.g_scale)   # portes (k)
        return np.concatenate([c, (c[:, :, None] * a[:, None, :]).reshape(len(X), -1)], 1)

    def fit(self, X, y):
        Z = self._feats(X, fit=True, y=y)
        self.classes = np.unique(y); Y = np.eye(len(self.classes))[np.searchsorted(self.classes, y)]
        A = np.c_[Z, np.ones(len(Z))]; self.Wro, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return self

    def predict(self, X):
        Z = self._feats(X); return self.classes[(np.c_[Z, np.ones(len(Z))] @ self.Wro).argmax(1)]

    def n_params(self):
        return int(self.U.size + self.W.size + self.S.size + self.Wro.size)


def fit_tricf_auto(Xtr, ytr, k_grid=(0, 1, 2, 4, 8), n_folds=3, seed=0):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best, bk = -1, 0
    for k in k_grid:
        accs = [(TriceptronCF(k).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]) == ytr[va]).mean() for tr, va in skf.split(Xtr, ytr)]
        if np.mean(accs) > best + 1e-9: best, bk = np.mean(accs), k
    return TriceptronCF(bk).fit(Xtr, ytr), bk


# ---------------------------------------------------------------- reseaux a une couche cachee, gradients manuels
class TriceptronBP:
    """Reseau a une couche cachee entraine par Adam (gradients manuels), deux familles et un saut lineaire optionnel :
        kind="glu"  (triceptron, TRI-BP) : h = sigmoid(X S + b_s) * (X W + b_w)
        kind="relu" (MLP)                : h = max(0, X W + b_w)
        logits = h V + b0 (+ X K si skip=True, K de taille d x C : le saut lineaire de la construction theorique).
    Meme initialisation, meme optimiseur, meme nombre de pas et meme penalite L2 pour les deux familles :
    la comparaison ne depend pas d'un optimiseur different. Entropie croisee + L2."""
    def __init__(self, r=16, steps=400, lr=1e-2, wd=1e-4, seed=0, init=None, kind="glu", skip=False):
        self.r, self.steps, self.lr, self.wd, self.seed, self.init, self.kind, self.skip = r, steps, lr, wd, seed, init, kind, skip

    def _fwd(self, p, X):
        W, bw, S, bs, V, b0, K = p
        c = X @ W + bw
        if self.kind == "glu": a = sigmoid(X @ S + bs); h = a * c
        else: a = (c > 0).astype(float); h = a * c
        out = h @ V + b0
        if self.skip: out = out + X @ K
        return out, (c, a, h)

    def _loss_grads(self, p, X, yi, C):
        W, bw, S, bs, V, b0, K = p; n = len(X)
        logits, (c, a, h) = self._fwd(p, X)
        z = logits - logits.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
        loss = -np.log(pr[np.arange(n), yi] + 1e-12).mean() + self.wd * (np.sum(W**2) + np.sum(S**2) + np.sum(V**2) + np.sum(K**2))
        dl = pr.copy(); dl[np.arange(n), yi] -= 1; dl /= n
        dV = h.T @ dl + 2 * self.wd * V; db0 = dl.sum(0)
        dK = (X.T @ dl if self.skip else 0.0) + 2 * self.wd * K
        dh = dl @ V.T
        dc = dh * a
        dW = X.T @ dc + 2 * self.wd * W; dbw = dc.sum(0)
        if self.kind == "glu":
            dz = dh * c * a * (1 - a); dS = X.T @ dz + 2 * self.wd * S; dbs = dz.sum(0)
        else:
            dS = np.zeros_like(S); dbs = np.zeros_like(bs)
        return loss, [dW, dbw, dS, dbs, dV, db0, dK]

    def _init(self, d, C, rng):
        r = self.r
        p = [rng.standard_normal((d, r)) / np.sqrt(d), np.zeros(r), rng.standard_normal((d, r)) / np.sqrt(d), np.zeros(r),
             rng.standard_normal((r, C)) / np.sqrt(r), np.zeros(C), np.zeros((d, C))]
        if self.kind != "glu": p[2] = np.zeros((d, r))                # pas de porte pour le MLP
        return p

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes, yi = np.unique(y, return_inverse=True); C = len(self.classes); d = X.shape[1]
        p = self._init(d, C, rng) if self.init is None else [q.copy() for q in self.init]
        m = [np.zeros_like(q) for q in p]; v = [np.zeros_like(q) for q in p]
        for t in range(1, self.steps + 1):
            _, g = self._loss_grads(p, X, yi, C)
            for i in range(len(p)):
                m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
                p[i] -= self.lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        self.p = p
        return self

    def predict(self, X):
        return self.classes[self._fwd(self.p, X)[0].argmax(1)]

    def n_params(self):
        W, bw, S, bs, V, b0, K = self.p
        n = W.size + bw.size + V.size + b0.size
        if self.kind == "glu": n += S.size + bs.size
        if self.skip: n += K.size
        return int(n)

    def max_weight(self):
        return float(max(np.abs(q).max() for q in self.p))

    def gradcheck(self, d=5, r=3, C=3, n=7, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, d)); yi = rng.integers(0, C, n)
        p = [rng.standard_normal((d, r)), rng.standard_normal(r), rng.standard_normal((d, r)), rng.standard_normal(r),
             rng.standard_normal((r, C)), rng.standard_normal(C), rng.standard_normal((d, C)) if self.skip else np.zeros((d, C))]
        if self.kind != "glu": p[2] = np.zeros((d, r)); p[3] = np.zeros(r)
        _, g = self._loss_grads(p, X, yi, C); worst = 0.0; eps = 1e-6
        active = [0, 1, 4, 5] + ([2, 3] if self.kind == "glu" else []) + ([6] if self.skip else [])
        for i in active:
            it = np.nditer(p[i], flags=["multi_index"])
            for _ in it:
                idx = it.multi_index; q = [a.copy() for a in p]
                q[i][idx] += eps; lp = self._loss_grads(q, X, yi, C)[0]
                q[i][idx] -= 2 * eps; lm = self._loss_grads(q, X, yi, C)[0]
                worst = max(worst, abs((lp - lm) / (2 * eps) - g[i][idx]) / (abs(g[i][idx]) + 1e-8))
        return worst


def width_for_params(d, C, n_target, skip):
    """largeur h d'un MLP (d->h->C, + saut lineaire d x C si skip) dont le nombre de parametres approche n_target."""
    return max(1, int(round((n_target - C - (d * C if skip else 0)) / (d + 1 + C))))


def mlp_width_for_params(d, C, n_target):
    """largeur h d'un MLP (d->h->C) dont le nombre de parametres approche n_target."""
    return max(2, int(round((n_target - C) / (d + 1 + C))))


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import load_digits, load_breast_cancer, load_wine, make_moons, make_circles
    from ropn_auto_nobp import make_multiclass
    from ropn_auto_finetune import make_hetero
    from ropn_auto_nobp_diag import readout_fit_predict, readout_centroid_fit_predict
    from ropn_auto_stats import fmt_paired

    print("gradcheck : " + " ; ".join(f"{k}{'+saut' if sk else ''} {TriceptronBP(kind=k, skip=sk).gradcheck():.1e}" for k in ("glu", "relu") for sk in (False, True)))

    def prep(X, y, sd):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
        keep = Xtr.std(0) > 1e-8; sc = StandardScaler().fit(Xtr[:, keep])
        return sc.transform(Xtr[:, keep]), sc.transform(Xte[:, keep]), ytr, yte

    def gen2d(name, seed):
        """tirage independant d'un probleme 2-D (n = 400)."""
        rng = np.random.default_rng(seed)
        if name == "two moons": return make_moons(400, noise=0.15, random_state=seed)
        if name == "circles": return make_circles(400, noise=0.1, factor=0.5, random_state=seed)
        X = rng.standard_normal((400, 2)); return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    STEPS, f = 2000, (lambda v: f"{np.mean(v)*100:5.1f}+/-{np.std(v)*100:4.1f}")
    # ------------------------------------------------------------------ 1. comparaison symetrique sur les problemes 2-D
    DRAWS = 10
    print(f"\n=== 1. Problemes 2-D : {DRAWS} tirages INDEPENDANTS (n = 400, 70/30), meme entraineur pour les deux familles (Adam, {STEPS} pas, lr 1e-2, L2 1e-4) ===")
    print("    GLU = triceptron (r unites), MLP = ReLU (h unites) a nombre de parametres voisin ; avec ou sans saut lineaire ; L-BFGS : MLP sklearn sans saut (autre optimiseur)")
    print(f"    {'jeu':10s} {'r':>2s} | {'par':>4s} {'GLU':>11s} | {'h':>2s} {'par':>4s} {'MLP':>11s} {'MLP L-BFGS':>11s} | {'par':>4s} {'GLU+saut':>11s} | {'h':>2s} {'par':>4s} {'MLP+saut':>11s} | GLU - MLP [IC 95 %] | GLU+saut - MLP+saut [IC 95 %]")
    for name in ("two moons", "circles", "XOR"):
        for r_ in (16, 4, 2, 1):
            res = {k: [] for k in ("glu", "relu", "lbfgs", "glu_s", "relu_s")}; meta = {}
            for sd in range(DRAWS):
                X, y = gen2d(name, sd); Xtr, Xte, ytr, yte = prep(X, y, sd); d, C = 2, 2
                g = TriceptronBP(r=r_, steps=STEPS, seed=sd, kind="glu").fit(Xtr, ytr); res["glu"].append((g.predict(Xte) == yte).mean())
                h = width_for_params(d, C, g.n_params(), False)
                m = TriceptronBP(r=h, steps=STEPS, seed=sd, kind="relu").fit(Xtr, ytr); res["relu"].append((m.predict(Xte) == yte).mean())
                res["lbfgs"].append(MLPClassifier(hidden_layer_sizes=(h,), solver="lbfgs", max_iter=2000, random_state=sd).fit(Xtr, ytr).score(Xte, yte))
                gs = TriceptronBP(r=r_, steps=STEPS, seed=sd, kind="glu", skip=True).fit(Xtr, ytr); res["glu_s"].append((gs.predict(Xte) == yte).mean())
                hs = width_for_params(d, C, gs.n_params(), True)
                ms = TriceptronBP(r=hs, steps=STEPS, seed=sd, kind="relu", skip=True).fit(Xtr, ytr); res["relu_s"].append((ms.predict(Xte) == yte).mean())
                meta = dict(pg=g.n_params(), h=h, pm=m.n_params(), pgs=gs.n_params(), hs=hs, pms=ms.n_params())
            print(f"    {name:10s} {r_:2d} | {meta['pg']:4d} {f(res['glu']):>11s} | {meta['h']:2d} {meta['pm']:4d} {f(res['relu']):>11s} {f(res['lbfgs']):>11s} | {meta['pgs']:4d} {f(res['glu_s']):>11s} | {meta['hs']:2d} {meta['pms']:4d} {f(res['relu_s']):>11s} | "
                  f"{fmt_paired(res['glu'], res['relu'])} | {fmt_paired(res['glu_s'], res['relu_s'])}")

    # ------------------------------------------------------------------ 2. poids construits contre poids appris
    print("\n=== 2. Poids CONSTRUITS (constructions du theoreme de capacite, aucun apprentissage) contre poids APPRIS, donnees brutes non standardisees, 10 tirages ===")
    sig = sigmoid
    for gam in (1.0, 0.5, 0.2, 0.05):
        acc_c, acc_x = [], []
        for sd in range(DRAWS):
            X, y = gen2d("circles", sd); fval = (sig(gam * X) * X - 0.5 * X).sum(1) - gam / 4 * 0.75 ** 2       # 2 unites + saut ; seuil au rayon 0.75
            acc_c.append(((fval < 0).astype(int) == y).mean())                                                  # classe 1 = cercle interieur
            X, y = gen2d("XOR", sd); fx = sig(gam * X[:, 0]) * X[:, 1] - 0.5 * X[:, 1]                           # 1 unite + saut
            acc_x.append(((fx < 0).astype(int) == y).mean())
        print(f"    construit, gamma = {gam:4.2f} : cercles (2 unites + saut) {f(acc_c)} | XOR (1 unite + saut) {f(acc_x)} | plus grand poids sous forme normalisee (lecture 4/gamma) = {4/gam:5.1f} | ecart radial theorique gamma^2 R^3/24 = {gam**2*0.75**3/24:.1e}")
    for name, r_ in (("circles", 2), ("XOR", 1)):
        acc_l, wmax, sharp = [], [], []
        for sd in range(DRAWS):
            X, y = gen2d(name, sd); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
            g = TriceptronBP(r=r_, steps=STEPS, seed=sd, kind="glu", skip=True).fit(Xtr, ytr)
            acc_l.append((g.predict(Xte) == yte).mean()); wmax.append(g.max_weight()); sharp.append(np.linalg.norm(g.p[2], axis=0).max())
        print(f"    appris, {name:8s} GLU+saut r={r_} ({g.n_params()} parametres) : precision test {f(acc_l)} | plus grand poids {np.mean(wmax):5.1f} | norme max d'une direction de porte ||s|| = {np.mean(sharp):4.1f}")
    print("    (precision des poids construits evaluee sur les 400 points de chaque tirage ; les unites apprises sont loin du regime doux gamma -> 0 de la construction.)")

    # ------------------------------------------------------------------ 3. jeux de dimension plus elevee
    rng = np.random.default_rng(0)
    ds = {}
    Xm, ym = make_moons(400, noise=0.15, random_state=0); ds["two moons (2D)"] = (Xm, ym)
    Xc, yc = make_circles(400, noise=0.1, factor=0.5, random_state=0); ds["circles (2D)"] = (Xc, yc)
    Xx = rng.standard_normal((400, 2)); yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int); ds["XOR (2D)"] = (Xx, yx)
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=10.0, sep=0.8, seed=0); ds["gauss n=15/cl"] = (X, y)
    X, y, *_ = make_hetero(n_classes=4, d=20, n_per=60, kappa=50.0, sep=0.8, seed=0); ds["hetero n=60/cl"] = (X, y)
    d_ = load_digits(); ds["digits"] = (d_.data, d_.target)
    b_ = load_breast_cancer(); ds["breast"] = (b_.data, b_.target)
    w_ = load_wine(); ds["wine"] = (w_.data, w_.target)
    SEEDS = 5
    print(f"\n=== 3. Tous les jeux, r = 16, {SEEDS} decoupages 70/30 d'un jeu fixe : bloc analytique, TRI-CF, GLU et MLP (meme entraineur), MLP sklearn L-BFGS ===")
    print(f"    {'jeu':16s} | {'bloc fast':>11s} | {'TRI-CF auto':>11s} {'k':>2s} | {'par':>5s} {'GLU':>11s} | {'h':>3s} {'par':>5s} {'MLP':>11s} {'MLP L-BFGS':>11s} | GLU - MLP [IC 95 %, variabilite du decoupage]")
    for name, (X, y) in ds.items():
        fast, cf, ks, res = [], [], [], {k: [] for k in ("glu", "relu", "lbfgs")}
        for sd in range(SEEDS):
            Xtr, Xte, ytr, yte = prep(X, y, sd); d, C = Xtr.shape[1], len(np.unique(ytr))
            U, W, r, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")
            fast.append((readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W) == yte).mean())   # lecture de reference
            m_, k = fit_tricf_auto(Xtr, ytr, seed=sd); cf.append((m_.predict(Xte) == yte).mean()); ks.append(k)
            g = TriceptronBP(r=16, steps=STEPS, seed=sd, kind="glu").fit(Xtr, ytr); res["glu"].append((g.predict(Xte) == yte).mean())
            h = width_for_params(d, C, g.n_params(), False)
            m = TriceptronBP(r=h, steps=STEPS, seed=sd, kind="relu").fit(Xtr, ytr); res["relu"].append((m.predict(Xte) == yte).mean())
            res["lbfgs"].append(MLPClassifier(hidden_layer_sizes=(h,), solver="lbfgs", max_iter=2000, random_state=sd).fit(Xtr, ytr).score(Xte, yte))
        print(f"    {name:16s} | {f(fast):>11s} | {f(cf):>11s} {np.mean(ks):2.0f} | {g.n_params():5d} {f(res['glu']):>11s} | {h:3d} {m.n_params():5d} {f(res['relu']):>11s} {f(res['lbfgs']):>11s} | {fmt_paired(res['glu'], res['relu'])}")
