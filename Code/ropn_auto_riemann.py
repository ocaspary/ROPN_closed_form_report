"""
Bloc oblique ferme + geometrie riemannienne (variete SPD des covariances, cadre standard des BCI).

Ce que deviennent les algorithmes quand les entrees sont des matrices de covariance :
  * bloc fast sur l'espace tangent au point moyen riemannien (metrique affine-invariante, TS-AI),
  * bloc fast sur log-euclidien (logm(Sigma), sans moyenne, TS-LE),
  * bloc fast euclidien sur vec(Sigma) (sans geometrie),
  * bloc a metrique de Kronecker sur les essais bruts (canaux x temps, geometrie doublement oblique),
  * MDM (distance riemannienne minimale aux moyennes de classe) et TS + MLP comme references.

Proposition d'invariance par congruence (verifiee ici) : la chaine {moyenne de Karcher -> espace tangent -> bloc fast (LW, GEVD C x C)
-> lecture} est EXACTEMENT invariante par congruence Sigma -> A Sigma A^T pour tout A inversible (gains de
canaux, re-referencement, melange), y compris avec le retrecissement de Ledoit-Wolf, parce que les vecteurs
tangents ne subissent qu'une rotation orthogonale. La version euclidienne vec(Sigma) ne l'est pas.

Deux regimes de donnees synthetiques :
  (MI)  classes = covariances spatiales differentes, signaux centres (imagerie motrice : ERD/ERS),
  (ERP) classes = motifs moyens differents, covariance commune (potentiels evoques).
numpy / scipy / sklearn seulement.
"""
import time
import numpy as np
from numpy.linalg import eigh, norm
from sklearn.neural_network import MLPClassifier
from ropn_auto_fast import fit_block_fast
from ropn_auto_nobp_diag import readout_centroid_fit_predict
from ropn_auto_bioblique import KroneckerMetricBlock


# ------------------------------------------------------------------ SPD utilities
def sym_fun(S, f):
    lam, V = eigh(S); return (V * f(lam)) @ V.T


def logm(S): return sym_fun(S, np.log)
def expm(S): return sym_fun(S, np.exp)
def sqrtm(S): return sym_fun(S, np.sqrt)
def invsqrtm(S): return sym_fun(S, lambda l: 1.0 / np.sqrt(l))


def vec_F(S):
    """Vectorisation isometrique (Frobenius) des matrices symetriques : diag, puis sqrt2 * hors-diag."""
    p = S.shape[-1]; iu = np.triu_indices(p); w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[..., iu[0], iu[1]] * w


def karcher_mean(Sig, iters=50, tol=1e-10, return_iters=False):
    G = Sig.mean(0)
    for it in range(1, iters + 1):
        Gs, Gis = sqrtm(G), invsqrtm(G)
        Tm = np.mean([logm(Gis @ S @ Gis) for S in Sig], 0)
        G = Gs @ expm(Tm) @ Gs
        if norm(Tm) < tol: break
    return (G, it) if return_iters else G


def tangent_space(Sig, G):
    Gis = invsqrtm(G); return np.stack([vec_F(logm(Gis @ S @ Gis)) for S in Sig])


def riem_dist(A, B):
    Ais = invsqrtm(A); return norm(logm(Ais @ B @ Ais))


class MDM:
    def fit(self, Sig, y):
        self.classes = np.unique(y); self.means = [karcher_mean(Sig[y == c]) for c in self.classes]; return self

    def predict(self, Sig):
        D = np.array([[riem_dist(M, S) for M in self.means] for S in Sig]); return self.classes[D.argmin(1)]


# ------------------------------------------------------------------ data
def make_trials(regime, C=3, p=8, T=64, n_per=20, sep=0.1, gain_sd=0.3, seed=0, A=None):
    """Essais X (n, p, T). MI : covariance spatiale par classe (base + sep * u u^T, rang 2), moyenne nulle.
    ERP : motif moyen par classe (sep * a * onde), covariance commune. Gain d'amplitude aleatoire par essai
    (exp(gain_sd * N(0,1)), non-stationnarite). A : melange/gains de canaux appliques a tous les essais.
    Retourne X, y (labels 0..C-1)."""
    rng = np.random.default_rng(seed)
    def rand_spd(scale):
        Q, _ = np.linalg.qr(rng.standard_normal((p, p))); lam = np.exp(scale * rng.standard_normal(p))
        return Q @ np.diag(lam) @ Q.T
    base = rand_spd(0.5); X, y = [], []
    for c in range(C):
        if regime == "MI":
            u = rng.standard_normal((p, 2)); Rc = sqrtm(base + sep * (u @ u.T)); Mc = 0.0
        else:
            Rc = sqrtm(base); a = rng.standard_normal((p, 1))
            wave = np.sin(2 * np.pi * rng.uniform(1, 3) * np.linspace(0, 1, T)); Mc = sep * a @ wave[None, :]
        for _ in range(n_per):
            g = np.exp(gain_sd * rng.standard_normal()); X.append(g * (Mc + Rc @ rng.standard_normal((p, T)))); y.append(c)
    X, y = np.stack(X), np.array(y); perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    if A is not None: X = np.einsum("ij,njt->nit", A, X)
    return X, y


def covariances(X):
    Xc = X - X.mean(2, keepdims=True); return np.einsum("npt,nqt->npq", Xc, Xc) / (X.shape[2] - 1)


# ------------------------------------------------------------------ pipelines
def block_on_features(Ftr, ytr, Fte):
    U, W, r, _ = fit_block_fast(Ftr, ytr, shrink=True, rank="C1")
    return readout_centroid_fit_predict(Ftr @ W, ytr, Fte @ W), W


def run_all(Xtr, ytr, Xte, yte, seed=0, kron=True):
    Ctr, Cte = covariances(Xtr), covariances(Xte); out = {}
    t0 = time.time(); G = karcher_mean(Ctr); Ttr, Tte = tangent_space(Ctr, G), tangent_space(Cte, G)
    pr, _ = block_on_features(Ttr, ytr, Tte); out["TS-AI + bloc fast"] = ((pr == yte).mean(), time.time() - t0)
    t0 = time.time(); Ltr = np.stack([vec_F(logm(S)) for S in Ctr]); Lte = np.stack([vec_F(logm(S)) for S in Cte])
    pr, _ = block_on_features(Ltr, ytr, Lte); out["TS-LE + bloc fast"] = ((pr == yte).mean(), time.time() - t0)
    t0 = time.time(); pr, _ = block_on_features(vec_F(Ctr), ytr, vec_F(Cte)); out["vec(Sigma) + bloc fast"] = ((pr == yte).mean(), time.time() - t0)
    if kron:
        t0 = time.time(); kb = KroneckerMetricBlock().fit(Xtr, ytr); out["Kronecker (essais bruts)"] = ((kb.predict(Xte) == yte).mean(), time.time() - t0)
    t0 = time.time(); out["MDM"] = ((MDM().fit(Ctr, ytr).predict(Cte) == yte).mean(), time.time() - t0)
    t0 = time.time(); mlp = MLPClassifier((64,), max_iter=800, random_state=seed).fit(Ttr, ytr)
    out["TS-AI + MLP(64)"] = (mlp.score(Tte, yte), time.time() - t0)
    return out


if __name__ == "__main__":
    print("=== Invariance exacte par congruence de la chaine TS-AI + bloc fast (regime MI, sep=0.1) ===")
    X, y = make_trials("MI", p=8, n_per=40, seed=0); n = len(y); Xtr, Xte, ytr, yte = X[:n // 2], X[n // 2:], y[:n // 2], y[n // 2:]
    rng = np.random.default_rng(1)
    A = np.diag(np.exp(rng.uniform(-2.3, 2.3, 8))) @ np.linalg.qr(rng.standard_normal((8, 8)))[0]   # gains 0.1..10 + rotation
    print(f"    A : gains de canaux entre 0.1 et 10 puis rotation, conditionnement {np.linalg.cond(A):.1f}")
    Ctr, Cte = covariances(Xtr), covariances(Xte)
    CtrA = np.einsum("ij,njk,lk->nil", A, Ctr, A); CteA = np.einsum("ij,njk,lk->nil", A, Cte, A)
    G = karcher_mean(Ctr); GA = karcher_mean(CtrA)
    print(f"    moyenne de Karcher equivariante : ||G(A Sigma A^T) - A G A^T|| / ||.|| = {norm(GA - A @ G @ A.T) / norm(GA):.1e}")
    for name, F, FA in [("TS-AI", (tangent_space(Ctr, G), tangent_space(Cte, G)), (tangent_space(CtrA, GA), tangent_space(CteA, GA))),
                        ("TS-LE", (np.stack([vec_F(logm(S)) for S in Ctr]), np.stack([vec_F(logm(S)) for S in Cte])),
                                  (np.stack([vec_F(logm(S)) for S in CtrA]), np.stack([vec_F(logm(S)) for S in CteA]))),
                        ("vec(Sigma)", (vec_F(Ctr), vec_F(Cte)), (vec_F(CtrA), vec_F(CteA)))]:
        pr, W = block_on_features(F[0], ytr, F[1]); prA, WA = block_on_features(FA[0], ytr, FA[1])
        gram_gap = norm(F[0] @ F[0].T - FA[0] @ FA[0].T) / norm(F[0] @ F[0].T)      # Gram identique <=> rotation orthogonale
        code_gap = norm(np.sort(F[0] @ W, 0) - np.sort(FA[0] @ WA, 0)) / norm(F[0] @ W)
        print(f"    {name:10s} : Gram des features conserve a {gram_gap:.1e} | codes conserves a {code_gap:.1e} | "
              f"precision {100*(pr==yte).mean():.1f}% -> apres A : {100*(prA==yte).mean():.1f}% | predictions identiques : {np.mean(pr == prA)*100:.1f}%")

    print("\n=== Regimes MI (covariances, sep=0.1) et ERP (moyennes, sep=0.4), p=8, T=64, gain aleatoire par essai, 5 graines ===")
    for regime, n_per, sep in [("MI", 10, 0.1), ("MI", 30, 0.1), ("ERP", 10, 0.4), ("ERP", 30, 0.4)]:
        acc = {}
        for sd in range(5):
            X, y = make_trials(regime, p=8, n_per=2 * n_per, sep=sep, seed=sd); n = len(y)
            res = run_all(X[:n // 2], y[:n // 2], X[n // 2:], y[n // 2:], seed=sd)
            for k, (a, t) in res.items(): acc.setdefault(k, []).append((a, t))
        print(f"  regime {regime}, {n_per} essais/classe en apprentissage (d_tangent = {8*9//2}) :")
        for k, v in acc.items():
            a = np.array([x[0] for x in v]); t = np.mean([x[1] for x in v])
            print(f"      {k:26s} : {100*a.mean():5.1f} +- {100*a.std():4.1f} %   ({t*1000:6.0f} ms)")

    print("\n=== Changement de session : apprentissage avec gains A1, test avec gains A2 = A1 * exp(perturbation) (MI, 30/classe) ===")
    for eps in [0.0, 0.2, 0.5]:
        acc = {}
        for sd in range(5):
            rng = np.random.default_rng(100 + sd)
            A1 = np.diag(np.exp(rng.uniform(-0.7, 0.7, 8))); A2 = A1 @ expm(eps * (lambda M: 0.5 * (M + M.T))(rng.standard_normal((8, 8))))
            X, y = make_trials("MI", p=8, n_per=60, sep=0.1, seed=sd); n = len(y)      # memes classes, deux sessions
            Xtr, ytr = np.einsum("ij,njt->nit", A1, X[:n // 2]), y[:n // 2]; Xte, yte = np.einsum("ij,njt->nit", A2, X[n // 2:]), y[n // 2:]
            Ctr, Cte = covariances(Xtr), covariances(Xte); r = {}
            G = karcher_mean(Ctr); Ttr = tangent_space(Ctr, G)
            r["TS-AI, moyenne d'apprentissage"] = (block_on_features(Ttr, ytr, tangent_space(Cte, G))[0] == yte).mean()
            G2 = karcher_mean(Cte); pr_rc = block_on_features(Ttr, ytr, tangent_space(Cte, G2))[0]
            r["TS-AI, re-centre par session"] = (pr_rc == yte).mean()
            # residu apres re-centrage : rotation orthogonale Q = G2^{-1/2} A G1^{1/2}, A = A2 A1^{-1}, G1 = moyenne test sous A1
            Cte1 = covariances(np.einsum("ij,njt->nit", A1, X[n // 2:])); G1 = karcher_mean(Cte1); Aq = A2 @ np.linalg.inv(A1)
            Q = invsqrtm(G2) @ Aq @ sqrtm(G1); pr_ref = block_on_features(Ttr, ytr, tangent_space(Cte1, G1))[0]
            r["  (||Q - I||_F residuel)"] = norm(Q - np.eye(8)); r["  (predictions = sans changement)"] = (pr_rc == pr_ref).mean()
            r["TS-LE"] = (block_on_features(np.stack([vec_F(logm(S)) for S in Ctr]), ytr, np.stack([vec_F(logm(S)) for S in Cte]))[0] == yte).mean()
            r["vec(Sigma)"] = (block_on_features(vec_F(Ctr), ytr, vec_F(Cte))[0] == yte).mean()
            r["MDM"] = (MDM().fit(Ctr, ytr).predict(Cte) == yte).mean()
            r["TS-AI + MLP(64), re-centre"] = MLPClassifier((64,), max_iter=800, random_state=sd).fit(Ttr, ytr).score(tangent_space(Cte, karcher_mean(Cte)), yte)
            for k, v in r.items(): acc.setdefault(k, []).append(v)
        print(f"  perturbation eps = {eps} :")
        for k, v in acc.items():
            if k.startswith("  (||Q"): print(f"      {k:32s} : {np.mean(v):5.2f}")
            else: print(f"      {k:32s} : {100*np.mean(v):5.1f} +- {100*np.std(v):4.1f} %")

    print("\n=== Passage a l'echelle : p=32 canaux (d_tangent=528), 30 essais/classe, T=100, regime MI (sans Kronecker : ~1 min) ===")
    X, y = make_trials("MI", p=32, T=100, n_per=60, sep=0.1, seed=0); n = len(y)
    res = run_all(X[:n // 2], y[:n // 2], X[n // 2:], y[n // 2:], kron=False)
    for k, (a, t) in res.items(): print(f"      {k:26s} : {100*a:5.1f} %   ({t*1000:6.0f} ms)")

    print("\n=== Diagnostics : plus petite valeur propre des covariances d'essai, iterations de Karcher (tolerance 1e-10), distance du point de base a l'identite ===")
    for p_, T_, seeds_ in [(8, 64, range(5)), (32, 100, [0])]:
        mins, its, dist = [], [], []
        for sd in seeds_:
            X, y = make_trials("MI", p=p_, T=T_, n_per=60, sep=0.1, seed=sd); n = len(y); Cv = covariances(X)
            mins.append(min(np.linalg.eigvalsh(S).min() for S in Cv)); G, it = karcher_mean(Cv[:n // 2], return_iters=True); its.append(it); dist.append(norm(logm(G)))
        print(f"      p={p_:2d}, T={T_:3d} : plus petite valeur propre {min(mins):.3f} | iterations de Karcher {its} | ||log G||_F = {np.round(dist, 2).tolist()}")
