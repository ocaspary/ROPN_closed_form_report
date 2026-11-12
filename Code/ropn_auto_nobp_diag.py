"""
Diagnostic du regime petits echantillons : ou le bloc ferme perd-il de l'information ?

Decomposition (par graine, moyenne sur 20 graines) :
  (0) Bayes oracle  : regle LDA avec la VRAIE covariance et les VRAIES moyennes (borne sup.)
  (1) LDA-shrinkage : reference classique (sklearn, shrinkage='auto')
  (2) bloc ferme, variantes d'ablation :
        base        : S_W brut (n>d), rang auto (permutation), lecture LS sur h_T (dim d), T=4
        +shrink     : Ledoit-Wolf systematique sur S_W
        +code       : lecture sur le code oblique c = W^T h_T (dim r) au lieu de h_T (dim d)
        +rankC1     : rang structurel r = C-1 (pas de test de permutation)
        +T1         : une seule application (pas de recursion)
        +code+rankC1 (sans retrecissement) : isole l'effet propre du retrecissement
      les lignes "+code" gardent la lecture par moindres carres du pipeline de base (un facteur a la fois) ;
        corrige, lecture centroides : retrecissement + code + C-1 avec la lecture par centroides (regle LDA),
        c'est le bloc corrige utilise dans le reste du rapport
  Mesures geometriques : angle principal max entre span(U_hat) et span(mu_c - mu)
  (representation vraie), et entre span(W_hat) et span(Sigma^{-1}(mu_c - mu)) (mesure vraie),
  au rang du test de permutation (rang variable) ET au rang C-1 (comparaison a rang egal).
  Regime extreme (4 exemples par classe, n_train < d) en fin de script.
"""
import numpy as np
from numpy.linalg import norm, pinv, eigh, svd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from scipy.linalg import eigh as geigh, subspace_angles
from ropn_auto_nobp import make_multiclass, make_closed_form_gate, recurse_gated, apply_P


def fit_block(X, y, shrink=False, rank="perm", n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(y); C = len(classes); n, d = X.shape
    mu = X.mean(0)
    mus = np.stack([X[y == c].mean(0) for c in classes])
    Xc = X - mus[np.searchsorted(classes, y)]
    ns = np.array([np.sum(y == c) for c in classes])
    S_B = ((mus - mu).T * ns) @ (mus - mu) / max(C - 1, 1)
    S_W = LedoitWolf().fit(Xc).covariance_ if shrink else (Xc.T @ Xc) / max(n - C, 1)
    reg = 1e-10 * np.trace(S_W) / d * np.eye(d)
    evals, evecs = geigh(S_B, S_W + reg)
    order = np.argsort(evals)[::-1]; evals, evecs = evals[order], evecs[:, order]
    r_struct = min(C - 1, d)
    if rank == "perm":
        null_max = np.empty(n_perm)
        for i in range(n_perm):
            yp = rng.permutation(y)
            musp = np.stack([X[yp == c].mean(0) for c in classes])
            S_Bp = ((musp - mu).T * ns) @ (musp - mu) / max(C - 1, 1)
            null_max[i] = geigh(S_Bp, S_W + reg, eigvals_only=True).max()
        r_star = max(1, int(min(r_struct, np.sum(evals[:r_struct] > np.quantile(null_max, 0.95)))))
    else:
        r_star = r_struct
    V_r = evecs[:, :r_star]; W_raw = V_r; U_raw = S_W @ V_r   # convention : w = S_W^{-1} u
    G = W_raw.T @ U_raw; gv, gV = eigh(G); gv = np.clip(gv, 1e-12, None)
    Gmh = gV @ np.diag(gv ** -0.5) @ gV.T
    return U_raw @ Gmh, W_raw @ Gmh, r_star


def readout_fit_predict(Ftr, ytr, Fte, ridge=0.0):
    classes = np.unique(ytr); Y = np.eye(len(classes))[ytr]
    A = np.c_[Ftr, np.ones(len(Ftr))]
    if ridge > 0:
        R = ridge * np.eye(A.shape[1]); R[-1, -1] = 0
        Wro = np.linalg.solve(A.T @ A + R, A.T @ Y)
    else:
        Wro, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return classes[(np.c_[Fte, np.ones(len(Fte))] @ Wro).argmax(1)]


def centroid_readout_weights(Ctr, ytr, priors=True):
    """Lecture par centroides ecrite comme une application AFFINE du code : scores = [c, 1] @ Wro, avec
    Wro = [M^T ; -||m_k||^2 / 2 + log pi_k] de forme (r+1, C) (M : centroides de classe dans l'espace du code).
    Meme nombre de coefficients que la lecture par moindres carres, mais aucun ajustement.
    Retourne (classes, Wro)."""
    classes, yi = np.unique(ytr, return_inverse=True)
    M = np.stack([Ctr[yi == k].mean(0) for k in range(len(classes))])
    bias = -0.5 * np.sum(M ** 2, 1)
    if priors: bias = bias + np.log(np.bincount(yi) / len(yi))
    return classes, np.vstack([M.T, bias[None, :]])


def readout_centroid_fit_predict(Ctr, ytr, Cte, priors=True):
    """Lecture analytique par centroides dans l'espace du code : score_k = -||c - m_k||^2 / 2 + log pi_k.
    Pour le bloc (W = S^{-1} D Q Lambda^{-1/2}) on a W^T S W = I_r : le code est blanchi par rapport a la
    covariance intra-classe S, et cette regle est EXACTEMENT la regle de la LDA de covariance S (rang C-1).
    Aucun ajustement : r C nombres (les centroides) et C a priori. C'est la lecture de reference du bloc."""
    classes, Wro = centroid_readout_weights(Ctr, ytr, priors)
    return classes[(np.c_[Cte, np.ones(len(Cte))] @ Wro).argmax(1)]


def block_predict(U, W, Xtr, ytr, Xte, T=4, feat="h", ridge=0.0, readout="ls"):
    """readout="ls" : moindres carres (lecture du pipeline de base, objet du diagnostic) ;
    readout="centroid" : lecture par centroides sur le code (regle LDA, lecture de reference)."""
    s, b = make_closed_form_gate(U, W, Xtr)
    Htr = recurse_gated(U, W, s, b, Xtr, T); Hte = recurse_gated(U, W, s, b, Xte, T)
    if feat == "code":
        Htr, Hte = Htr @ W, Hte @ W          # code oblique c = W^T h (dim r)
    if readout == "centroid":
        assert feat == "code", "la lecture par centroides n'a le sens de la regle LDA que sur le code"
        return readout_centroid_fit_predict(Htr, ytr, Hte)
    return readout_fit_predict(Htr, ytr, Hte, ridge)


def oracle_bayes(Sigma, mus, Xte):
    Sinv = pinv(Sigma)
    scores = np.stack([Xte @ Sinv @ m - 0.5 * m @ Sinv @ m for m in mus], 1)
    return scores.argmax(1)


def true_subspaces(Sigma, mus):
    mu = mus.mean(0)
    D = (mus - mu).T                       # (d, C) directions de moyennes (rang C-1)
    Uq, s, _ = svd(D, full_matrices=False); U_true = Uq[:, :mus.shape[0] - 1]   # span des moyennes
    W_true = pinv(Sigma) @ U_true          # directions de mesure de Fisher vraies : Sigma^{-1} u
    return U_true, W_true


def run(kappa, n_per, seeds=20, d=20, C=4, sep=0.8):
    keys = ["oracle", "LDA-shrink", "MLP", "base", "+shrink", "+code", "+rankC1", "+T1",
            "+shrink+code", "+code+rankC1", "+shrink+code+rankC1", "+shrink+code+rankC1+T1",
            "corrige, lecture centroides"]
    geo = {"S_W brut, rang perm.": (False, "perm"), "Ledoit-Wolf, rang perm.": (True, "perm"),
           "S_W brut, rang C-1": (False, "C1"), "Ledoit-Wolf, rang C-1": (True, "C1")}
    acc = {k: [] for k in keys}; angU = {k: [] for k in geo}; angW = {k: [] for k in geo}
    ranks = {k: [] for k in geo}
    for s in range(seeds):
        X, y, Sigma, mus = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=kappa, sep=sep, seed=s)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
        U_true, W_true = true_subspaces(Sigma, mus)
        acc["oracle"].append((oracle_bayes(Sigma, mus, Xte) == yte).mean())
        acc["LDA-shrink"].append(LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).score(Xte, yte))
        acc["MLP"].append(MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=s).fit(Xtr, ytr).score(Xte, yte))

        cfg = {
            "base":       dict(shrink=False, rank="perm", T=4, feat="h"),
            "+shrink":    dict(shrink=True,  rank="perm", T=4, feat="h"),
            "+code":      dict(shrink=False, rank="perm", T=4, feat="code"),
            "+rankC1":    dict(shrink=False, rank="C1",   T=4, feat="h"),
            "+T1":        dict(shrink=False, rank="perm", T=1, feat="h"),
            "+shrink+code":            dict(shrink=True, rank="perm", T=4, feat="code"),
            "+code+rankC1":            dict(shrink=False, rank="C1",  T=4, feat="code"),
            "+shrink+code+rankC1":     dict(shrink=True, rank="C1",   T=4, feat="code"),
            "+shrink+code+rankC1+T1":  dict(shrink=True, rank="C1",   T=1, feat="code"),
            # bloc corrige avec la lecture de reference (regle LDA) ; T sans effet : le code est invariant
            "corrige, lecture centroides": dict(shrink=True, rank="C1", T=4, feat="code", readout="centroid"),
        }
        cache = {}
        for k, c in cfg.items():
            key = (c["shrink"], c["rank"])
            if key not in cache:
                cache[key] = fit_block(Xtr, ytr, shrink=c["shrink"], rank=c["rank"], seed=s)
            U, W, r = cache[key]
            acc[k].append((block_predict(U, W, Xtr, ytr, Xte, T=c["T"], feat=c["feat"], readout=c.get("readout", "ls")) == yte).mean())
        for k, key in geo.items():
            U, W, r = cache[key]
            angU[k].append(np.degrees(subspace_angles(U, U_true).max()))
            angW[k].append(np.degrees(subspace_angles(W, W_true).max()))
            ranks[k].append(r)
    print(f"\n=== kappa={kappa:.0f}, n={n_per}/classe (n_train={int(0.7*n_per*C)}), d={d}, C={C}, {seeds} graines ===")
    for k in keys:
        print(f"  {k:26s} acc = {np.mean(acc[k])*100:5.1f} +/- {np.std(acc[k])*100:4.1f}")
    for k in angU:
        print(f"  geometrie {k:24s}: angle max span(U_hat)/span(U_vrai) = {np.mean(angU[k]):5.1f} deg | "
              f"span(W_hat)/span(W_vrai) = {np.mean(angW[k]):5.1f} deg | r* = {np.mean(ranks[k]):.2f}")
    from ropn_auto_stats import fmt_paired
    ref = acc["corrige, lecture centroides"]
    print("  differences appariees au bloc corrige, lecture par centroides [IC 95 %] :")
    for k in ("oracle", "LDA-shrink", "MLP", "+shrink+code+rankC1"):
        print(f"    {k:26s} - bloc corrige (centroides) : {fmt_paired(acc[k], ref)}")
    return acc


if __name__ == "__main__":
    for kappa in [10.0, 200.0]:
        run(kappa, 15)
    print("\n--- controle : regime abondant (n=200/classe), kappa=200 ---")
    run(200.0, 200, seeds=10)

    # regime extreme : 4 exemples par classe, n_train = 11 < d = 20 (apres decoupage 70/30 : 5 points de test par graine)
    from sklearn.neighbors import NearestCentroid
    print("\n--- regime extreme : 4 exemples/classe (n_train = 11 < d = 20), 20 graines, 5 points de test par graine ---")
    for kappa in (10.0, 200.0):
        a = {k: [] for k in ("bloc corrige", "bloc corrige (MC)", "MLP", "LDA-shrink", "centroide")}
        for s in range(20):
            X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=4, kappa=kappa, sep=0.8, seed=s)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=s)
            U, W, r = fit_block(Xtr, ytr, shrink=True, rank="C1", seed=s)
            a["bloc corrige"].append((readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W) == yte).mean())
            a["bloc corrige (MC)"].append((readout_fit_predict(Xtr @ W, ytr, Xte @ W) == yte).mean())
            a["MLP"].append(MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=s).fit(Xtr, ytr).score(Xte, yte))
            a["LDA-shrink"].append(LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).score(Xte, yte))
            a["centroide"].append(NearestCentroid().fit(Xtr, ytr).score(Xte, yte))
        print(f"  kappa={kappa:5.0f} (n_train={len(ytr)}, n_test={len(yte)}) : " + "  ".join(f"{k} {100*np.mean(v):.1f} +/- {100*np.std(v):.1f}" for k, v in a.items()))
