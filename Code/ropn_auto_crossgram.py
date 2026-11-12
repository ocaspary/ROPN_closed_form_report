"""
Cross-Gram euclidien vs cross-Gram de Fisher.

Deux objets portent le nom de "cross-Gram" :
  (a) le couplage G = W^T U (r x r) du projecteur P = U G^{-1} W^T. Le projecteur ne depend que des
      sous-espaces (im U, ker W^T) : changer G ne change pas P (invariance de jauge). Remplacer G^{-1}
      par une matrice A quelconque donne U A W^T, carte de rang r NON idempotente (LoRA) : on perd
      les garanties sans gagner de sous-espace.
  (b) la covariance croisee entree/cible S_XY = X_c^T Y_c (d x C). Le bloc de Fisher = CCA(X, Y) :
      directions de mesure W = S_XX^{-1} S_XY ... (metrique intra-classe). Le cross-Gram EUCLIDIEN
      (sans metrique, W = U = vecteurs singuliers de S_XY) est la PLS-DA : projecteur ORTHOGONAL.
      Entre les deux : la metrique retrecie (S_W + lambda I)^{-1}, lambda -> infini donne PLS-DA.
Ce script mesure ou chaque extremite gagne (petits n, n << d, digits) et verifie la relation.
"""
import numpy as np
from numpy.linalg import svd
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from ropn_auto_nobp import make_multiclass
from ropn_auto_fast import fit_block_fast
from ropn_auto_nobp_diag import readout_centroid_fit_predict   # lecture de reference : centroides sur le code (regle LDA)


def fit_crossgram_block(X, y, lam, r=None):
    """Bloc oblique a metrique retrecie M = (S_W + lam * tr(S_W)/d I)^{-1} ; lam=inf -> PLS-DA (euclidien).
    Retourne U, W avec W^T U = I_r (reduction C x C exacte)."""
    classes, yi = np.unique(y, return_inverse=True); C = len(classes); n, d = X.shape
    ns = np.bincount(yi, minlength=C).astype(float); mu = X.mean(0)
    mus = np.stack([X[yi == k].mean(0) for k in range(C)])
    D = ((mus - mu).T * np.sqrt(ns)) / np.sqrt(max(C - 1, 1))          # S_B = D D^T,  D ~ S_XY N^{-1/2}
    Xc = X - mus[yi]; S_W = Xc.T @ Xc / max(n - C, 1)
    if np.isinf(lam):
        Z = D.copy()                                                    # metrique euclidienne : W = U
    else:
        S = S_W + lam * np.trace(S_W) / d * np.eye(d)
        Z = cho_solve(cho_factor(S, lower=True), D)
    M = D.T @ Z; M = 0.5 * (M + M.T)
    lamb, Q = np.linalg.eigh(M); o = np.argsort(lamb)[::-1]; lamb, Q = lamb[o], Q[:, o]
    r = C - 1 if r is None else r
    lr = np.clip(lamb[:r], 1e-12, None)
    return D @ Q[:, :r] / np.sqrt(lr), Z @ Q[:, :r] / np.sqrt(lr)


def acc_block(Xtr, ytr, Xte, yte, lam):
    U, W = fit_crossgram_block(Xtr, ytr, lam)
    return (readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W) == yte).mean()


def acc_plsda(Xtr, ytr, Xte, yte, n_comp):
    classes = np.unique(ytr); Y = np.eye(len(classes))[np.searchsorted(classes, ytr)]
    pls = PLSRegression(n_components=n_comp, scale=False).fit(Xtr, Y)
    return (classes[pls.predict(Xte).argmax(1)] == yte).mean()


if __name__ == "__main__":
    def prep(X, y, sd):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
        keep = Xtr.std(0) > 1e-8; sc = StandardScaler().fit(Xtr[:, keep])
        return sc.transform(Xtr[:, keep]), sc.transform(Xte[:, keep]), ytr, yte

    cases = {}
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=10.0, sep=0.8, seed=0);   cases["gauss d=20, n=15/cl, kappa=10"] = (X, y)
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=200.0, sep=0.8, seed=0);  cases["gauss d=20, n=15/cl, kappa=200"] = (X, y)
    X, y, *_ = make_multiclass(n_classes=4, d=500, n_per=15, kappa=50.0, sep=0.6, seed=0);  cases["gauss d=500, n=15/cl (n<<d)"] = (X, y)
    d_ = load_digits(); cases["digits 8x8"] = (d_.data, d_.target)

    # verification : lam -> inf du bloc = PLS-DA (memes sous-espaces)
    Xtr, Xte, ytr, yte = prep(*cases["gauss d=20, n=15/cl, kappa=10"], 0)
    U, W = fit_crossgram_block(Xtr, ytr, np.inf)
    print(f"[verif] cross-Gram euclidien : W^T U - I = {np.linalg.norm(W.T @ U - np.eye(W.shape[1])):.1e}, "
          f"||U - W||/||U|| = {np.linalg.norm(U - W)/np.linalg.norm(U):.1e} (projecteur orthogonal)")

    lams = [("LW auto (bloc fast)", None), ("lam=0.01", 0.01), ("lam=0.1", 0.1), ("lam=1", 1.0), ("lam=10", 10.0), ("euclidien (lam=inf)", np.inf)]

    def lam_by_cv(Xtr, ytr, sd, grid=(0.01, 0.1, 1.0, 10.0), n_folds=3):
        """lambda choisi par validation croisee INTERNE a l'apprentissage (aucun acces au test)."""
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=sd); best, bl = -1.0, grid[0]
        for lam in grid:
            a_ = np.mean([acc_block(Xtr[tr], ytr[tr], Xtr[va], ytr[va], lam) for tr, va in skf.split(Xtr, ytr)])
            if a_ > best + 1e-9: best, bl = a_, lam
        return bl

    print("\n(5 decoupages 70/30 stratifies d'un meme tirage ; les colonnes lam=... sont des lambda FIXES ; 'lam CV' est choisi par validation croisee interne)")
    print(f"\n{'cas':34s} " + " ".join(f"{n[:12]:>12s}" for n, _ in lams) + f" {'lam CV':>8s} {'PLS-DA(C-1)':>12s} {'MLP':>8s}")
    for cname, (X, y) in cases.items():
        res = {n: [] for n, _ in lams}; pls, mlp, cv = [], [], []
        for sd in range(5):
            Xtr, Xte, ytr, yte = prep(X, y, sd)
            for n_, lam in lams:
                if lam is None:
                    U, W, r, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")
                    res[n_].append((readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W) == yte).mean())
                else:
                    res[n_].append(acc_block(Xtr, ytr, Xte, yte, lam))
            cv.append(acc_block(Xtr, ytr, Xte, yte, lam_by_cv(Xtr, ytr, sd)))
            pls.append(acc_plsda(Xtr, ytr, Xte, yte, len(np.unique(ytr)) - 1))
            h = int(np.clip(2 * Xtr.shape[1], 32, 256))
            mlp.append(MLPClassifier(hidden_layer_sizes=(h,), max_iter=500, random_state=sd).fit(Xtr, ytr).score(Xte, yte))
        print(f"{cname:34s} " + " ".join(f"{np.mean(res[n])*100:12.1f}" for n, _ in lams) + f" {np.mean(cv)*100:8.1f} {np.mean(pls)*100:12.1f} {np.mean(mlp)*100:8.1f}")
