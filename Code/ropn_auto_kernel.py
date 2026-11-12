
"""
Bloc oblique A NOYAU (piste "cachee" dans la forme duale) et comparaison aux alternatives
classiques du MLP (SVM-RBF, gradient boosting, k-NN, LDA-shrinkage).

La forme duale de ropn_auto_fast (Woodbury sur la Gram Xc Xc^T) ne depend des donnees que
par des produits scalaires : remplacer la Gram par un noyau K = k(x_i, x_j) donne, en forme
fermee, le meme bloc oblique dans l'espace de caracteristiques phi :
    representation U = Phi^T B          (moyennes de classe dans l'espace phi, rang C-1)
    mesure         W = S^{-1} U = Phi^T A,  S = s Phi_c^T Phi_c + c I   (Woodbury -> n x n)
    code           W^T phi(x) = alpha^T k(X, x)
C'est l'analyse discriminante de Fisher a noyau (Mika et al. 1999) relue comme projecteur
oblique : W^T U = I_r automatiquement, idempotence exacte, rang C-1, capacite gouvernee par n
(pas par la profondeur), zero gradient. Hyperparametres : gamma (heuristique de la mediane
x facteur) et ridge, choisis par validation croisee interne.
"""
import numpy as np
from numpy.linalg import eigh
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def rbf(A, B, gamma):
    return np.exp(-gamma * cdist(A, B, "sqeuclidean"))


class KernelObliqueBlock:
    def __init__(self, gamma_scale=1.0, ridge=1e-2):
        self.gamma_scale, self.ridge = gamma_scale, ridge

    def fit(self, X, y):
        n = len(X)
        self.X = X
        self.classes, yi = np.unique(y, return_inverse=True); C = len(self.classes)
        ns = np.bincount(yi, minlength=C).astype(float)
        d2 = cdist(X, X, "sqeuclidean")
        self.gamma = self.gamma_scale / np.median(d2[d2 > 0])
        K = np.exp(-self.gamma * d2)
        Y = np.zeros((n, C)); Y[np.arange(n), yi] = 1.0
        H = Y @ np.diag(1.0 / ns) @ Y.T                      # projecteur sur les moyennes de classe
        I = np.eye(n); Hc = I - H
        Kc = Hc @ K @ Hc                                       # Gram des residus intra-classe (Phi_c Phi_c^T)
        B = (I - np.ones((n, n)) / n) @ Y / np.sqrt(ns) / np.sqrt(max(C - 1, 1))   # D = Phi^T B
        s = 1.0 / n; c = self.ridge * np.trace(Kc) / n         # S = s Phi_c^T Phi_c + c I
        cf = cho_factor(c * I + s * Kc, lower=True)
        # S^{-1} D = Phi^T [ B/c - (s/c) Hc (cI + sKc)^{-1} Hc K B ]
        A = B / c - (s / c) * (Hc @ cho_solve(cf, Hc @ (K @ B)))
        M = B.T @ K @ A; M = 0.5 * (M + M.T)                  # D^T S^{-1} D  (C x C)
        lam, Q = eigh(M); o = np.argsort(lam)[::-1]; lam, Q = lam[o], Q[:, o]
        r = C - 1; lr = np.clip(lam[:r], 1e-12, None)
        self.alpha = A @ Q[:, :r] / np.sqrt(lr)                # W = Phi^T alpha
        self.beta = B @ Q[:, :r] / np.sqrt(lr)                 # U = Phi^T beta
        self.WtU = self.alpha.T @ K @ self.beta                # doit valoir I_r
        F = K @ self.alpha
        from ropn_auto_nobp_diag import centroid_readout_weights
        _, self.Wro = centroid_readout_weights(F, yi)                   # lecture par centroides sur le code noyau
        return self

    def predict(self, X):
        F = rbf(X, self.X, self.gamma) @ self.alpha
        return self.classes[(np.c_[F, np.ones(len(F))] @ self.Wro).argmax(1)]


def fit_kernel_auto(Xtr, ytr, gammas=(0.25, 1.0, 4.0), ridges=(1e-3, 1e-2, 1e-1), n_folds=3, seed=0):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best, cfg = -1, None
    for g in gammas:
        for r in ridges:
            accs = [(KernelObliqueBlock(g, r).fit(Xtr[tr], ytr[tr]).predict(Xtr[va]) == ytr[va]).mean()
                    for tr, va in skf.split(Xtr, ytr)]
            if np.mean(accs) > best + 1e-9: best, cfg = np.mean(accs), (g, r)
    return KernelObliqueBlock(*cfg).fit(Xtr, ytr), cfg


if __name__ == "__main__":
    import time, warnings; warnings.filterwarnings("ignore")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_digits, load_breast_cancer, load_wine
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from ropn_auto_nobp import make_multiclass
    from ropn_auto_finetune import make_hetero
    from ropn_auto_fast import fit_block_fast
    from ropn_auto_nobp_diag import readout_fit_predict, readout_centroid_fit_predict
    from ropn_auto_lattice import fit_lattice_auto

    def prep(X, y, sd):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
        keep = Xtr.std(0) > 1e-8; sc = StandardScaler().fit(Xtr[:, keep])
        return sc.transform(Xtr[:, keep]), sc.transform(Xte[:, keep]), ytr, yte

    ds = {}
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=10.0, sep=0.8, seed=0); ds["gauss k=10 n=15/cl"] = (X, y)
    X, y, *_ = make_hetero(n_classes=4, d=20, n_per=60, kappa=50.0, sep=0.8, seed=0);       ds["hetero k=50 n=60/cl"] = (X, y)
    d_ = load_digits(); ds["digits 8x8"] = (d_.data, d_.target)
    b_ = load_breast_cancer(); ds["breast cancer"] = (b_.data, b_.target)
    w_ = load_wine(); ds["wine"] = (w_.data, w_.target)

    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

    def svm_cv(Xtr, ytr, Xte, sd):
        """SVM RBF, (C, gamma) par validation croisee interne (3 plis)."""
        d = Xtr.shape[1]; grid = {"C": [0.1, 1.0, 10.0, 100.0], "gamma": [0.3 / d, 1.0 / d, 3.0 / d]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=sd)
        return GridSearchCV(SVC(kernel="rbf"), grid, cv=cv).fit(Xtr, ytr).predict(Xte)

    def qda_cv(Xtr, ytr, Xte, sd):
        """QDA regularisee (reg_param de sklearn), choisie par validation croisee interne (3 plis)."""
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=sd)
        return GridSearchCV(QDA(), {"reg_param": [0.01, 0.03, 0.1, 0.3, 0.6, 0.9]}, cv=cv).fit(Xtr, ytr).predict(Xte)

    methods = {
        "bloc fast (lineaire)": lambda Xtr, ytr, Xte, sd: (lambda U, W, r, _: readout_fit_predict(Xtr @ W, ytr, Xte @ W))(*fit_block_fast(Xtr, ytr, shrink=True, rank="C1")),
        "bloc fast, lecture centroide": lambda Xtr, ytr, Xte, sd: (lambda U, W, r, _: readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W))(*fit_block_fast(Xtr, ytr, shrink=True, rank="C1")),
        "treillis AUTO":        lambda Xtr, ytr, Xte, sd: fit_lattice_auto(Xtr, ytr, seed=sd)[0].predict(Xte),
        "bloc oblique a NOYAU (CV)": lambda Xtr, ytr, Xte, sd: fit_kernel_auto(Xtr, ytr, seed=sd)[0].predict(Xte),
        "LDA-shrinkage":        lambda Xtr, ytr, Xte, sd: LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte),
        "SVM RBF (C=1)":        lambda Xtr, ytr, Xte, sd: SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr, ytr).predict(Xte),
        "SVM RBF (C=10)":       lambda Xtr, ytr, Xte, sd: SVC(kernel="rbf", C=10.0, gamma="scale").fit(Xtr, ytr).predict(Xte),
        "SVM RBF (CV interne)": svm_cv,
        "QDA regularisee (CV interne)": qda_cv,
        "HistGradientBoosting": lambda Xtr, ytr, Xte, sd: HistGradientBoostingClassifier(random_state=sd).fit(Xtr, ytr).predict(Xte),
        "k-NN (k=5)":           lambda Xtr, ytr, Xte, sd: KNeighborsClassifier(5).fit(Xtr, ytr).predict(Xte),
        "MLP (backprop)":       lambda Xtr, ytr, Xte, sd: MLPClassifier(hidden_layer_sizes=(int(np.clip(2 * Xtr.shape[1], 32, 256)),), max_iter=500, random_state=sd).fit(Xtr, ytr).predict(Xte),
    }
    SEEDS = 5
    for dname, (X, y) in ds.items():
        print(f"\n=== {dname} (n={X.shape[0]}, d={X.shape[1]}, C={len(np.unique(y))}, {SEEDS} decoupages 70/30 d'un meme tirage) ===")
        for mname, fn in methods.items():
            accs, ts = [], []
            try:
                for sd in range(SEEDS):
                    Xtr, Xte, ytr, yte = prep(X, y, sd)
                    t0 = time.perf_counter(); pred = fn(Xtr, ytr, Xte, sd); ts.append(time.perf_counter() - t0)
                    accs.append((pred == yte).mean())
            except Exception as e:                      # p. ex. QDA quand une classe a moins d'exemples que de variables
                print(f"  {mname:27s} echec : {type(e).__name__}"); continue
            print(f"  {mname:30s} acc = {np.mean(accs)*100:5.1f} +/- {np.std(accs)*100:4.1f} | {np.mean(ts)*1000:7.0f} ms")
    # sanity : biorthogonalite dans l'espace de caracteristiques
    Xtr, Xte, ytr, yte = prep(*ds["digits 8x8"], 0)
    m = KernelObliqueBlock().fit(Xtr, ytr)
    print(f"\n[noyau] ||W^T U - I_r|| dans l'espace phi = {np.linalg.norm(m.WtU - np.eye(m.WtU.shape[0])):.1e}")
