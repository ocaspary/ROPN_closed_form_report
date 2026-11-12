"""
ROPN-auto sans back-propagation : bloc unique a rang selectionne, applique par recursion.

Ce script :
  1) construit un projecteur oblique idempotent P = U (W^T U)^{-1} W^T en forme fermee
     (GEVP de Fisher multi-classe S_B v = lambda S_W v, sans descente de gradient) ;
  2) verifie numeriquement le theoreme de collapse : l'application recursive de L copies
     du MEME bloc lineaire idempotent est EXACTEMENT egale a une seule application du bloc
     avec un coefficient scalaire c_L = (1+1/L)^L - 1 ;
  3) etend le bloc avec une porte oblique non-lineaire (elle aussi fixee en forme fermee) ;
     NB : le code W^T h_T est invariant par la recursion, et avec s dans span(W) la
     porte collapse aussi : h_T = P x + (1-a(x))^T (I-P) x  (proposition d'invariance du code) ;
  4) selectionne le rang automatiquement (borne structurelle C-1 + test de permutation) ;
  5) compare accuracy / #parametres a un MLP entraine par retropropagation (sklearn).
Ce script produit la demonstration de base (collapse, bloc de reference, comparaison a un MLP).
"""
import numpy as np
from numpy.linalg import qr, eigh, norm, pinv
from scipy.linalg import eigh as geigh
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------------
# 0. Donnees synthetiques anisotropes multi-classes (kappa(S_W) controle)
# ----------------------------------------------------------------------------
def make_multiclass(n_classes=4, d=20, n_per=60, kappa=200.0, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, np.log10(kappa), d); eig = eig / eig.mean()
    Q, _ = qr(rng.standard_normal((d, d)))
    Sigma = Q @ np.diag(eig) @ Q.T
    L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
    mus = sep * rng.standard_normal((n_classes, d))
    mus -= mus.mean(0, keepdims=True)  # centrer les moyennes de classe
    X = np.zeros((n_classes * n_per, d))
    y = np.zeros(n_classes * n_per, dtype=int)
    for c in range(n_classes):
        Z = rng.standard_normal((n_per, d)) @ L.T
        X[c*n_per:(c+1)*n_per] = Z + mus[c]
        y[c*n_per:(c+1)*n_per] = c
    return X, y, Sigma, mus


# ----------------------------------------------------------------------------
# 1. Construction fermee du bloc oblique idempotent (Fisher multi-classe, M-orthogonal)
# ----------------------------------------------------------------------------
def fit_oblique_block_closed_form(X, y, r=None, shrink="auto", perm_test=True, n_perm=200, seed=0):
    """Construit P = U (W^T U)^{-1} W^T, idempotent, W^T U = I_r, EN FORME FERMEE.

    Methode : GEVP de Fisher S_B v = lambda S_W v (LDA multi-classe generalisee a l'oblique).
    W = les r premiers vecteurs propres generalises (directions de MESURE, w_i = S_W^{-1} u_i),
    U = S_W W (directions de REPRESENTATION, dans span des moyennes de classe), puis
    reblanchiment pour imposer W^T U = I_r exactement (idempotence, kappa(G)=1 par construction).
    Les vecteurs propres generalises de scipy etant deja S_W-orthonormes, le reblanchiment ne
    corrige que le petit terme de regularisation et les arrondis.
    Aucune retropropagation, aucun optimiseur : uniquement algebre lineaire (eigh, solve, QR).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    C, n, d = len(classes), *X.shape
    mu = X.mean(0)
    mus = np.stack([X[y == c].mean(0) for c in classes])

    # Scatter intra-classe (poolee) et inter-classe
    Xc = X - mus[np.searchsorted(classes, y)]
    S_W = (Xc.T @ Xc) / max(n - C, 1)
    ns = np.array([np.sum(y == c) for c in classes])
    S_B = ((mus - mu).T * ns) @ (mus - mu) / max(C - 1, 1)

    # Shrinkage Ledoit-Wolf si regime pauvre en donnees (n <= d), sinon S_W brut
    # (variante de reference ; le bloc corrige applique le retrecissement systematiquement)
    used_shrink = False
    if shrink == "auto":
        shrink = (n <= d)
    if shrink:
        S_W = LedoitWolf().fit(Xc).covariance_
        used_shrink = True

    # GEVP S_B v = lambda S_W v  (borne structurelle : rang(S_B) <= C-1)
    evals, evecs = geigh(S_B, S_W + 1e-10 * np.trace(S_W) / d * np.eye(d))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]

    # ---- selection automatique du rang (sans backprop) --------------------
    r_struct = min(C - 1, d)
    if r is not None:
        r_star = r
    elif perm_test:
        null_max = np.empty(n_perm)
        for i in range(n_perm):
            yp = rng.permutation(y)
            musp = np.stack([X[yp == c].mean(0) for c in classes])
            S_Bp = ((musp - mu).T * ns) @ (musp - mu) / max(C - 1, 1)
            ev = geigh(S_Bp, S_W + 1e-10 * np.trace(S_W) / d * np.eye(d), eigvals_only=True)
            null_max[i] = ev.max()
        thresh = np.quantile(null_max, 0.95)
        r_star = int(min(r_struct, np.sum(evals[:r_struct] > thresh)))
        r_star = max(r_star, 1)
    else:
        r_star = r_struct

    # Convention de Fisher (rang 1 : u = mu_1 - mu_0, w* = S_W^{-1} u), generalisee :
    #   les vecteurs propres generalises v_i sont les directions de MESURE de Fisher
    #   (v_i = S_W^{-1} u_i), et u_i = S_W v_i = S_B v_i / lambda_i appartient a
    #   span(mu_c - mu) : ce sont les directions de REPRESENTATION (moyennes de classe).
    #   Le code c = G^{-1} W^T h est alors la statistique suffisante de la LDA.
    V_r = evecs[:, :r_star]
    W_raw = V_r                # mesure : w_i = v_i = S_W^{-1} u_i
    U_raw = S_W @ V_r          # representation : u_i = S_W v_i  (dans span des moyennes)

    # Reblanchiment pour imposer EXACTEMENT W^T U = I_r (forme idempotente)
    G = W_raw.T @ U_raw                      # (r,r) SPD
    Gvals, Gvecs = eigh(G)
    Gvals = np.clip(Gvals, 1e-12, None)
    G_mhalf = Gvecs @ np.diag(Gvals ** -0.5) @ Gvecs.T
    U = U_raw @ G_mhalf
    W = W_raw @ G_mhalf

    diagnostics = dict(r_star=r_star, r_struct=r_struct, eigenvalues=evals,
                        shrink=used_shrink, WtU_err=norm(W.T @ U - np.eye(r_star)))
    return U, W, diagnostics


def apply_P(U, W, h):
    return (h @ W) @ U.T


# ----------------------------------------------------------------------------
# 2. Theoreme de collapse : (I + P/L)^L = I + [(1+1/L)^L - 1] P  (P idempotent)
# ----------------------------------------------------------------------------
def recurse_linear(U, W, h, L):
    for _ in range(L):
        h = h + apply_P(U, W, h) / L
    return h


def collapse_closed_form(U, W, h, L):
    c_L = (1.0 + 1.0 / L) ** L - 1.0
    return h + c_L * apply_P(U, W, h)


# ----------------------------------------------------------------------------
# 3. Porte oblique non-lineaire fixee en forme fermee (pas de backprop)
# ----------------------------------------------------------------------------
def make_closed_form_gate(U, W, X):
    """s = direction de Fisher dominante (colonne de W), b centre les deux classes
    de score sur cette direction. Purement algebrique, aucune optimisation."""
    s = W[:, 0] / norm(W[:, 0])
    proj = X @ s
    b = -proj.mean()
    return s, b


def sigmoid(z):
    z = np.clip(z, -60.0, 60.0)  # evite l'overflow de exp pour |z| grand
    return 1.0 / (1.0 + np.exp(-z))


def recurse_gated(U, W, s, b, h, T):
    """Bloc PARTAGE applique T fois (poids identiques a chaque pas, comme un
    reseau recurrent / DEQ) : h <- (1-a(h))h + a(h) P h,  a(h)=sigma(s^T h + b)."""
    for _ in range(T):
        a = sigmoid(h @ s + b)[:, None]
        h = (1 - a) * h + a * apply_P(U, W, h)
    return h


if __name__ == "__main__":
    # ---- Donnees ----
    X, y, Sigma, mus = make_multiclass(n_classes=4, d=20, n_per=60, kappa=200.0, sep=2.0, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    # ---- Bloc ferme ----
    U, W, diag = fit_oblique_block_closed_form(Xtr, ytr, seed=0)
    print("=== Bloc auto-ROPN ferme (sans backprop) ===")
    print(f"rang structurel C-1 = {diag['r_struct']}, rang auto-selectionne r* = {diag['r_star']}")
    print(f"valeurs propres GEVP (top 6) = {np.round(diag['eigenvalues'][:6], 3)}")
    print(f"||W^T U - I_r|| = {diag['WtU_err']:.2e}  (idempotence exacte attendue)")
    P = U @ W.T
    print(f"||P^2 - P||_F = {norm(P @ P - P):.2e}")

    # ---- Theoreme de collapse (verification numerique) ----
    print("\n=== Verification du theoreme de collapse ===")
    h0 = rng.standard_normal((5, X.shape[1]))
    for L in [1, 2, 4, 8, 16, 64, 256]:
        h_iter = recurse_linear(U, W, h0.copy(), L)
        h_closed = collapse_closed_form(U, W, h0.copy(), L)
        err = norm(h_iter - h_closed) / norm(h_closed)
        c_L = (1 + 1 / L) ** L - 1
        print(f"L={L:4d}  c_L={c_L:.6f}  erreur relative iteratif-vs-ferme = {err:.2e}")

    # ---- Recursion non-lineaire : verifier que ca NE collapse PAS trivialement ----
    print("\n=== Porte non-lineaire : evolution de h_T avec T (forme close : (1-a)^T) ===")
    s, b = make_closed_form_gate(U, W, Xtr)
    for T in [1, 2, 4, 8, 16]:
        hT = recurse_gated(U, W, s, b, h0.copy(), T)
        print(f"T={T:2d}  ||h_T - h_0|| = {norm(hT - h0):.4f}   ||h_T|| = {norm(hT):.4f}")

    # ---- Comparaison accuracy / parametres ----
    print("\n=== Comparaison accuracy / parametres ===")

    def closed_form_predict(U, W, s, b, Xtr, ytr, Xte, T):
        Htr = recurse_gated(U, W, s, b, Xtr, T)
        Hte = recurse_gated(U, W, s, b, Xte, T)
        # tete de lecture fermee (moindres carres / ridge, PAS de backprop)
        classes = np.unique(ytr)
        Ytr = np.eye(len(classes))[ytr]
        Wro, *_ = np.linalg.lstsq(np.c_[Htr, np.ones(len(Htr))],
                                   Ytr, rcond=None)
        pred = np.c_[Hte, np.ones(len(Hte))] @ Wro
        return classes[pred.argmax(1)], Wro

    results = {}
    for T in [1, 4, 8]:
        pred, Wro = closed_form_predict(U, W, s, b, Xtr, ytr, Xte, T)
        acc = (pred == yte).mean()
        n_params_block = U.size  # W, s, b derives de U (pas des parametres libres supplementaires)
        n_params_head = Wro.size
        results[f"auto-ROPN-1bloc (T={T})"] = (acc, n_params_block + n_params_head, 0)

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)
    mlp.fit(Xtr, ytr)
    acc_mlp = mlp.score(Xte, yte)
    n_params_mlp = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
    results["MLP (backprop, 1 couche cachee=32)"] = (acc_mlp, n_params_mlp, n_params_mlp)

    print(f"{'modele':38s} {'acc':>8s} {'#param (tot)':>13s} {'#param (backprop)':>18s}")
    for k, (acc, npar, nbp) in results.items():
        print(f"{k:38s} {acc*100:7.1f}% {npar:13d} {nbp:18d}")
