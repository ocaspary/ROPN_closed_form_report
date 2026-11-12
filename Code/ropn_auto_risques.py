"""
Audit des sept risques classiques d'un bloc oblique a rang selectionne, applique au bloc analytique :
  R1 amplification du bruit        : ||P||_2, precision sous bruit d'entree et hors distribution ;
  R2 sous-espaces presque degeneres: sigma_min(Q_W^T Q_U) (= 1/||P||_2) ;
  R3 perte de capacite             : rang retenu contre rangs fixes (la reinjection est traitee dans la section petits echantillons) ;
  R4 instabilite en profondeur     : conditionnement du Jacobien GLOBAL de T pas (lineaire, porte non
                                     normalisee h <- (1-a) h + a P h, porte normalisee h <- h + (a/T)(P - I) h) ;
  R5 selection instable : variabilite du rang (test corrige) et de la configuration du treillis
                                     entre graines ;
  R6 gain attribue a tort a l'obliquite : variante orthogonale et variante bas rang non contrainte, meme
                                     rang, meme lecture ;
  R7 regularisation qui modifie le modele : ||P^2 - P||, et ||P_LW - P_brut|| / ||P_brut|| en fonction de n.
numpy / scipy / sklearn seulement.
"""
import numpy as np
from numpy.linalg import norm, qr, svd
from sklearn.model_selection import train_test_split
from ropn_auto_nobp import make_multiclass, make_closed_form_gate, recurse_gated, apply_P
from ropn_auto_fast import fit_block_fast
from ropn_auto_nobp_diag import readout_fit_predict, readout_centroid_fit_predict
from ropn_auto_lattice import fit_lattice_auto


def split(X, y, sd): return train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)


def fit_orthogonal(X, y, r):
    """Variante orthogonale de meme rang : U = W = base orthonormee des r premieres directions
    inter-classes (vecteurs singuliers de D). P = Q Q^T."""
    classes, yi = np.unique(y, return_inverse=True); ns = np.bincount(yi).astype(float); mu = X.mean(0)
    mus = np.stack([X[yi == k].mean(0) for k in range(len(classes))]); D = ((mus - mu).T * np.sqrt(ns))
    Q, _, _ = svd(D, full_matrices=False); Q = Q[:, :r]; return Q, Q


def fit_lowrank_unconstrained(X, y, r, lam=1e-2):
    """Variante bas rang NON contrainte (pas de projecteur) : regression de rang reduit de l'indicatrice
    de classe sur X (ridge), tronquee au rang r ; code = X @ W_r, meme lecture que le bloc."""
    classes, yi = np.unique(y, return_inverse=True); Y = np.eye(len(classes))[yi]; Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    B = np.linalg.solve(Xc.T @ Xc + lam * np.trace(Xc.T @ Xc) / X.shape[1] * np.eye(X.shape[1]), Xc.T @ Yc)
    Uy, s, Vt = svd(Xc @ B, full_matrices=False); return B @ Vt[:r].T                 # (d, r)


def acc(Ftr, ytr, Fte, yte, readout="centroid"):
    """lecture de reference : centroides sur le code (regle LDA pour le bloc) ; "ls" : moindres carres."""
    f = readout_centroid_fit_predict if readout == "centroid" else readout_fit_predict
    return float((f(Ftr, ytr, Fte) == yte).mean())


if __name__ == "__main__":
    C, d, seeds = 4, 20, 20
    print("=== R1/R2/R6 : norme de P, angle entre sous-espaces, robustesse, et variantes de controle (C=4, d=20, 20 graines) ===")
    print("    meme lecture pour les trois variantes : lecture par centroides sur le code (tableau principal), puis")
    print("    lecture par moindres carres sur le code (controle de sensibilite a la lecture)")
    print(f"{'cas':24s} {'||P||_2':>8s} {'s_min(QwQu)':>11s} {'1/s_min':>8s} | precision (%) : oblique / orthogonal / bas-rang libre")
    print(f"{'':24s} {'':8s} {'':11s} {'':8s} |  propre    bruit 0.5   bruit 1.0   dilatation x2")
    tab_ls = []
    for kappa in (10.0, 200.0):
        for n_per in (15, 100):
            Pn, smin, res = [], [], {k: [] for k in ("obl", "ort", "lr", "obl_ls", "ort_ls", "lr_ls")}
            for sd in range(seeds):
                X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=kappa, sep=0.8, seed=sd)
                Xtr, Xte, ytr, yte = split(X, y, sd); rng = np.random.default_rng(100 + sd)
                U, W, r, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); P = U @ W.T
                Pn.append(norm(P, 2)); Qw, Qu = qr(W)[0], qr(U)[0]; smin.append(svd(Qw.T @ Qu, compute_uv=False).min())
                Qo, _ = fit_orthogonal(Xtr, ytr, r); Wl = fit_lowrank_unconstrained(Xtr, ytr, r)
                sig = Xtr.std(0).mean()
                tests = [Xte, Xte + 0.5 * sig * rng.standard_normal(Xte.shape), Xte + 1.0 * sig * rng.standard_normal(Xte.shape),
                         Xte.mean(0) + 2.0 * (Xte - Xte.mean(0))]                  # bruit isotrope ; nuage de test dilate d'un facteur 2 autour de sa moyenne
                res["obl"].append([acc(Xtr @ W, ytr, T @ W, yte) for T in tests])
                res["ort"].append([acc(Xtr @ Qo, ytr, T @ Qo, yte) for T in tests])
                res["lr"].append([acc(Xtr @ Wl, ytr, T @ Wl, yte) for T in tests])
                res["obl_ls"].append([acc(Xtr @ W, ytr, T @ W, yte, "ls") for T in tests])
                res["ort_ls"].append([acc(Xtr @ Qo, ytr, T @ Qo, yte, "ls") for T in tests])
                res["lr_ls"].append([acc(Xtr @ Wl, ytr, T @ Wl, yte, "ls") for T in tests])
            m = {k: 100 * np.mean(v, 0) for k, v in res.items()}
            print(f"kappa={kappa:5.0f} n/cl={n_per:3d}      {np.mean(Pn):8.2f} {np.mean(smin):11.3f} {1/np.mean(smin):8.2f} | "
                  + "  ".join(f"{m['obl'][j]:4.1f}/{m['ort'][j]:4.1f}/{m['lr'][j]:4.1f}" for j in range(4)))
            tab_ls.append(f"kappa={kappa:5.0f} n/cl={n_per:3d}      lecture MC | "
                          + "  ".join(f"{m['obl_ls'][j]:4.1f}/{m['ort_ls'][j]:4.1f}/{m['lr_ls'][j]:4.1f}" for j in range(4)))
    for line in tab_ls: print(line)
    print("    (||P||_2 = 1/sigma_min(Q_W^T Q_U) exactement : les deux colonnes sont le meme risque ; la norme euclidienne de P")
    print("     est 1/sin de l'angle minimal entre im P et ker P, et vaut 1 dans la metrique S_W^{-1} ou P est orthogonal.)")

    print("\n=== R3 : rang retenu contre rangs fixes (kappa=200, sep=0.8, 20 graines) ===")
    for n_per in (5, 15, 100):
        rows = {r: [] for r in (1, 2, 3)}; auto = []; rs = []
        for sd in range(seeds):
            X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=200.0, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
            U, W, r, info = fit_block_fast(Xtr, ytr, shrink=True, rank="perm", seed=sd); rs.append(info["r_tested"])
            for rf in rows:
                Uf, Wf, _, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); rows[rf].append(acc(Xtr @ Wf[:, :rf], ytr, Xte @ Wf[:, :rf], yte))
            auto.append(acc(Xtr @ W, ytr, Xte @ W, yte))
        print(f"  n/cl={n_per:3d} : r=1 {100*np.mean(rows[1]):.1f}  r=2 {100*np.mean(rows[2]):.1f}  r=3 (C-1) {100*np.mean(rows[3]):.1f}  | auto {100*np.mean(auto):.1f}  rang teste moyen {np.mean(rs):.2f} +- {np.std(rs):.2f}")
    print("  (reglage fin par retropropagation : aucun gain net observe dans le protocole teste, voir ropn_auto_finetune.py.)")

    print("\n=== R4 : Jacobien global de T pas partages (kappa=200, n/cl=100, d=20) ===")
    X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=100, kappa=200.0, sep=0.8, seed=0); Xtr, Xte, ytr, yte = split(X, y, 0)
    U, W, r, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); P = U @ W.T; s, b = make_closed_form_gate(U, W, Xtr)
    from ropn_auto_nobp import sigmoid
    def rec_norm(h, T):                                    # porte normalisee : h <- h + (a/L)(P - I) h avec L = T
        for _ in range(T):
            a = sigmoid(h @ s + b)[:, None]; h = h + (a / T) * (h @ P.T - h)
        return h
    def worst_jac(f):                                      # Jacobien numerique de h0 -> h_T en 5 points test, pire conditionnement
        worst = (0, np.inf, 1, 0)
        for h0 in Xte[:5]:
            Jg = np.zeros((d, d)); eps = 1e-6
            for j in range(d):
                e = np.zeros(d); e[j] = eps; Jg[:, j] = (f((h0 + e)[None])[0] - f((h0 - e)[None])[0]) / (2 * eps)
            sg = svd(Jg, compute_uv=False)
            if sg.max() / sg.min() > worst[2]: worst = (sg.max(), sg.min(), sg.max() / sg.min(), np.linalg.det(Jg))
        return worst
    nP = norm(P, 2)
    print(f"  ||P||_2 = {nP:.3f}")
    print(f"  {'T':>3s} | {'lineaire (I+P/T)^T : cond':>26s} {'borne':>7s} | {'porte non normalisee : cond':>28s} {'det':>10s} | {'porte normalisee (L=T) : cond':>30s} {'det':>10s}")
    for T in (1, 2, 4, 8, 32):
        J = np.linalg.matrix_power(np.eye(d) + P / T, T); sv = svd(J, compute_uv=False); cL = (1 + 1 / T) ** T - 1
        bound = (1 + cL * nP) * (1 + cL * nP / (1 + cL))  # cond(I + c_L P) <= (1 + c_L ||P||)(1 + c_L ||P|| / (1 + c_L))
        w1 = worst_jac(lambda h: recurse_gated(U, W, s, b, h, T)); w2 = worst_jac(lambda h: rec_norm(h, T))
        print(f"  {T:3d} | {sv.max()/sv.min():26.2f} {bound:7.2f} | {w1[2]:28.2e} {w1[3]:10.2e} | {w2[2]:30.2f} {w2[3]:10.2e}")
    print("  (configuration recommandee : T=1, porte analytique, lecture sur le code ; la prediction ne depend alors pas de T.)")

    print("\n=== R5 : stabilite de la selection (rang, configuration du treillis) entre graines ===")
    for kappa in (10.0, 200.0):
        for n_per in (5, 15, 50):
            rs = []
            for sd in range(seeds):
                X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=kappa, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
                rs.append(fit_block_fast(Xtr, ytr, shrink=True, rank="perm", seed=sd)[3]["r_tested"])
            print(f"  rang (test S_T) kappa={kappa:5.0f} n/cl={n_per:3d} : moyenne {np.mean(rs):.2f}, ecart-type {np.std(rs):.2f}, histogramme {np.bincount(rs, minlength=4).tolist()} (rang vrai 3)")
    cfgs = []
    for sd in range(8):
        X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=30, kappa=200.0, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
        _, cfg = fit_lattice_auto(Xtr, ytr, seed=sd); cfgs.append((cfg["use_codes_sq"], 0 if not cfg["use_resid_quad"] else cfg["k_quad"]))
    print(f"  treillis AUTO (codes^2, k_quad) sur 8 graines, donnees gaussiennes : {cfgs}")

    print("\n=== R7 : idempotence et effet du retrecissement sur l'operateur (kappa=200, 10 graines) ===")
    for n_per in (15, 100, 1000):
        idem, dP, alphas = [], [], []
        for sd in range(10):
            X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=200.0, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
            U, W, r, info = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); P = U @ W.T; idem.append(norm(P @ P - P) / norm(P))
            U0, W0, _, _ = fit_block_fast(Xtr, ytr, shrink=False, rank="C1"); P0 = U0 @ W0.T; dP.append(norm(P - P0) / norm(P0))
            from ropn_auto_fast import ledoit_wolf_coef
            yi = np.unique(ytr, return_inverse=True)[1]; mus = np.stack([Xtr[yi == k].mean(0) for k in range(C)]); alphas.append(ledoit_wolf_coef(Xtr - mus[yi])[0])
        print(f"  n/cl={n_per:4d} : ||P^2-P||/||P|| = {np.max(idem):.1e} | ||P_LW - P_brut||/||P_brut|| = {np.mean(dP):.3f} | coefficient LW moyen {np.mean(alphas):.3f}")
    print("  (operateur exact : P = U W^T, W = S^{-1} D Q Lambda^{-1/2}, U = D Q Lambda^{-1/2}, S = (1-a) S_W + a mu I (Ledoit-Wolf),")
    print("   D Q Lambda Q^T = decomposition de D^T S^{-1} D ; W^T U = I_r par construction.)")
