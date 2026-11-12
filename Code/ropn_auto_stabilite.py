"""
Limites de stabilite des projecteurs obliques et leur controle dans le bloc ferme : verifications numeriques.

  S1  norme et angle : ||P||_2 = 1/sin(theta_min) = 1/sigma_min(Q_W^T Q_U) ; ||I - P||_2 = ||P||_2 ;
      la borne ||P(x+dx) - Px|| <= ||P||_2 ||dx|| est atteinte dans la direction singuliere dominante ;
      exemple 2x2 P_a = [[1, a], [0, 0]] : valeurs propres {0, 1}, ||P_a||_2 = sqrt(1 + a^2).
  S2  le conditionnement du cross-Gram ne suffit pas : W^T U = I_r (kappa = 1) alors que ||P||_2 est arbitraire.
  S3  borne metrique : pour un projecteur M-orthogonal, ||P||_2 <= (sqrt(kappa) + 1/sqrt(kappa))/2 <= sqrt(kappa),
      kappa = kappa(M) (inegalite de Wielandt ; la premiere borne est atteinte au rang un pour u = v_min/sqrt(l_min) + v_max/sqrt(l_max)) ; avec M = S^{-1},
      S = (1-a) S_W + a mu I (Ledoit-Wolf), ||P||_2 <= sqrt(kappa(S)) <= sqrt(1 + (1-a) lambda_max(S_W) / (a mu)) :
      le retrecissement BORNE l'obliquite admissible ; mesure de ||P||_2 brut / retreci en fonction de n.
  S4  meilleure approximation : pas en norme euclidienne (||x - Px|| <= ||P||_2 ||x - Q_S x||, borne atteinte),
      mais oui dans la metrique de Fisher ||.||_M (P est M-orthogonal).
  S5  perte d'information : une perturbation dans ker(W^T) ne change ni le code ni la prediction ;
      la forme residuelle I + P/L n'est pas idempotente mais inversible, d'inverse I - P/(L+1).
  S6  regulariser le cross-Gram, (W^T U + lambda I)^{-1}, detruit l'idempotence ; regulariser la METRIQUE
      (retrecissement de S_W) la conserve exactement.
  S7  profondeur : ||T||_2 <= prod ||P_l||_2 (borne lache) ; inversibilite, stabilite et conditionnement sont
      trois proprietes distinctes (det != 0 avec sigma_min -> 0).
  S8  spectre singulier : les r valeurs singulieres non nulles de P sont 1/cos(theta_i), theta_i = angles
      principaux entre span(U) et span(W) (lecture en deux etapes : projection orthogonale sur span(W), puis
      re-expression dans span(U) avec un facteur 1/cos(theta_i) par direction) ; le code ne depend de x que par
      sa projection orthogonale sur span(W).
  S9  seuil d'interpolation : ||P||_2 du projecteur de Fisher BRUT (pseudo-inverse de S_W) presente un pic
      quand n_train - C ~ d (S_W presque singuliere) ; le retrecissement supprime le pic.
numpy / scipy / sklearn seulement.
"""
import numpy as np
from numpy.linalg import norm, qr, svd, eigvalsh, inv
from sklearn.model_selection import train_test_split
from ropn_auto_nobp import make_multiclass, make_closed_form_gate, recurse_gated
from ropn_auto_fast import fit_block_fast, ledoit_wolf_coef
from ropn_auto_nobp_diag import readout_centroid_fit_predict   # lecture de reference : centroides sur le code (regle LDA)


def split(X, y, sd): return train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)


def block(X, y, shrink=True):
    U, W, r, info = fit_block_fast(X, y, shrink=shrink, rank="C1"); return U, W, U @ W.T


def shrunk_metric(X, y):
    """S = (1-a) S_W + a mu I (convention sklearn, denominateur n), identique au solveur du bloc fast."""
    classes, yi = np.unique(y, return_inverse=True); mus = np.stack([X[yi == k].mean(0) for k in range(len(classes))])
    Xc = X - mus[yi]; a, mu = ledoit_wolf_coef(Xc); S_W = Xc.T @ Xc / len(X)
    return (1 - a) * S_W + a * mu * np.eye(X.shape[1]), a, mu, S_W


if __name__ == "__main__":
    rng = np.random.default_rng(0); C, d = 4, 20
    print("=== S1. Norme, angle, borne d'amplification atteinte, exemple 2x2 ===")
    for a in (0.0, 1.0, 10.0):
        Pa = np.array([[1.0, a], [0.0, 0.0]]); print(f"  P_a, a={a:4.1f} : P^2 = P ? {np.allclose(Pa @ Pa, Pa)} | valeurs propres {np.round(np.linalg.eigvals(Pa), 6)} | ||P_a||_2 = {norm(Pa, 2):.4f} = sqrt(1+a^2) = {np.sqrt(1 + a * a):.4f}"
              f" | perturbation (0, eps) -> {np.round(Pa @ np.array([0, 1e-3]), 5)}")
    for kappa in (10.0, 200.0):
        X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=100, kappa=kappa, sep=0.8, seed=0); Xtr, Xte, ytr, yte = split(X, y, 0)
        U, W, P = block(Xtr, ytr); Qu, Qw = qr(U)[0], qr(W)[0]; smin = svd(Qw.T @ Qu, compute_uv=False).min()
        # angle minimal entre im P = span(U) et ker P = span(W)^perp : sin(theta_min) = sigma_min(Q_W^T Q_U)
        Nk = qr(np.eye(d) - Qw @ Qw.T)[0][:, :d - U.shape[1]]           # base de ker P (= (span W)^perp)
        cos_max = svd(Qu.T @ Nk, compute_uv=False).max(); theta_min = np.arccos(min(cos_max, 1.0))
        _, sv, Vt = svd(P); dx = Vt[0]                                   # direction singuliere dominante
        x = Xte[0]; amp = norm(P @ (x + dx) - P @ x) / norm(dx)
        print(f"  bloc ferme kappa={kappa:5.0f} : ||P||_2 = {norm(P, 2):.4f} | 1/sigma_min(Q_W^T Q_U) = {1/smin:.4f} | 1/sin(theta_min) = {1/np.sin(theta_min):.4f} "
              f"| ||I-P||_2 = {norm(np.eye(d) - P, 2):.4f} | amplification mesuree le long de v_1 = {amp:.4f} (borne atteinte)")

    print("\n=== S2. kappa(W^T U) = 1 par construction ne borne pas ||P||_2 (rang 1) ===")
    u = np.array([1.0, 0.0])
    for t in (0.0, 1.0, 5.0, 50.0):
        w = u + t * np.array([0.0, 1.0]); P1 = np.outer(u, w) / (w @ u)   # w^T u = 1 : cross-Gram = [1], kappa = 1
        print(f"  w = u + {t:4.1f} v_perp : W^T U = {w @ u:.1f} (kappa = 1) | ||P||_2 = {norm(P1, 2):.3f} = sqrt(1 + t^2) = {np.sqrt(1 + t * t):.3f} | P^2 = P ? {np.allclose(P1 @ P1, P1)}")

    print("\n=== S3. Borne metrique ||P||_2 <= (sqrt(kappa)+1/sqrt(kappa))/2 <= sqrt(kappa), kappa = kappa(S), et controle de l'obliquite par le retrecissement (kappa=200, 10 graines) ===")
    print(f"  {'n/cl':>5s} {'a (LW)':>7s} {'||P||_2 brut':>12s} {'||P||_2 LW':>10s} {'borne fine (S_LW)':>18s} {'sqrt(kappa(S_LW))':>18s} {'sqrt(1+(1-a)lmax/(a mu))':>24s} {'sqrt(kappa(S_W))':>16s}")
    for n_per in (10, 15, 30, 100, 1000):
        rows = []
        for sd in range(10):
            X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=200.0, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
            S, a, mu, S_W = shrunk_metric(Xtr, ytr); ev = eigvalsh(S); evw = eigvalsh(S_W)
            _, _, P_lw = block(Xtr, ytr, shrink=True); _, _, P_raw = block(Xtr, ytr, shrink=False)
            rows.append((a, norm(P_raw, 2), norm(P_lw, 2), np.sqrt(ev.max() / ev.min()), np.sqrt(1 + (1 - a) * evw.max() / (a * mu)),
                         np.sqrt(evw.max() / max(evw.min(), 1e-300)), norm(inv(S) @ P_lw - P_lw.T @ inv(S)) / norm(inv(S) @ P_lw)))
        m = np.mean(rows, 0); fine = np.mean([0.5 * (r_[3] + 1 / r_[3]) for r_ in rows]); viol = max(r_[2] - 0.5 * (r_[3] + 1 / r_[3]) for r_ in rows)
        print(f"  {n_per:5d} {m[0]:7.3f} {m[1]:12.2f} {m[2]:10.2f} {fine:18.2f} {m[3]:18.2f} {m[4]:24.2f} {m[5]:16.2f}   | ||M P - P^T M||/||M P|| = {m[6]:.1e} | max(||P||_2 - borne fine) = {viol:.2f}")
    print("  (la borne fine est respectee pour chaque graine ; le coefficient a decroit avec n et l'obliquite admissible croit.)")
    # la borne fine est atteinte au rang un : u = e_1 + e_d dans la base propre de M
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(2000):
        dd = int(rng.integers(2, 8)); A_ = rng.standard_normal((dd, dd)); Mm = A_ @ A_.T + 0.05 * np.eye(dd); rr = int(rng.integers(1, dd))
        Uu = rng.standard_normal((dd, rr)); Pp = Uu @ np.linalg.solve(Uu.T @ Mm @ Uu, Uu.T @ Mm); k_ = np.sqrt(np.linalg.cond(Mm))
        worst = max(worst, norm(Pp, 2) / (0.5 * (k_ + 1 / k_)))
    ev_, V_ = np.linalg.eigh(Mm); u_ = V_[:, [0]] / np.sqrt(ev_[0]) + V_[:, [-1]] / np.sqrt(ev_[-1]); P1 = u_ @ (u_.T @ Mm) / (u_.T @ Mm @ u_); k_ = np.sqrt(ev_[-1] / ev_[0])
    print(f"  2000 couples (M, U) aleatoires : max ||P||_2 / borne fine = {worst:.4f} ; rang un, u = v_min/sqrt(l_min) + v_max/sqrt(l_max) : ||P||_2 = {norm(P1, 2):.4f}, borne fine = {0.5 * (k_ + 1 / k_):.4f} (atteinte)")

    print("\n=== S4. Meilleure approximation : euclidienne non, dans la metrique de Fisher oui (kappa=200, n/cl=100) ===")
    X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=100, kappa=200.0, sep=0.8, seed=0); Xtr, Xte, ytr, yte = split(X, y, 0)
    U, W, P = block(Xtr, ytr); S, a, mu, _ = shrunk_metric(Xtr, ytr); M = inv(S); Qu = qr(U)[0]; Qs = Qu @ Qu.T
    Lm = np.linalg.cholesky(M)                                            # ||v||_M = ||L^T v||
    r_eu = [norm(x - P @ x) / norm(x - Qs @ x) for x in Xte]; r_M = [norm(Lm.T @ (x - P @ x)) / norm(Lm.T @ (x - Qs @ x)) for x in Xte]
    print(f"  ||x - Px|| / ||x - Q_S x|| (euclidien) : min {min(r_eu):.3f}, max {max(r_eu):.3f} <= ||P||_2 = {norm(P, 2):.3f} ; ratio > 1 sur {100*np.mean(np.array(r_eu) > 1):.0f}% des points")
    print(f"  meme ratio en norme M = S^-1 (Fisher)  : min {min(r_M):.3f}, max {max(r_M):.3f} (<= 1 partout : P est la meilleure approximation M-orthogonale)")

    print("\n=== S5. Perte d'information et forme residuelle ===")
    s_, b_ = make_closed_form_gate(U, W, Xtr); pred0 = readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W)
    Nk = qr(np.eye(d) - qr(W)[0] @ qr(W)[0].T)[0][:, :d - W.shape[1]]   # base de ker(W^T)
    sig = Xtr.std(0).mean(); dz = (Nk @ rng.standard_normal((Nk.shape[1], len(Xte)))).T; dz *= 3 * sig / norm(dz, axis=1, keepdims=True)
    du = (rng.standard_normal((len(Xte), U.shape[1])) @ U.T); du *= 3 * sig / norm(du, axis=1, keepdims=True)
    pk = readout_centroid_fit_predict(Xtr @ W, ytr, (Xte + dz) @ W); pu = readout_centroid_fit_predict(Xtr @ W, ytr, (Xte + du) @ W)
    print(f"  perturbation de norme 3 sigma dans ker(W^T) : code change de {norm((Xte + dz) @ W - Xte @ W):.1e}, predictions identiques {100*np.mean(pk == pred0):.1f}%")
    print(f"  perturbation de meme norme dans span(U)     : predictions identiques {100*np.mean(pu == pred0):.1f}%  (le noyau est invisible, l'image ne l'est pas)")
    for L in (1, 2, 4):
        T = np.eye(d) + P / L; Tinv = np.eye(d) - P / (L + 1)
        print(f"  I + P/{L} : idempotent ? ||T^2 - T||/||T|| = {norm(T @ T - T)/norm(T):.2f} | inverse I - P/{L+1} : ||T (I - P/(L+1)) - I|| = {norm(T @ Tinv - np.eye(d)):.1e} | det = (1+1/L)^r = {np.linalg.det(T):.4f} vs {(1 + 1/L)**U.shape[1]:.4f}")

    print("\n=== S6. Regulariser le cross-Gram detruit l'idempotence ; regulariser la metrique la conserve ===")
    Uf, Wf, _, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); G = Wf.T @ Uf
    for lam in (0.0, 1e-2, 1e-1, 1.0):
        Pl = Uf @ inv(G + lam * np.eye(G.shape[0])) @ Wf.T
        print(f"  (W^T U + {lam:4.2f} I)^-1 : ||P^2 - P||/||P|| = {norm(Pl @ Pl - Pl)/norm(Pl):.1e}")
    print(f"  metrique retrecie (Ledoit-Wolf, a = {a:.3f}) : ||P^2 - P||/||P|| = {norm(P @ P - P)/norm(P):.1e}, ||W^T U - I|| = {norm(Wf.T @ Uf - np.eye(G.shape[0])):.1e}")

    print("\n=== S7. Profondeur : borne produit et inversibilite != conditionnement ===")
    for L in (2, 4, 8):
        T = np.linalg.matrix_power(np.eye(d) + P / L, L); cL = (1 + 1 / L) ** L - 1
        print(f"  L={L} : ||(I+P/L)^L||_2 = {norm(T, 2):.3f} | borne produit (1 + ||P||_2/L)^L = {(1 + norm(P, 2)/L)**L:.3f} | exact ||I + c_L P||_2 = {norm(np.eye(d) + cL * P, 2):.3f}")
    for T_ in (1, 4, 8):
        h0 = Xte[0]; J = np.zeros((d, d)); eps = 1e-6
        for j in range(d):
            e = np.zeros(d); e[j] = eps
            J[:, j] = (recurse_gated(U, W, s_, b_, (h0 + e)[None], T_)[0] - recurse_gated(U, W, s_, b_, (h0 - e)[None], T_)[0]) / (2 * eps)
        sv = svd(J, compute_uv=False)
        print(f"  porte, T={T_} : det = {np.linalg.det(J):.2e} (non nul) | sigma_min = {sv.min():.3e} | sigma_max = {sv.max():.3f} | conditionnement = {sv.max()/sv.min():.1f}")

    print("\n=== S8. Spectre singulier d'un projecteur oblique et lecture en deux etapes (kappa=200, n/cl=100) ===")
    sv = svd(P, compute_uv=False)[:U.shape[1]]; cosang = svd(qr(W)[0].T @ qr(U)[0], compute_uv=False)
    print(f"  valeurs singulieres non nulles de P : {np.round(sv, 4)}")
    print(f"  1/cos(theta_i), angles principaux span(U)/span(W) = {np.round(np.degrees(np.arccos(cosang)), 2)} deg : {np.round(np.sort(1 / cosang)[::-1], 4)}")
    Qw = qr(W)[0]; x = Xte[0]; print(f"  W^T x - W^T (Q_W Q_W^T x) = {norm(W.T @ x - W.T @ (Qw @ Qw.T @ x)):.1e} : le code ne lit que la projection orthogonale de x sur span(W)")

    print("\n=== S9. Seuil d'interpolation : ||P||_2 brut (pseudo-inverse de S_W) contre retreci, en fonction de (n_train - C)/d (kappa=200, 10 graines) ===")
    from numpy.linalg import pinv, eigh
    def block_metric(Xtr_, ytr_, Smat):
        cl, yi_ = np.unique(ytr_, return_inverse=True); Cn = len(cl); ns_ = np.bincount(yi_).astype(float); mu_ = Xtr_.mean(0)
        mus_ = np.stack([Xtr_[yi_ == k].mean(0) for k in range(Cn)]); D_ = ((mus_ - mu_).T * np.sqrt(ns_)) / np.sqrt(Cn - 1)
        Z_ = pinv(Smat) @ D_; M_ = D_.T @ Z_; M_ = 0.5 * (M_ + M_.T); lam_, Q_ = eigh(M_); o_ = np.argsort(lam_)[::-1]; lam_, Q_ = lam_[o_], Q_[:, o_]
        lr_ = np.clip(lam_[:Cn - 1], 1e-12, None); return D_ @ Q_[:, :Cn - 1] / np.sqrt(lr_), Z_ @ Q_[:, :Cn - 1] / np.sqrt(lr_)
    print(f"  {'n/cl':>5s} {'(n_train-C)/d':>14s} {'||P|| brut':>10s} {'||P|| LW':>9s} {'acc brut':>9s} {'acc LW':>7s}")
    for n_per in (4, 6, 7, 8, 9, 10, 12, 15, 30):
        rows = []
        for sd in range(10):
            X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=200.0, sep=0.8, seed=sd); Xtr, Xte, ytr, yte = split(X, y, sd)
            S_, a_, mu_, S_W_ = shrunk_metric(Xtr, ytr); Ur, Wr = block_metric(Xtr, ytr, S_W_); Ul, Wl = block_metric(Xtr, ytr, S_)
            rows.append((len(Xtr), norm(Ur @ Wr.T, 2), norm(Ul @ Wl.T, 2), (readout_centroid_fit_predict(Xtr @ Wr, ytr, Xte @ Wr) == yte).mean(), (readout_centroid_fit_predict(Xtr @ Wl, ytr, Xte @ Wl) == yte).mean()))
        m = np.mean(rows, 0); print(f"  {n_per:5d} {(m[0] - C) / d:14.2f} {m[1]:10.2f} {m[2]:9.2f} {100 * m[3]:9.1f} {100 * m[4]:7.1f}")
    print("  (le pic de ||P|| brut est a (n_train - C)/d ~ 0.9 ; le retrecissement le supprime et la precision brute ne montre pas de double descente ici.)")
