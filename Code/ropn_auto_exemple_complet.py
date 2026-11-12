"""
Exemple complet du fonctionnement d'un bloc analytique avec selection du rang (d = 3, C = 3), avec toutes les matrices
affichees, et generation des figures pedagogiques du rapport (fichiers PDF, matplotlib uniquement).

  Exemple : donnees gaussiennes a covariance commune anisotrope, trois classes dont les moyennes vivent dans un
  plan ; le bloc ferme calcule S_W (retreci), D, Z = S^{-1} D, M = D^T Z (3 x 3), ses valeurs propres, le test
  de permutation (rang 2 attendu), U et W, verifie W^T U = I, projette un point test, lit le code et predit.
  Figures :
    fig_oblique_geometry.pdf   geometrie d'un projecteur oblique en 2D (image, noyau, angle, norme)
    fig_worked_example.pdf     l'exemple complet : nuage 3D projete, directions u_i / w_i, codes, test de rang
    fig_recursion.pdf          recursion gatee : le code est invariant, le residu se contracte
    fig_norm_vs_n.pdf          obliquite admissible ||P||_2 contre n et borne metrique (sqrt(kappa(S)) + 1/sqrt(kappa(S)))/2
    fig_triceptron.pdf         XOR avec un triceptron + saut et cercle avec deux triceptrons (Thm 4)
    fig_double_descent.pdf     norme du projecteur brut (pseudo-inverse de S_W) autour du seuil n_train - C ~ d, contre retreci
"""
import numpy as np
from numpy.linalg import norm, eigh, inv, qr, svd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ropn_auto_fast import fit_block_fast, ledoit_wolf_coef
from ropn_auto_nobp_diag import readout_fit_predict, readout_centroid_fit_predict, centroid_readout_weights
from ropn_auto_nobp import make_multiclass, make_closed_form_gate, recurse_gated

np.set_printoptions(precision=3, suppress=True, linewidth=120)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8, "figure.dpi": 110})
C_OBL, C_ORT, C_KER, C_IMG = "#b2182b", "#2166ac", "#4d4d4d", "#1b7837"


def mat(name, A):
    """Affiche une matrice avec son nom (3 decimales)."""
    A = np.atleast_2d(A); print(f"  {name} =")
    for row in A: print("     [" + " ".join(f"{v:8.3f}" for v in row) + "]")


# ====================================================================== 1. l'exemple complet
def worked_example(seed=0):
    rng = np.random.default_rng(seed); d, C, n_per = 3, 3, 30
    # covariance commune anisotrope : x1, x2 correles (0.7), x3 correle a x1 (0.5) mais sans difference de moyenne
    Sigma = np.array([[1.0, 0.7, 0.5], [0.7, 1.0, 0.2], [0.5, 0.2, 0.6]])
    R = np.linalg.cholesky(Sigma)
    mus = np.array([[0.0, 0.0, 0.0], [3.5, 0.5, 0.0], [0.5, 3.5, 0.0]])
    X = np.concatenate([mus[c] + (R @ rng.standard_normal((d, n_per))).T for c in range(C)]); y = np.repeat(np.arange(C), n_per)
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    Xtr, ytr, Xte, yte = X[:60], y[:60], X[60:], y[60:]
    print("=== Exemple complet : d = 3, C = 3, 30 exemples par classe, decoupage aleatoire non stratifie en 60 (apprentissage) et 30 (test) ===")
    classes, yi = np.unique(ytr, return_inverse=True); ns = np.bincount(yi).astype(float); mu = Xtr.mean(0)
    mus_hat = np.stack([Xtr[yi == k].mean(0) for k in range(C)])
    print("  tailles de classes :", ns.astype(int)); mat("moyennes de classe (lignes)", mus_hat); mat("moyenne globale", mu[None])
    Xc = Xtr - mus_hat[yi]; S_W = Xc.T @ Xc / len(Xtr); a, muLW = ledoit_wolf_coef(Xc); S = (1 - a) * S_W + a * muLW * np.eye(d)
    mat("S_W (intra-classe, denominateur n)", S_W); print(f"  Ledoit-Wolf : a = {a:.3f}, mu = {muLW:.3f}"); mat("S = (1-a) S_W + a mu I", S)
    D = ((mus_hat - mu).T * np.sqrt(ns)) / np.sqrt(C - 1); mat("D = [sqrt(n_c)(mu_c - mu)]/sqrt(C-1)  (d x C)", D)
    Z = np.linalg.solve(S, D); mat("Z = S^{-1} D", Z)
    M = D.T @ Z; M = 0.5 * (M + M.T); mat("M = D^T S^{-1} D  (C x C, rang <= C-1)", M)
    lam, Q = eigh(M); o = np.argsort(lam)[::-1]; lam, Q = lam[o], Q[:, o]
    print(f"  valeurs propres de M : {np.round(lam, 4)}   (la derniere est nulle : rang structurel C-1 = 2)")
    U_, W_, r, info = fit_block_fast(Xtr, ytr, shrink=True, rank="perm", n_perm=500, seed=seed)
    print(f"  test de permutation (metrique S_T, 500 permutations) : valeurs propres dans la metrique S_T = {np.round(info['eigenvalues_test'][:C-1], 2)}, seuil 95% = {info['null_q95']:.4f}, rang teste = {info['r_tested']}, rang retenu r = {r}")
    W = Z @ Q[:, :r] / np.sqrt(lam[:r]); U = D @ Q[:, :r] / np.sqrt(lam[:r])
    mat("W = Z Q_r Lambda_r^{-1/2}  (mesure, d x r)", W); mat("U = D Q_r Lambda_r^{-1/2}  (representation, d x r)", U)
    mat("W^T U", W.T @ U); P = U @ W.T; mat("P = U W^T", P); print(f"  ||P^2 - P|| = {norm(P @ P - P):.1e}, rang(P) = {np.linalg.matrix_rank(P)}")
    Qu, Qw = qr(U)[0], qr(W)[0]; smin = svd(Qw.T @ Qu, compute_uv=False).min()
    print(f"  ||P||_2 = {norm(P, 2):.3f} = 1/sigma_min(Q_W^T Q_U) = {1/smin:.3f} ; angle minimal image/noyau = {np.degrees(np.arcsin(smin)):.1f} deg ; borne metrique (sqrt(k)+1/sqrt(k))/2 = {0.5*(np.sqrt(np.linalg.cond(S)) + 1/np.sqrt(np.linalg.cond(S))):.2f} <= sqrt(k) = {np.sqrt(np.linalg.cond(S)):.2f}, k = kappa(S)")
    print(f"  ecart entre le bloc 'a la main' et fit_block_fast : ||U - U_fast|| = {norm(np.abs(U) - np.abs(U_)):.1e} (au signe des colonnes pres)")
    # lecture
    # lecture de reference : centroides de classe dans l'espace du code (regle LDA, aucun ajustement)
    Ctr, Cte = Xtr @ W, Xte @ W; _, Wro = centroid_readout_weights(Ctr, ytr)
    mat("centroides de classe dans l'espace du code (C x r)", Wro[:-1].T)
    print(f"  termes constants -||m_k||^2/2 + log pi_k : {np.round(Wro[-1], 3)}")
    pred = classes[(np.c_[Cte, np.ones(len(Cte))] @ Wro).argmax(1)]
    pred_ls = readout_fit_predict(Ctr, ytr, Cte)
    print(f"  precision test : {100*np.mean(pred == yte):.1f}%  (parametres : d r + r C + C = {d*r + r*C + C}) ; "
          f"lecture par moindres carres sur le meme code : {100*np.mean(pred_ls == yte):.1f}%, predictions identiques sur {100*np.mean(pred_ls == pred):.0f}% des points")
    Sinv_mu = np.linalg.solve(S, mus_hat.T)                          # LDA de covariance S, calculee directement
    sc_lda = Xte @ Sinv_mu - 0.5 * np.sum(mus_hat.T * Sinv_mu, 0) + np.log(np.bincount(ytr) / len(ytr))
    print(f"  LDA calculee directement avec la meme covariance S : predictions identiques sur {100*np.mean(classes[sc_lda.argmax(1)] == pred):.0f}% des points de test")
    x = Xte[0]; c = W.T @ x; print(f"  point test x = {np.round(x, 3)} (classe {yte[0]}) : code c = W^T x = {np.round(c, 3)}, P x = {np.round(P @ x, 3)}, residu (I-P) x = {np.round(x - P @ x, 3)}")
    print(f"     distances au carre aux centroides ||c - m_k||^2 = {np.round(np.sum((c[None] - Wro[:-1].T) ** 2, 1), 3)}")
    print(f"     scores = {np.round(np.r_[c, 1] @ Wro, 3)} -> classe predite {pred[0]}")
    # recursion gatee : code invariant
    s, b = make_closed_form_gate(U, W, Xtr)
    print("  recursion gatee partagee (invariance du code) sur ce point :")
    for T in (0, 1, 2, 4):
        hT = recurse_gated(U, W, s, b, x[None], T)[0] if T > 0 else x
        print(f"     T={T} : h_T = {np.round(hT, 3)}, code W^T h_T = {np.round(W.T @ hT, 3)}, ||(I-P) h_T|| = {norm(hT - P @ hT):.3f}")
    # test de rang : distribution nulle pour la figure
    null = []
    rng2 = np.random.default_rng(1); Xbar = Xtr - mu
    aT, mT = ledoit_wolf_coef(Xbar); ST = (1 - aT) * Xbar.T @ Xbar / len(Xtr) + aT * mT * np.eye(d)
    MT = D.T @ np.linalg.solve(ST, D); lamT = np.sort(eigh(0.5 * (MT + MT.T))[0])[::-1]
    for _ in range(500):
        yp = rng2.permutation(yi); musp = np.stack([Xtr[yp == k].mean(0) for k in range(C)]); Dp = ((musp - mu).T * np.sqrt(ns)) / np.sqrt(C - 1)
        Mp = Dp.T @ np.linalg.solve(ST, Dp); null.append(eigh(0.5 * (Mp + Mp.T))[0].max())
    return dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, U=U, W=W, P=P, mus=mus_hat, mu=mu, lamT=lamT, null=np.array(null), s=s, b=b, Wro=Wro, S=S)


# ====================================================================== 2. figures
def fig_geometry():
    """Projecteur oblique en 2D : image = span(u), noyau = span(w)^perp, point x, Px, Q_S x."""
    u = np.array([1.0, 0.35]); u /= norm(u); w = np.array([1.0, -1.2]); w /= norm(w); w /= (w @ u)
    P = np.outer(u, w); Q = np.outer(u, u); x = np.array([0.6, 1.35])
    kdir = np.array([-w[1], w[0]]); kdir /= norm(kdir)                     # direction du noyau (w^perp)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    t = np.linspace(-2.4, 2.4, 2)
    ax.plot(t * u[0], t * u[1], color=C_IMG, lw=1.6, label=r"image $\mathrm{im}\,P=\mathrm{span}(u)$")
    ax.plot(t * kdir[0], t * kdir[1], color=C_KER, lw=1.6, ls="--", label=r"kernel $\ker P=\mathrm{span}(w)^\perp$")
    wn = w / norm(w)
    ax.annotate("", xy=wn * 1.0, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C_OBL, lw=1.4)); ax.text(wn[0] * 1.05, wn[1] * 1.05 - 0.1, r"$w$ (measurement direction, $\perp$ kernel)", color=C_OBL, fontsize=8)
    ax.annotate("", xy=u * 1.0, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C_IMG, lw=1.4)); ax.text(u[0] * 0.55, u[1] * 0.55 - 0.3, r"$u$ (representation)", color=C_IMG, fontsize=8)
    Px, Qx = P @ x, Q @ x; xw = wn * (wn @ x)                                # projection orthogonale de x sur span(w) : etape 1
    ax.plot([x[0], xw[0]], [x[1], xw[1]], color=C_OBL, lw=0.9, ls="-."); ax.plot(*xw, "D", color=C_OBL, ms=4)
    ax.text(xw[0] + 0.1, xw[1] - 0.3, r"$\hat x_W$ (step 1: orthogonal onto span$(w)$)", color=C_OBL, fontsize=7)
    ax.annotate("", xy=Px, xytext=xw, arrowprops=dict(arrowstyle="->", color=C_OBL, lw=0.9, ls="-."))
    ax.plot(*x, "ko"); ax.text(x[0] + 0.06, x[1] + 0.04, r"$x$")
    ax.plot([x[0], Px[0]], [x[1], Px[1]], color=C_OBL, lw=1.2); ax.plot(*Px, "o", color=C_OBL); ax.text(Px[0] - 0.9, Px[1] - 0.42, r"$Px$ (oblique, along the kernel)", color=C_OBL, fontsize=8)
    ax.plot([x[0], Qx[0]], [x[1], Qx[1]], color=C_ORT, lw=1.2, ls=":"); ax.plot(*Qx, "s", color=C_ORT, ms=5); ax.text(Qx[0] + 0.12, Qx[1] + 0.22, r"$Q_Sx$ (orthogonal)", color=C_ORT, fontsize=8)
    th = np.degrees(np.arccos(abs(u @ kdir)))
    ax.text(-2.3, -1.9, rf"$\theta_{{\min}}={th:.0f}^\circ,\ \|P\|_2=1/\sin\theta_{{\min}}=1/\cos\angle(u,w)={1/np.sin(np.radians(th)):.2f}$", fontsize=8)
    ax.text(-2.3, -2.15, "two steps: project orthogonally onto span$(w)$, then stretch by $1/\\cos\\angle(u,w)$ along $u$ (step 2)", fontsize=7)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); ax.legend(loc="upper left", frameon=False)
    ax.set_title("Oblique projection: onto the image, parallel to the kernel", fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig("fig_oblique_geometry.pdf"); plt.close(fig)


def fig_worked(ex):
    Xtr, ytr, Xte, yte, U, W, mus, mu = ex["Xtr"], ex["ytr"], ex["Xte"], ex["yte"], ex["U"], ex["W"], ex["mus"], ex["mu"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6)); cols = ["#1b9e77", "#d95f02", "#7570b3"]
    ax = axs[0]
    for k in range(3):
        ax.scatter(Xtr[ytr == k, 0], Xtr[ytr == k, 1], s=14, color=cols[k], alpha=0.7, label=f"class {k}")
    for i in range(U.shape[1]):                                   # directions affichees a longueur 2.2 (echelle du nuage)
        ud = U[:2, i] / norm(U[:2, i]) * 2.2; wd = W[:2, i] / norm(W[:2, i]) * 2.2
        ax.annotate("", xy=mu[:2] + ud, xytext=mu[:2], arrowprops=dict(arrowstyle="->", color=C_IMG, lw=1.8))
        ax.annotate("", xy=mu[:2] + wd, xytext=mu[:2], arrowprops=dict(arrowstyle="->", color=C_OBL, lw=1.8, ls="--"))
        ax.text(*(mu[:2] + ud * 1.12), rf"$u_{i+1}$", color=C_IMG, fontsize=10); ax.text(*(mu[:2] + wd * 1.12), rf"$w_{i+1}$", color=C_OBL, fontsize=10)
    ax.set_aspect("equal"); ax.legend(frameon=False, loc="lower right"); ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title("Data (first two coordinates), $u_i$ in the span of the means,\n$w_i=S^{-1}u_i$ tilted by the within-class covariance")
    ax = axs[1]; Cte = Xte @ W; Ctr = Xtr @ W
    for k in range(3):
        ax.scatter(Ctr[ytr == k, 0], Ctr[ytr == k, 1], s=14, color=cols[k], alpha=0.5); ax.scatter(Cte[yte == k, 0], Cte[yte == k, 1], s=30, color=cols[k], marker="^", edgecolor="k", lw=0.5)
    ax.set_xlabel(r"$c_1=w_1^\top x$"); ax.set_ylabel(r"$c_2=w_2^\top x$"); ax.set_title("Oblique code $c=W^\\top x$ (train: dots, test: triangles)\nthe read-out is a linear map of this $r=2$ vector")
    ax = axs[2]; null, lamT = ex["null"], ex["lamT"]
    ax.hist(null, bins=30, color="#bbbbbb", label="null (500 permutations)"); q = np.quantile(null, 0.95); ax.axvline(q, color="k", ls="--", label="95th percentile")
    for i, l in enumerate(lamT[:2]): ax.axvline(l, color=C_OBL, lw=2); ax.text(l, ax.get_ylim()[1] * (0.85 - 0.12 * i), rf"$\lambda_{i+1}$", color=C_OBL, ha="right")
    ax.set_xlabel(r"largest eigenvalue of $D_p^\top S_T^{-1}D_p$"); ax.set_title("Rank test: two eigenvalues above the null,\n$r=2=C-1$ retained"); ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig("fig_worked_example.pdf"); plt.close(fig)


def fig_recursion(ex):
    """Invariance du code en 2D : trajectoires h_t de la recursion gatee, code invariant, residu contracte."""
    U, W, P, s, b = ex["U"], ex["W"], ex["P"], ex["s"], ex["b"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axs[0]; Xte, yte = ex["Xte"], ex["yte"]; cols = ["#1b9e77", "#d95f02", "#7570b3"]
    for x, k in zip(Xte[:9], yte[:9]):
        traj = [x] + [recurse_gated(U, W, s, b, x[None], T)[0] for T in range(1, 7)]; traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 2], "-", color=cols[k], lw=1, alpha=0.8); ax.plot(traj[0, 0], traj[0, 2], "o", color=cols[k], ms=4); ax.plot(traj[-1, 0], traj[-1, 2], "s", color=cols[k], ms=4)
    ax.set_xlabel(r"$h_1$"); ax.set_ylabel(r"$h_3$ (kernel-dominated coordinate)"); ax.set_title("Gated recursion $h_{t+1}=(1-a)h_t+aPh_t$, $T=0\\to6$\n(dots: $h_0$, squares: $h_6$): motion along the kernel only")
    ax = axs[1]; Ts = np.arange(0, 13)
    for x, k in zip(Xte[:6], yte[:6]):
        codes = [norm(W.T @ (recurse_gated(U, W, s, b, x[None], T)[0] if T > 0 else x) - W.T @ x) for T in Ts]
        res = [norm((np.eye(3) - P) @ (recurse_gated(U, W, s, b, x[None], T)[0] if T > 0 else x)) for T in Ts]
        ax.semilogy(Ts, np.maximum(res, 1e-17), "-", color=cols[k], lw=1); ax.semilogy(Ts, np.maximum(codes, 1e-17), ":", color=cols[k], lw=1)
    ax.set_xlabel(r"$T$"); ax.set_ylabel("norm"); ax.set_title("Residual $\\|(I-P)h_T\\|$ decays as $(1-a)^T$ (solid);\ncode change $\\|W^\\top h_T-W^\\top h_0\\|$ stays at $10^{-16}$ (dotted)")
    fig.tight_layout(); fig.savefig("fig_recursion.pdf"); plt.close(fig)


def fig_norm_vs_n():
    from sklearn.model_selection import train_test_split
    ns, raw, lw, bound = [5, 10, 15, 30, 100, 300, 1000], [], [], []
    for n_per in ns:
        r_, l_, b_ = [], [], []
        for sd in range(10):
            X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=n_per, kappa=200.0, sep=0.8, seed=sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
            U1, W1, _, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1"); l_.append(norm(U1 @ W1.T, 2))
            yi = np.unique(ytr, return_inverse=True)[1]; mus = np.stack([Xtr[yi == k].mean(0) for k in range(4)]); Xc = Xtr - mus[yi]
            a, mu = ledoit_wolf_coef(Xc); S = (1 - a) * Xc.T @ Xc / len(Xtr) + a * mu * np.eye(20); k_ = np.sqrt(np.linalg.cond(S)); b_.append(0.5 * (k_ + 1 / k_))
            if len(Xtr) > 24:
                U0, W0, _, _ = fit_block_fast(Xtr, ytr, shrink=False, rank="C1"); r_.append(norm(U0 @ W0.T, 2))
        raw.append(np.mean(r_) if r_ else np.nan); lw.append(np.mean(l_)); bound.append(np.mean(b_))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.semilogx(ns, lw, "o-", color=C_OBL, label=r"$\|P\|_2$, shrunk metric (Ledoit--Wolf)"); ax.semilogx(ns, raw, "s--", color=C_KER, label=r"$\|P\|_2$, raw $S_W$")
    ax.semilogx(ns, bound, "^:", color=C_ORT, label=r"metric bound $\frac{1}{2}(\sqrt{\kappa(S)}+1/\sqrt{\kappa(S)})$")
    ax.set_xlabel("training samples per class"); ax.set_ylabel("spectral norm"); ax.set_title("Admissible obliquity grows with $n$ as the shrinkage vanishes\n($d=20$, $C=4$, $\\kappa(S_W)=200$, 10 seeds)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig("fig_norm_vs_n.pdf"); plt.close(fig)


def fig_triceptron():
    sig = lambda z: 1 / (1 + np.exp(-z)); g = np.linspace(-2, 2, 201); X1, X2 = np.meshgrid(g, g)
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.6))
    f_xor = sig(0.05 * X1) * X2 - 0.5 * X2
    cs = axs[0].contourf(X1, X2, f_xor / 0.0125, levels=21, cmap="RdBu_r"); axs[0].contour(X1, X2, f_xor, levels=[0], colors="k")
    axs[0].set_title("XOR with ONE triceptron $+$ skip:\n$f=\\sigma(0.05x_1)x_2-x_2/2\\approx 0.0125\\,x_1x_2$"); fig.colorbar(cs, ax=axs[0], shrink=0.85)
    f_c = sig(0.05 * X1) * X1 + sig(0.05 * X2) * X2 - 0.5 * (X1 + X2)
    cs = axs[1].contourf(X1, X2, f_c / 0.0125, levels=21, cmap="viridis"); axs[1].contour(X1, X2, f_c, levels=[0.0125], colors="w")
    th = np.linspace(0, 2 * np.pi, 200); axs[1].plot(np.cos(th), np.sin(th), "r--", lw=1, label="unit circle")
    axs[1].set_title("Unit circle with TWO triceptrons $+$ skip:\n$f\\approx0.0125(x_1^2+x_2^2)$, level set (white) vs circle (red)"); axs[1].legend(frameon=False, loc="lower left"); fig.colorbar(cs, ax=axs[1], shrink=0.85)
    for ax in axs: ax.set_aspect("equal"); ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    fig.tight_layout(); fig.savefig("fig_triceptron.pdf"); plt.close(fig)


def fig_double_descent():
    """Norme du projecteur de Fisher brut (pseudo-inverse de S_W) et retreci autour du seuil d'interpolation."""
    from numpy.linalg import pinv
    from sklearn.model_selection import train_test_split
    def block_metric(Xtr_, ytr_, Smat):
        cl, yi_ = np.unique(ytr_, return_inverse=True); Cn = len(cl); ns_ = np.bincount(yi_).astype(float); mu_ = Xtr_.mean(0)
        mus_ = np.stack([Xtr_[yi_ == k].mean(0) for k in range(Cn)]); D_ = ((mus_ - mu_).T * np.sqrt(ns_)) / np.sqrt(Cn - 1)
        Z_ = pinv(Smat) @ D_; M_ = D_.T @ Z_; M_ = 0.5 * (M_ + M_.T); lam_, Q_ = eigh(M_); o_ = np.argsort(lam_)[::-1]; lam_, Q_ = lam_[o_], Q_[:, o_]
        lr_ = np.clip(lam_[:Cn - 1], 1e-12, None); return D_ @ Q_[:, :Cn - 1] / np.sqrt(lr_), Z_ @ Q_[:, :Cn - 1] / np.sqrt(lr_)
    d, C = 20, 4; ratios, nr, nl, ar, al = [], [], [], [], []
    for n_per in list(range(3, 21)) + [25, 30, 40, 50]:
        rows = []
        for sd in range(20):
            X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=200.0, sep=0.8, seed=sd)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
            yi = np.unique(ytr, return_inverse=True)[1]; mus = np.stack([Xtr[yi == k].mean(0) for k in range(C)]); Xc = Xtr - mus[yi]
            S_W = Xc.T @ Xc / len(Xtr); a, mu = ledoit_wolf_coef(Xc); S = (1 - a) * S_W + a * mu * np.eye(d)
            Ur, Wr = block_metric(Xtr, ytr, S_W); Ul, Wl = block_metric(Xtr, ytr, S)
            rows.append(((len(Xtr) - C) / d, norm(Ur @ Wr.T, 2), norm(Ul @ Wl.T, 2),
                         (readout_centroid_fit_predict(Xtr @ Wr, ytr, Xte @ Wr) == yte).mean(), (readout_centroid_fit_predict(Xtr @ Wl, ytr, Xte @ Wl) == yte).mean()))
        m = np.mean(rows, 0); ratios.append(m[0]); nr.append(m[1]); nl.append(m[2]); ar.append(m[3]); al.append(m[4])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(ratios, nr, "s-", color=C_KER, label=r"raw Fisher projector ($S_W^{+}$)"); axs[0].plot(ratios, nl, "o-", color=C_OBL, label="shrunk metric (Ledoit--Wolf)")
    axs[0].axvline(1.0, color="k", ls=":", lw=1); axs[0].set_xlabel(r"$(n_{\mathrm{train}}-C)/d$"); axs[0].set_ylabel(r"$\|P\|_2$"); axs[0].set_xlim(0, 7.2)
    axs[0].set_title("Norm of the block across the interpolation threshold\n(dotted: $S_W$ becomes full rank)"); axs[0].legend(frameon=False)
    axs[1].plot(ratios, 100 * np.array(ar), "s-", color=C_KER, label="raw"); axs[1].plot(ratios, 100 * np.array(al), "o-", color=C_OBL, label="shrunk")
    axs[1].axvline(1.0, color="k", ls=":", lw=1); axs[1].set_xlabel(r"$(n_{\mathrm{train}}-C)/d$"); axs[1].set_ylabel("test accuracy (%)"); axs[1].set_xlim(0, 7.2)
    axs[1].set_title("Test accuracy of the code read-out\n($d=20$, $C=4$, $\\kappa(S_W)=200$, 20 seeds)"); axs[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig("fig_double_descent.pdf"); plt.close(fig)
    print("  seuil d'interpolation : (n_train-C)/d, ||P|| brut, ||P|| LW, acc brut, acc LW")
    for r_, a_, b_, c_, d_ in zip(ratios, nr, nl, ar, al): print(f"     {r_:5.2f}  {a_:6.2f}  {b_:5.2f}  {100*c_:5.1f}  {100*d_:5.1f}")


if __name__ == "__main__":
    ex = worked_example()
    fig_geometry(); fig_worked(ex); fig_recursion(ex); fig_norm_vs_n(); fig_triceptron(); fig_double_descent()
    print("\n  figures ecrites : fig_oblique_geometry.pdf, fig_worked_example.pdf, fig_recursion.pdf, fig_norm_vs_n.pdf, fig_triceptron.pdf, fig_double_descent.pdf")
