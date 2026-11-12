"""
Verifications numeriques des resultats sur la recursion a poids partages et sur la capacite du triceptron.

A. Reduction de la profondeur gatee a poids partages :
   bloc h <- h + (1/L) U diag(a(h)) W^T h, a = sigma(S^T h + b), P = U W^T idempotent.
   (i) le residu rho = (I-P) h est invariant ; (ii) le code c = W^T h suit c <- c * (1 + a/L) avec
   a = sigma(A c + beta + b), A = S^T U, beta = S^T rho ; (iii) le code c_T ne depend de l'entree qu'a travers
   (c_0, S^T rho_0) : deux entrees de meme (c_0, S^T rho_0) et residus differents donnent le meme c_T ;
   (iv) si S dans span(W) alors beta = 0 et c_T ne depend pas du residu ; (v) pour r = 1 la
   reparametrisation c_0 -> c_T est strictement croissante sous la condition du corollaire, et un
   contre-exemple montre que la condition est necessaire.
   Variante scalaire : rho_T = Lambda_T rho_0, direction du residu fixe.

B. Capacite quadrique du triceptron :
   (i) swish = triceptron avec s = w ; ReLU = limite uniforme (ecart sup 0.2785 / kappa) ;
   (ii) k unites en regime doux + saut lineaire approchent toute quadrique d'inertie max(p, n) <= k :
        cercle (2 unites, ecart radial kappa^2 R^3 / 24) et XOR (1 unite + saut, exact en signe) ;
   (iv) approximation du disque par un reseau ReLU a m unites : la frontiere a au plus N(m) = 1 + m + C(m,2)
        segments, d'ou une erreur d'encadrement >= R tan^2(pi / 2N) ; comparaison avec l'erreur mesuree.

C. Invariance du code, determinant exact du bloc gate scalaire et determinant r x r du bloc gate par
   coordonnee : comparaison aux differences finies.
"""
import numpy as np
from numpy.linalg import norm, qr
from ropn_auto_fast import fit_block_fast
from ropn_auto_nobp import make_multiclass


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ================================================================== A
def gated_block_steps(h, U, W, S, b, L, T):
    for _ in range(T):
        a = sigmoid(h @ S + b)                      # (n, r) portes par coordonnee de code
        h = h + (1.0 / L) * ((h @ W) * a) @ U.T     # h + U diag(a) W^T h / L
    return h


def verify_A(seed=0):
    rng = np.random.default_rng(seed)
    X, y, *_ = make_multiclass(n_classes=4, d=12, n_per=40, kappa=20.0, sep=1.0, seed=seed)
    U, W, r, _ = fit_block_fast(X, y, shrink=True, rank="C1"); d = X.shape[1]
    P = U @ W.T; Q = np.eye(d) - P
    S = rng.standard_normal((d, r)); b = rng.standard_normal(r); L, T = 4, 6
    h0 = rng.standard_normal((5, d))
    hT = gated_block_steps(h0, U, W, S, b, L, T)
    # (i) residu invariant
    e_rho = norm(hT @ Q.T - h0 @ Q.T) / norm(h0 @ Q.T)
    # (ii) recursion du code en dimension r
    c = h0 @ W; beta = (h0 @ Q.T) @ S; A = S.T @ U
    for _ in range(T):
        a = sigmoid(c @ A.T + beta + b); c = c * (1 + a / L)
    e_code = norm(hT @ W - c) / norm(c)
    # (iii) deux entrees de meme (c_0, S^T rho_0), residus differents -> meme c_T
    #  construire rho' = rho + delta avec delta dans ker(P) et S^T delta = 0
    rho = h0 @ Q.T
    Nk = qr(Q)[0][:, :d - r]                            # base de ker(P) (= im(I-P))
    M = S.T @ Nk                                        # (r, d-r) ; on cherche delta = Nk z, M z = 0
    _, _, Vt = np.linalg.svd(M); z = Vt[-1]; delta = Nk @ z
    assert norm(P @ delta) < 1e-10 and norm(S.T @ delta) < 1e-10
    h0b = h0 + 3.0 * delta[None, :]
    hTb = gated_block_steps(h0b, U, W, S, b, L, T)
    e_iii = norm(hTb @ W - hT @ W) / norm(hT @ W)
    # (iv) S dans span(W) -> c_T independant du residu
    S2 = W @ rng.standard_normal((r, r))
    hT1 = gated_block_steps(h0, U, W, S2, b, L, T)
    hT2 = gated_block_steps(h0 + 5.0 * (rng.standard_normal((5, d)) @ Q.T), U, W, S2, b, L, T)   # residu change
    e_iv = norm(hT1 @ W - hT2 @ W) / norm(hT1 @ W)
    # (v) r = 1 : monotonie de c_0 -> c_T a beta fixe (sous condition d'inversibilite)
    cs = np.linspace(-3, 3, 401); A1 = 0.4; beta1 = 0.3; b1 = 0.1; cT = cs.copy()
    for _ in range(T):
        cT = cT * (1 + sigmoid(A1 * cT + beta1 + b1) / L)
    mono = bool(np.all(np.diff(cT) > 0))
    # variante scalaire : direction du residu fixe
    s = rng.standard_normal(d); bs = 0.2; h = h0.copy()
    for _ in range(T):
        a = sigmoid(h @ s + bs)[:, None]; h = h + (a / L) * (h @ P.T - h)
    rhoT = h @ Q.T; cosang = np.sum(rhoT * rho, 1) / (norm(rhoT, axis=1) * norm(rho, axis=1))
    print("=== A. Reduction de la profondeur gatee ===")
    print(f"  (i)   residu invariant                 : {e_rho:.1e}")
    print(f"  (ii)  recursion du code (dim r={r})     : {e_code:.1e}")
    print(f"  (iii) meme (c0, S^T rho0) -> meme c_T   : {e_iii:.1e}  (residus differents de {3*norm(delta):.2f})")
    print(f"  (iv)  S dans span(W) -> c_T sans residu : {e_iv:.1e}")
    print(f"  (v)   r=1, c0 -> c_T strictement croissant : {mono}")
    print(f"  scalaire : cos(angle(rho_T, rho_0)) = {cosang.min():.12f} (direction du residu fixe)")


# ================================================================== B
def tri_circle(G, eps):
    """deux triceptrons doux (s_i = w_i = e_i) + saut lineaire : f = sum_i (x_i/2) tanh(eps x_i / 2)."""
    return sigmoid(eps * G[:, 0]) * G[:, 0] + sigmoid(eps * G[:, 1]) * G[:, 1] - 0.5 * (G[:, 0] + G[:, 1])


def radial_gap(eps, R=1.0, n_ang=721):
    """ecart radial max entre la courbe de niveau f = eps R^2 / 4 et le cercle de rayon R (bissection par angle)."""
    th = np.linspace(0, 2 * np.pi, n_ang); lo = np.full(n_ang, 0.5 * R); hi = np.full(n_ang, 2.0 * R)
    for _ in range(60):
        mid = 0.5 * (lo + hi); val = tri_circle(np.c_[mid * np.cos(th), mid * np.sin(th)], eps) - eps * R ** 2 / 4
        lo = np.where(val < 0, mid, lo); hi = np.where(val < 0, hi, mid)
    return float(np.abs(0.5 * (lo + hi) - R).max())


def verify_B():
    print("\n=== B. Capacite quadrique du triceptron ===")
    x = np.linspace(-3, 3, 301); X1, X2 = np.meshgrid(x, x); G = np.c_[X1.ravel(), X2.ravel()]
    # (i) inclusion : triceptron(s=w, kappa=1) = swish ; kappa -> inf : ReLU, uniformement, ecart sup = 0.2785 / kappa
    z = np.linspace(-5, 5, 2000001)
    for kap in (1.0, 50.0, 1000.0):
        gap = np.abs(sigmoid(kap * z) * z - np.maximum(z, 0)).max()
        print(f"  (i)  s=w : sup_z |sigma(kappa z) z - ReLU(z)| a kappa={kap:6.0f} : {gap:.3e}  (0.2785/kappa = {0.27846/kap:.3e})")
    # (ii) cercle avec 2 unites en regime doux : sigma(eps t) t = t/2 + eps t^2/4 - eps^3 t^4/48 + ...
    for eps in (0.5, 0.2, 0.05):
        f = tri_circle(G, eps); q = eps / 4 * (G[:, 0] ** 2 + G[:, 1] ** 2)
        print(f"  (ii) cercle R=1, 2 unites + saut (kappa={eps}) : ecart relatif a la quadrique sur [-3,3]^2 {norm(f-q)/norm(q):.1e} ; "
              f"ecart radial max de la courbe de niveau {radial_gap(eps):.2e}  (kappa^2 R^3/24 = {eps**2/24:.2e})")
    eps = 0.05
    # XOR : 1 unite + saut (exact en signe pour tout kappa), ou 2 unites sans saut
    f1 = sigmoid(eps * G[:, 0]) * G[:, 1] - 0.5 * G[:, 1]                               # = (x2/2) tanh(eps x1/2)
    f2 = sigmoid(50 * G[:, 0]) * G[:, 1] - sigmoid(-50 * G[:, 0]) * G[:, 1]              # = x2 tanh(25 x1)
    nz = (G[:, 0] != 0) & (G[:, 1] != 0); truth = (G[:, 0] * G[:, 1]) > 0
    print(f"  (ii) XOR, 1 unite + saut : accord du signe {100*np.mean((f1>0)[nz]==truth[nz]):.2f}% ; 2 unites (regime dur) : {100*np.mean((f2>0)[nz]==truth[nz]):.2f}% (hors axes)")
    # (iv) disque SANS marge (points uniformes dans [-1.5,1.5]^2, classe = ||x|| > 1). Erreur d'encadrement d'une
    #      region D : plus petit e tel que B(0,R-e) c D c B(0,R+e) ; ici e_mes = max | ||x|| - 1 | sur les points
    #      de grille mal classes du domaine d'apprentissage. Borne : e >= R tan^2(pi / 2N), N <= N(m) segments
    #      (N <= 2m si la region est convexe).
    from sklearn.neural_network import MLPClassifier
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-1.5, 1.5, (4000, 2)); ytr = (norm(Xtr, axis=1) > 1.0).astype(int)
    xg = np.linspace(-1.5, 1.5, 601); A1, A2 = np.meshgrid(xg, xg); Gd = np.c_[A1.ravel(), A2.ravel()]
    rad = norm(Gd, axis=1); truth_g = rad > 1.0
    print("  (iv) disque sans marge, grille 601x601 sur [-1.5,1.5]^2, MLP ReLU a m unites (L-BFGS, 3 graines, graine retenue = meilleure precision d'APPRENTISSAGE) :")
    print(f"        {'m':>3s} {'N(m)':>5s} {'erreur de surface':>18s} {'encadrement mesure':>19s} {'borne tan^2(pi/2N(m))':>22s} {'borne convexe tan^2(pi/4m)':>27s}")
    for m in [2, 3, 4, 6, 10, 20, 40]:
        best = None
        for sd in range(3):
            clf = MLPClassifier((m,), solver="lbfgs", max_iter=3000, random_state=sd).fit(Xtr, ytr); tr = clf.score(Xtr, ytr)
            if best is None or tr > best[0]: best = (tr, clf)
        wrong = best[1].predict(Gd) != truth_g; N = 1 + m + m * (m - 1) // 2
        e_mes = np.abs(rad[wrong] - 1.0).max() if wrong.any() else 0.0
        print(f"        {m:3d} {N:5d} {100*wrong.mean():17.2f}% {e_mes:19.4f} {np.tan(np.pi/(2*N))**2:22.5f} {np.tan(np.pi/(4*m))**2:27.5f}")
    wrong = (tri_circle(Gd, eps) > eps / 4) != truth_g
    print(f"        2 triceptrons doux + saut, poids explicites (kappa={eps}) : erreur de surface {100*wrong.mean():.3f}%, encadrement (ecart radial) {radial_gap(eps):.2e}")
    for eps_t in (1e-2, 1e-3, 1e-4):
        Nmin = int(np.ceil(np.pi / np.arccos((1 - eps_t) / (1 + eps_t)))); m = 1
        while 1 + m + m * (m - 1) // 2 < Nmin: m += 1
        print(f"        precision {eps_t:.0e} : N >= {Nmin} segments, donc m >= {m} unites (m >= {int(np.ceil(Nmin/2))} si la region est convexe)")
    # avec marge (make_circles, facteur 0.5) : un triangle suffit -> la marge, pas la courbure, fixe le besoin
    from sklearn.datasets import make_circles
    Xc, yc = make_circles(2000, noise=0.0, factor=0.5, random_state=0)
    accs = [max(MLPClassifier((m,), solver="lbfgs", max_iter=3000, random_state=sd).fit(Xc, yc).score(Xc, yc) for sd in range(3)) for m in [2, 3]]
    print(f"        (avec marge, make_circles f=0.5, precision d'apprentissage : ReLU m=2 -> {accs[0]*100:.1f}%, m=3 -> {accs[1]*100:.1f}% : un triangle suffit quand il y a une marge)")


# ================================================================== C
def num_jac(F, h, eps=1e-6):
    d = len(h); J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps; J[:, j] = (F(h + e) - F(h - e)) / (2 * eps)
    return J


def verify_C(seed=0):
    print("\n=== C. Invariance du code et determinants exacts (differences finies) ===")
    rng = np.random.default_rng(seed); d, r = 12, 3
    U = rng.standard_normal((d, r)); W = rng.standard_normal((d, r)); W = W @ np.linalg.inv(U.T @ W)   # W^T U = I_r
    P = U @ W.T; Q = np.eye(d) - P; x = rng.standard_normal(d); b = 0.3
    # invariance du code et collapse de la porte (recursion h <- (1-a) h + a P h)
    def rec(s, T):
        h = x.copy(); a_hist = []
        for _ in range(T):
            a = sigmoid(s @ h + b); a_hist.append(a); h = (1 - a) * h + a * (P @ h)
        return h, np.array(a_hist)
    s_in = W @ rng.standard_normal(r)                               # porte dans span(W)
    s_out = Q.T @ rng.standard_normal(d); s_out *= 1.5 / abs(s_out @ x)   # porte dans (span U)^perp : lit seulement le residu ; |s^T x| = 1.5
    e_code, e_form = 0.0, 0.0
    for T in (1, 4, 16):
        hT, a_h = rec(s_in, T); a = a_h[0]
        e_code = max(e_code, norm(W.T @ hT - W.T @ x)); e_form = max(e_form, norm(hT - (P @ x + (1 - a) ** T * (Q @ x))))
        e_code = max(e_code, norm(W.T @ rec(s_out, T)[0] - W.T @ x))
    a_out = rec(s_out, 16)[1]
    print(f"  code : max ||W^T h_T - W^T x|| = {e_code:.1e} (T = 1, 4, 16, portes dans et hors de span(W)) ; forme fermee de h_T (porte dans span(W)) : {e_form:.1e}")
    print(f"  porte hors de span(W) : a_t varie de {a_out.max() - a_out.min():.3f} le long de l'orbite (T = 16) ; porte dans span(W) : variation {np.ptp(rec(s_in, 16)[1]):.0e}")
    # determinant du bloc gate scalaire F(h) = h + (a(h)/L)(P - I) h
    def det_formula(h, s, L):
        a = sigmoid(s @ h + b); return (1 - a / L) ** (d - r) * (1 - a * (1 - a) / (L - a) * (s @ Q @ h))
    def F_of(s, L): return lambda h: h + (sigmoid(s @ h + b) / L) * (P @ h - h)
    for L in (1, 2):
        s = 0.5 * rng.standard_normal(d); h = rng.standard_normal(d)
        print(f"  bloc gate scalaire, L={L}, s quelconque : det formule = {det_formula(h, s, L):.7e}, differences finies = {np.linalg.det(num_jac(F_of(s, L), h)):.7e}")
    h = rng.standard_normal(d); a = sigmoid(s_in @ h + b)
    print(f"  s dans span(W), L=1 : det differences finies = {np.linalg.det(num_jac(F_of(s_in, 1), h)):.7e}, (1-a)^(d-r) = {(1 - a) ** (d - r):.7e}")
    # cas construit a s^T (I-P) h = 1 (L = 1) : s = t q avec q = (I-P)h, t solution de sigma(t |q|^2 + b) t |q|^2 = 1
    q = Q @ h; q2 = q @ q; lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sigmoid(mid * q2 + b) * mid * q2 < 1: lo = mid
        else: hi = mid
    s_sing = (Q.T @ Q @ h) * lo / (q @ Q.T @ Q @ h) * q2            # s^T h = s^T (I-P) h = lo |q|^2
    a = sigmoid(s_sing @ h + b)
    print(f"  cas construit a * s^T(I-P)h = {a * (s_sing @ Q @ h):.6f} : det differences finies = {np.linalg.det(num_jac(F_of(s_sing, 1), h)):.1e} (formule : 0)")
    # determinant r x r du bloc gate par coordonnee
    d2 = 10; U2 = rng.standard_normal((d2, r)); W2 = rng.standard_normal((d2, r)); W2 = W2 @ np.linalg.inv(U2.T @ W2)
    S2 = 0.5 * rng.standard_normal((d2, r)); b2 = 0.2 * rng.standard_normal(r); L = 2; A2 = S2.T @ U2
    Fv = lambda h: h + (1.0 / L) * U2 @ (sigmoid(S2.T @ h + b2) * (W2.T @ h))
    for _ in range(3):
        h = rng.standard_normal(d2); c = W2.T @ h; a = sigmoid(S2.T @ h + b2)
        small = np.linalg.det(np.eye(r) + (np.diag(a) + np.diag(c * a * (1 - a)) @ A2) / L)
        print(f"  bloc gate par coordonnee (d={d2}, r={r}, L={L}) : det r x r = {small:.8f}, differences finies d x d = {np.linalg.det(num_jac(Fv, h)):.8f}")
    # rang un : contre-exemple a la monotonie hors condition, et constante optimale
    phi = lambda c, L, al, be: c * (1 + sigmoid(al * c + be) / L)
    print(f"  rang un, L=1, alpha=1, b=0, beta=12 : phi(-16) = {phi(-16, 1, 1, 12):.2f}, phi(-12) = {phi(-12, 1, 1, 12):.2f}, phi(-8) = {phi(-8, 1, 1, 12):.2f} (non monotone : alpha c < -4L)")
    for L in (1, 2, 4):
        t = np.linspace(-30, 0, 30001); u = np.linspace(-30, 30, 6001)[:, None]        # t = alpha c <= 0, u = alpha c + beta + b
        dphi = 1 + sigmoid(u) / L + t[None, :] * sigmoid(u) * (1 - sigmoid(u)) / L
        t_star = -t[np.argmax(dphi.min(0) > 0)]
        print(f"  rang un, L={L} : phi' > 0 pour tout beta des que alpha c > -{t_star:.2f}  (2L+1+2 sqrt(L(L+1)) = {2*L+1+2*np.sqrt(L*(L+1)):.2f} ; condition suffisante 4L = {4*L})")


if __name__ == "__main__":
    verify_A()
    verify_B()
    verify_C()
