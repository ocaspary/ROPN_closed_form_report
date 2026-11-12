"""
Reinjection par retropropagation : fine-tuning du bloc unique a partir de son
initialisation fermee (warm start de Fisher), en numpy avec gradients manuels
(verifies par differences finies).

Modele fine-tune (T applications du meme bloc, poids partages) :
    Uo  : base orthonormee de span(U_ferme)  -- FIXEE (representation)
    W   = Uo + (I - Uo Uo^T) M                -- M appris : W^T Uo = I_r reste EXACT
    a_t = sigma(h_t s + b)                    -- porte, (s, b) appris
    h_{t+1} = (1 - a_t) h_t + a_t (h_t W) Uo^T
    logits = h_T Wo + bo                      -- lecture apprise sur l'etat complet (dim d)
    perte = entropie croisee + wd * (||M||^2 + ||Wo||^2)

Initialisation fermee : le bloc CORRIGE (retrecissement, rang C-1, lecture par centroides sur le code,
c'est-a-dire la regle LDA), ecrit comme une lecture affine sur h_T dont les poids sont dans span(W)
(Wo = W M^T, bo = -||m_k||^2/2 + log pi_k) : le code W^T h_T etant invariant par la recursion, avant tout
pas de gradient le modele fine-tune predit exactement comme le bloc corrige, et ses logits sont les
scores discriminants de la LDA.
Variantes du modele contraint (Uo FIXE dans tous les cas : seul le sous-espace de mesure peut bouger) :
(M, porte, lecture) avec deux reglages wd / nombre de pas, lecture seule, porte + lecture (M fixe),
M + lecture (porte fixe), et initialisation aleatoire du meme modele aux memes reglages.

Modele LIBRE (les DEUX sous-espaces evoluent) : U et W tous deux appris,
    h_{t+1} = (1 - a_t) h_t + a_t (h_t W) U^T,
dans un plan factoriel 2 x 2 x 2 qui separe trois effets : initialisation (bloc ferme / aleatoire),
contrainte geometrique (P = U (W^T U)^{-1} W^T, idempotent exactement a chaque pas, le gradient traversant
l'inverse ; ou P = U W^T sans contrainte), regularisation (wd 1e-3, 300 pas / wd 1e-1, 100 pas). La penalite porte sur la
lecture et sur l'ecart de (U, W) a leur initialisation.

Chaque variante est comparee au bloc ferme par differences APPARIEES sur les memes tirages (intervalle
de confiance a 95 %, marge d'equivalence de 1 point : voir ropn_auto_stats.py).

Deux jeux de donnees :
    gauss  : covariance partagee (LDA = Bayes) -> la forme fermee est optimale en population
    hetero : covariances differentes par classe (LDA mal specifiee) -> information hors span(W)
"""
import numpy as np
from numpy.linalg import qr, norm, pinv, slogdet
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from ropn_auto_nobp import make_multiclass, make_closed_form_gate, recurse_gated, sigmoid
from ropn_auto_nobp_diag import fit_block, readout_centroid_fit_predict, centroid_readout_weights
from ropn_auto_stats import fmt_paired


# ---------------------------------------------------------------- donnees heteroscedastiques
def make_hetero(n_classes=4, d=20, n_per=60, kappa=50.0, sep=0.8, seed=0):
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, np.log10(kappa), d); eig = eig / eig.mean()
    mus = sep * rng.standard_normal((n_classes, d)); mus -= mus.mean(0, keepdims=True)
    X = np.zeros((n_classes * n_per, d)); y = np.zeros(n_classes * n_per, dtype=int); Sigmas = []
    for c in range(n_classes):
        Q, _ = qr(rng.standard_normal((d, d)))           # orientation PROPRE a chaque classe
        Sigma_c = Q @ np.diag(eig) @ Q.T; Sigmas.append(Sigma_c)
        L = np.linalg.cholesky(Sigma_c + 1e-9 * np.eye(d))
        X[c*n_per:(c+1)*n_per] = rng.standard_normal((n_per, d)) @ L.T + mus[c]
        y[c*n_per:(c+1)*n_per] = c
    return X, y, Sigmas, mus


def oracle_qda(Sigmas, mus, X):
    sc = []
    for S, m in zip(Sigmas, mus):
        Si = pinv(S); Xc = X - m
        sc.append(-0.5 * np.einsum("ij,jk,ik->i", Xc, Si, Xc) - 0.5 * slogdet(S)[1])
    return np.stack(sc, 1).argmax(1)


# ---------------------------------------------------------------- modele + gradients manuels
def forward(params, Uo, X, T):
    M, s, b, Wo, bo = params
    d, r = Uo.shape
    Pperp = np.eye(d) - Uo @ Uo.T
    W = Uo + Pperp @ M
    hs, As, Ps = [X], [], []
    h = X
    for _ in range(T):
        a = sigmoid(h @ s + b)[:, None]
        p = (h @ W) @ Uo.T
        h = (1 - a) * h + a * p
        As.append(a); Ps.append(p); hs.append(h)
    logits = h @ Wo + bo
    return logits, (W, Pperp, hs, As, Ps)


def loss_and_grads(params, Uo, X, y, T, wd):
    M, s, b, Wo, bo = params
    n = len(y)
    logits, (W, Pperp, hs, As, Ps) = forward(params, Uo, X, T)
    z = logits - logits.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    loss = -np.log(pr[np.arange(n), y] + 1e-12).mean() + wd * (np.sum(M**2) + np.sum(Wo**2))
    dlog = pr.copy(); dlog[np.arange(n), y] -= 1.0; dlog /= n
    hT = hs[-1]
    dWo = hT.T @ dlog + 2 * wd * Wo; dbo = dlog.sum(0)
    dh = dlog @ Wo.T
    dW = np.zeros_like(W); ds = np.zeros_like(s); db = 0.0
    for t in reversed(range(T)):
        h, a, p = hs[t], As[t], Ps[t]
        da = np.sum(dh * (p - h), 1, keepdims=True)
        dz = da * a * (1 - a)
        ds += h.T @ dz[:, 0]; db += dz.sum()
        dp = a * dh
        dW += h.T @ (dp @ Uo)
        dh_new = (1 - a) * dh + (dp @ Uo) @ W.T + dz * s[None, :]
        dh = dh_new
    dM = Pperp @ dW + 2 * wd * M
    return loss, (dM, ds, db, dWo, dbo)


def gradcheck(seed=0):
    rng = np.random.default_rng(seed)
    d, r, C, n, T = 6, 2, 3, 7, 3
    Uo, _ = qr(rng.standard_normal((d, r)))
    params = [rng.standard_normal((d, r)) * 0.3, rng.standard_normal(d) * 0.3, 0.1,
              rng.standard_normal((d, C)) * 0.3, rng.standard_normal(C) * 0.1]
    X = rng.standard_normal((n, d)); y = rng.integers(0, C, n)
    loss, grads = loss_and_grads(params, Uo, X, y, T, wd=1e-3)
    eps = 1e-6; worst = 0.0
    for i, p in enumerate(params):
        g = grads[i]
        if np.ndim(p) == 0:
            pp = list(params); pp[i] = p + eps; lp = loss_and_grads(pp, Uo, X, y, T, 1e-3)[0]
            pp[i] = p - eps; lm = loss_and_grads(pp, Uo, X, y, T, 1e-3)[0]
            worst = max(worst, abs((lp - lm) / (2 * eps) - g) / (abs(g) + 1e-8))
        else:
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                pp = [q.copy() if np.ndim(q) else q for q in params]
                pp[i][idx] += eps; lp = loss_and_grads(pp, Uo, X, y, T, 1e-3)[0]
                pp[i][idx] -= 2 * eps; lm = loss_and_grads(pp, Uo, X, y, T, 1e-3)[0]
                worst = max(worst, abs((lp - lm) / (2 * eps) - g[idx]) / (abs(g[idx]) + 1e-8))
    return worst


def adam_train(params, Uo, X, y, T, steps=300, lr=1e-2, wd=1e-3, train=(1, 1, 1, 1, 1)):
    """train : masque (M, s, b, Wo, bo) des groupes de parametres appris ; les autres restent figes."""
    params = [p.copy() if np.ndim(p) else float(p) for p in params]
    m = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]
    v = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(1, steps + 1):
        _, g = loss_and_grads(params, Uo, X, y, T, wd)
        for i in range(len(params)):
            if not train[i]: continue
            m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * (g[i] ** 2)
            mh = m[i] / (1 - b1 ** k); vh = v[i] / (1 - b2 ** k)
            params[i] = params[i] - lr * mh / (np.sqrt(vh) + eps)
    return params


def predict(params, Uo, X, T):
    return forward(params, Uo, X, T)[0].argmax(1)


# ---------------------------------------------------------------- modele libre : U et W appris tous les deux
def forward_free(params, X, T, idem):
    """idem=True : p = h W (U^T W)^{-1} U^T, projecteur oblique EXACTEMENT idempotent pour tous (U, W) tels que
    U^T W soit inversible ; idem=False : p = h W U^T, application de rang r sans contrainte."""
    U, W, s, b, Wo, bo = params
    Gi = np.linalg.inv(U.T @ W) if idem else None
    hs, As, Ps, Bs = [X], [], [], []; h = X
    for _ in range(T):
        a = sigmoid(h @ s + b)[:, None]; B = (h @ W) @ Gi if idem else h @ W; p = B @ U.T
        h = (1 - a) * h + a * p
        As.append(a); Ps.append(p); Bs.append(B); hs.append(h)
    return h @ Wo + bo, (hs, As, Ps, Bs, Gi)


def loss_and_grads_free(params, ref, X, y, T, wd, idem):
    """perte = entropie croisee + wd (||Wo||^2 + ||U - U_ref||^2 + ||W - W_ref||^2), ref = (U_ref, W_ref)."""
    U, W, s, b, Wo, bo = params; n = len(y)
    logits, (hs, As, Ps, Bs, Gi) = forward_free(params, X, T, idem)
    z = logits - logits.max(1, keepdims=True); pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
    loss = -np.log(pr[np.arange(n), y] + 1e-12).mean() + wd * (np.sum(Wo**2) + np.sum((U - ref[0])**2) + np.sum((W - ref[1])**2))
    dlog = pr.copy(); dlog[np.arange(n), y] -= 1.0; dlog /= n
    dWo = hs[-1].T @ dlog + 2 * wd * Wo; dbo = dlog.sum(0); dh = dlog @ Wo.T
    dU = np.zeros_like(U); dW = np.zeros_like(W); ds = np.zeros_like(s); db = 0.0
    for t in reversed(range(T)):
        h, a, p, B = hs[t], As[t], Ps[t], Bs[t]
        dz = np.sum(dh * (p - h), 1, keepdims=True) * a * (1 - a)
        ds += h.T @ dz[:, 0]; db += dz.sum()
        dp = a * dh; dB = dp @ U; dU += dp.T @ B
        if idem:                                   # B = (h W) G^{-1}, G = U^T W
            dA = dB @ Gi.T; dG = -(B.T @ dB) @ Gi.T
            dU += W @ dG.T; dW += U @ dG
        else:
            dA = dB
        dW += h.T @ dA
        dh = (1 - a) * dh + dA @ W.T + dz * s[None, :]
    return loss, (dU + 2 * wd * (U - ref[0]), dW + 2 * wd * (W - ref[1]), ds, db, dWo, dbo)


def gradcheck_free(idem, seed=0):
    rng = np.random.default_rng(seed); d, r, C, n, T = 6, 2, 3, 7, 3
    U0 = rng.standard_normal((d, r)) * 0.5
    params = [U0, U0 + 0.3 * rng.standard_normal((d, r)), rng.standard_normal(d) * 0.3, 0.1,
              rng.standard_normal((d, C)) * 0.3, rng.standard_normal(C) * 0.1]
    ref = (rng.standard_normal((d, r)) * 0.5, rng.standard_normal((d, r)) * 0.5)
    X = rng.standard_normal((n, d)); y = rng.integers(0, C, n)
    _, grads = loss_and_grads_free(params, ref, X, y, T, 1e-3, idem); eps = 1e-6; worst = 0.0
    for i, p in enumerate(params):
        g = grads[i]
        if np.ndim(p) == 0:
            pp = list(params); pp[i] = p + eps; lp = loss_and_grads_free(pp, ref, X, y, T, 1e-3, idem)[0]
            pp[i] = p - eps; lm = loss_and_grads_free(pp, ref, X, y, T, 1e-3, idem)[0]
            worst = max(worst, abs((lp - lm) / (2 * eps) - g) / (abs(g) + 1e-8))
        else:
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index; pp = [q.copy() if np.ndim(q) else q for q in params]
                pp[i][idx] += eps; lp = loss_and_grads_free(pp, ref, X, y, T, 1e-3, idem)[0]
                pp[i][idx] -= 2 * eps; lm = loss_and_grads_free(pp, ref, X, y, T, 1e-3, idem)[0]
                worst = max(worst, abs((lp - lm) / (2 * eps) - g[idx]) / (abs(g[idx]) + 1e-8))
    return worst


def adam_train_free(params, X, y, T, steps=300, lr=1e-2, wd=1e-3, idem=True):
    """Adam sur (U, W, s, b, Wo, bo) ; les deux sous-espaces evoluent. idem=True : l'idempotence est exacte a
    chaque pas par construction (le gradient traverse (U^T W)^{-1}) ; idem=False : aucune contrainte."""
    params = [p.copy() if np.ndim(p) else float(p) for p in params]; ref = (params[0].copy(), params[1].copy())
    m = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]; v = [np.zeros_like(p) if np.ndim(p) else 0.0 for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(1, steps + 1):
        _, g = loss_and_grads_free(params, ref, X, y, T, wd, idem)
        for i in range(len(params)):
            m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * (g[i] ** 2)
            params[i] = params[i] - lr * (m[i] / (1 - b1 ** k)) / (np.sqrt(v[i] / (1 - b2 ** k)) + eps)
    return params


def init_free_closed_form(U, W, Xtr, ytr):
    """bloc corrige ecrit dans le modele libre : (U, W) biorthogonaux, porte fermee, lecture par centroides
    remontee sur h_T (Wo = W M^T) : h W est invariant par la recursion, donc predictions initiales = bloc."""
    s0, b0 = make_closed_form_gate(U, W, Xtr)
    _, V = centroid_readout_weights(Xtr @ W, ytr)
    return [U.copy(), W.copy(), s0.copy(), float(b0), W @ V[:-1], V[-1].copy()]


def init_free_random(d, r, C, rng):
    U, _ = qr(rng.standard_normal((d, r)))
    return [U.copy(), U + 0.1 * rng.standard_normal((d, r)), rng.standard_normal(d) * 0.1, 0.0, rng.standard_normal((d, C)) * 0.1, np.zeros(C)]


# ---------------------------------------------------------------- inits
def init_closed_form(U, W, Xtr, ytr, T):
    """Warm start de Fisher = bloc corrige : Uo = QR(U), M tel que W_param = W R^T exactement,
    porte fermee, lecture par centroides sur le code W^T x remontee sur h_T (Wo = W M^T) : le projecteur
    Uo W_param^T est egal a U W^T, donc W^T h_T = W^T x et les predictions initiales sont celles du bloc."""
    Uo, R = qr(U)
    Wp = W @ R.T                                   # W_p^T Uo = I_r
    M0 = Wp - Uo                                   # (I - Uo Uo^T) M0 = Wp - Uo exactement
    s0, b0 = make_closed_form_gate(U, W, Xtr)
    _, V = centroid_readout_weights(Xtr @ W, ytr)
    return Uo, [M0, s0.copy(), float(b0), W @ V[:-1], V[-1].copy()]


def init_random(d, r, C, rng):
    Uo, _ = qr(rng.standard_normal((d, r)))
    return Uo, [rng.standard_normal((d, r)) * 0.1, rng.standard_normal(d) * 0.1, 0.0,
                rng.standard_normal((d, C)) * 0.1, np.zeros(C)]


# ---------------------------------------------------------------- experience
VARIANTS = [  # modele contraint, Uo fixe : (nom, init, masque (M, s, b, Wo, bo), wd, pas)
    ("Uo fixe | ferme -> M, porte, lecture, wd 1e-3, 300 pas", "ferme", (1, 1, 1, 1, 1), 1e-3, 300),
    ("Uo fixe | ferme -> M, porte, lecture, wd 1e-1, 100 pas", "ferme", (1, 1, 1, 1, 1), 1e-1, 100),
    ("Uo fixe | ferme -> lecture seule, wd 1e-3, 300 pas",     "ferme", (0, 0, 0, 1, 1), 1e-3, 300),
    ("Uo fixe | ferme -> porte, lecture, wd 1e-3, 300 pas",    "ferme", (0, 1, 1, 1, 1), 1e-3, 300),
    ("Uo fixe | ferme -> M, lecture, wd 1e-3, 300 pas",        "ferme", (1, 0, 0, 1, 1), 1e-3, 300),
    ("Uo fixe | alea  -> M, porte, lecture, wd 1e-3, 300 pas", "alea",  (1, 1, 1, 1, 1), 1e-3, 300),
    ("Uo fixe | alea  -> M, porte, lecture, wd 1e-1, 100 pas", "alea",  (1, 1, 1, 1, 1), 1e-1, 100),
]
FREE_VARIANTS = [  # modele libre, U et W appris : (nom, init, retraction, wd, pas) -- plan factoriel 2 x 2 x 2
    (f"U, W libres | {ini:5s} | {'idempotent     ' if rt else 'sans contrainte'} | wd {wd:g}, {st} pas", ini, rt, wd, st)
    for ini in ("ferme", "alea") for rt in (True, False) for wd, st in ((1e-3, 300), (1e-1, 100))
]


def run(dataset, kappa, n_per, seeds=20, T=4, lr=1e-2, d=20, C=4):
    keys = ["oracle", "LDA-shrink", "MLP", "ferme corrige (shrink+code+C-1)", "ferme, lecture h_T (init. du fine-tuning)"] + [v[0] for v in VARIANTS] + [v[0] for v in FREE_VARIANTS]
    acc = {k: [] for k in keys}
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        if dataset == "gauss":
            X, y, Sigma, mus = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=kappa, sep=0.8, seed=sd)
            Sigmas = [Sigma] * C
        else:
            X, y, Sigmas, mus = make_hetero(n_classes=C, d=d, n_per=n_per, kappa=kappa, sep=0.8, seed=sd)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=sd)
        acc["oracle"].append((oracle_qda(Sigmas, mus, Xte) == yte).mean())
        acc["LDA-shrink"].append(LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).score(Xte, yte))
        acc["MLP"].append(MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=sd).fit(Xtr, ytr).score(Xte, yte))

        Ub, Wb, _ = fit_block(Xtr, ytr, shrink=True, rank="C1", seed=sd)
        acc["ferme corrige (shrink+code+C-1)"].append((readout_centroid_fit_predict(Xtr @ Wb, ytr, Xte @ Wb) == yte).mean())

        Uo, p0 = init_closed_form(Ub, Wb, Xtr, ytr, T)
        acc["ferme, lecture h_T (init. du fine-tuning)"].append((predict(p0, Uo, Xte, T) == yte).mean())
        Uo_r, pr0 = init_random(d, Ub.shape[1], C, rng)
        for name, init, mask, wd, steps in VARIANTS:
            U_, p_ = (Uo, p0) if init == "ferme" else (Uo_r, pr0)
            p1 = adam_train(p_, U_, Xtr, ytr, T, steps, lr, wd, train=mask)
            acc[name].append((predict(p1, U_, Xte, T) == yte).mean())
        f0 = init_free_closed_form(Ub, Wb, Xtr, ytr); fr0 = init_free_random(d, Ub.shape[1], C, rng)
        for name, init, retract, wd, steps in FREE_VARIANTS:
            pf = adam_train_free(f0 if init == "ferme" else fr0, Xtr, ytr, T, steps, lr, wd, idem=retract)
            acc[name].append((forward_free(pf, Xte, T, retract)[0].argmax(1) == yte).mean())
    print(f"\n=== {dataset}, kappa={kappa:.0f}, n={n_per}/classe, T={T}, {seeds} graines ===")
    ref = acc["ferme corrige (shrink+code+C-1)"]
    print(f"  {'methode':66s} {'precision':>14s}   difference appariee au bloc ferme : moyenne [IC 95 %] verdict (marge 1 point)")
    for k in keys:
        print(f"  {k:66s} {np.mean(acc[k])*100:5.1f} +/- {np.std(acc[k])*100:4.1f}   {fmt_paired(acc[k], ref) if k != 'ferme corrige (shrink+code+C-1)' else ''}")
    # effets principaux du plan factoriel 2 x 2 x 2 du modele libre : pour chaque graine, moyenne des variantes d'un
    # niveau, puis difference appariee entre les deux niveaux (memes graines, memes decoupages)
    def level(pred):
        return np.mean([acc[v[0]] for v in FREE_VARIANTS if pred(v)], axis=0)
    print("  effets du plan factoriel (U, W libres), differences appariees :")
    print(f"    initialisation : ferme - alea (moyenne sur contrainte et regularisation)        {fmt_paired(level(lambda v: v[1] == 'ferme'), level(lambda v: v[1] == 'alea'))}")
    print(f"    contrainte : idempotent - sans contrainte, depuis l'initialisation fermee        {fmt_paired(level(lambda v: v[1] == 'ferme' and v[2]), level(lambda v: v[1] == 'ferme' and not v[2]))}")
    print(f"    contrainte : idempotent - sans contrainte, depuis l'initialisation aleatoire     {fmt_paired(level(lambda v: v[1] == 'alea' and v[2]), level(lambda v: v[1] == 'alea' and not v[2]))}")
    print(f"    regularisation : (wd 1e-3, 300 pas) - (wd 1e-1, 100 pas), initialisation fermee  {fmt_paired(level(lambda v: v[1] == 'ferme' and v[3] == 1e-3), level(lambda v: v[1] == 'ferme' and v[3] == 1e-1))}")
    print(f"    regularisation : (wd 1e-3, 300 pas) - (wd 1e-1, 100 pas), initialisation aleatoire {fmt_paired(level(lambda v: v[1] == 'alea' and v[3] == 1e-3), level(lambda v: v[1] == 'alea' and v[3] == 1e-1))}")
    return acc


if __name__ == "__main__":
    print(f"gradcheck (erreur relative max) = {gradcheck():.2e} ; modele libre idempotent : {gradcheck_free(True):.2e}, sans contrainte : {gradcheck_free(False):.2e}")
    run("gauss", 10.0, 15)
    run("gauss", 200.0, 15)
    run("hetero", 50.0, 15)
    run("hetero", 50.0, 60)
