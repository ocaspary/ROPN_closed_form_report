"""
Operateurs de Fourier dans le ROPN auto sans retropropagation.

Un bloc spectral de FNO est K = F^{-1} R F avec R bloc-diagonal par mode : R(k) in C^{CxC}. Sans canaux (C=1)
c'est un bloc circulant (un gain complexe par frequence) ; il n'est idempotent que si chaque gain
vaut 0 ou 1 (masque de modes). Avec canaux, chaque R(k) peut etre un projecteur oblique de rang r_k <= C :
la geometrie oblique vit DANS chaque mode, et les K problemes sont independants (parallelisme trivial).

Forme fermee : par mode k, un probleme C x C (moindres carres pour un operateur, GEVD de Fisher pour un
classifieur), plus un test de permutation par mode qui met r_k = 0 sur les modes sans information :
n_modes, hyperparametre du FNO souvent mal regle selon Duruisseaux, Kossaifi et Anandkumar (2025),
peut alors etre fixe par les donnees. Attention : dans E1 et E2 les modes ecartes sont ceux d'energie
nulle (condition initiale a bande limitee, de-aliasing 2/3), elimines par le garde-fou d'energie ; le
test de permutation n'en retire aucun de plus. La selection par test n'est demontree que dans E3.

Trois experiences :
  E1. EDP lineaire periodique (advection-diffusion) : le bloc spectral ferme retrouve exactement le
      propagateur exp((-nu k^2 - i c k) dt) ; FNO (retropropagation, numpy) et MLP l'approchent.
  E2. Burgers (non lineaire) : bloc spectral lineaire ferme, bloc spectral QUADRATIQUE ferme (dictionnaire
      [u, u^2, u^3] de-aliase, = EDMD en base de Fourier, Petrov-Galerkin de rang 1 par mode), FNO numpy
      a gradients manuels (verifies par differences finies), MLP ; erreur a 1 pas et en deroulement.
  E3. Classification : covariances par bande de frequence + bloc fast avec rang
      par permutation : les bandes sans information recoivent r=0 (selection automatique de bande).
numpy / scipy / sklearn seulement, ASCII seulement.
"""
import time
import numpy as np
from numpy.linalg import norm
from scipy.special import erf
from sklearn.neural_network import MLPRegressor, MLPClassifier
from ropn_auto_fast import fit_block_fast, fwer_select
from ropn_auto_nobp_diag import readout_fit_predict


# ====================================================================== EDP spectrales 1D periodiques
def grf_init(rng, N, kmax=6, s=1.5, n=1):
    """Conditions initiales : serie de Fourier aleatoire a spectre decroissant (champ gaussien)."""
    k = np.arange(N // 2 + 1); U = np.zeros((n, N // 2 + 1), complex)
    for i in range(n):
        amp = (1 + k ** 2) ** (-s / 2) * (k <= kmax)
        U[i] = amp * (rng.standard_normal(N // 2 + 1) + 1j * rng.standard_normal(N // 2 + 1)); U[i, 0] = 0
    u = np.fft.irfft(U, n=N, axis=-1) * N
    return u / np.abs(u).max(axis=-1, keepdims=True)


def advdiff_step(u, dt, c=1.0, nu=0.02):
    N = u.shape[-1]; k = np.fft.rfftfreq(N, 1.0 / N)
    return np.fft.irfft(np.fft.rfft(u, axis=-1) * np.exp((-nu * k ** 2 - 1j * c * k) * dt), n=N, axis=-1)


def burgers_step(u, dt, nu=0.05, n_sub=50):
    """u_t + u u_x = nu u_xx, pseudo-spectral, facteur integrant + RK4 sur le terme non lineaire, 2/3 de-aliasing."""
    N = u.shape[-1]; k = np.fft.rfftfreq(N, 1.0 / N); h = dt / n_sub
    dealias = (np.abs(k) < N / 3.0); E = np.exp(-nu * k ** 2 * h / 2); E2 = E ** 2
    def NL(uh):
        ur = np.fft.irfft(uh, n=N, axis=-1)
        return -0.5j * k * np.fft.rfft(ur * ur, axis=-1) * dealias
    uh = np.fft.rfft(u, axis=-1)
    for _ in range(n_sub):
        a = NL(uh); b = NL(E * (uh + h / 2 * a)); c_ = NL(E * uh + h / 2 * b); d = NL(E2 * uh + h * E * c_)
        uh = E2 * uh + h / 6 * (E2 * a + 2 * E * (b + c_) + d)
    return np.fft.irfft(uh, n=N, axis=-1)


def make_pairs(step, rng, n_traj, n_steps, N, dt, **kw):
    u0 = grf_init(rng, N, n=n_traj); X, Y, traj = [], [], [u0]
    u = u0
    for _ in range(n_steps):
        v = step(u, dt, **kw); X.append(u); Y.append(v); traj.append(v); u = v
    return np.concatenate(X), np.concatenate(Y), np.stack(traj, 1)   # (n_traj*n_steps, N), idem, (n_traj, n_steps+1, N)


def rel_l2(pred, true): return norm(pred - true) / norm(true)


def rollout_err(model_step, traj):
    """Erreur relative L2 moyenne sur l'horizon en deroulant depuis traj[:, 0]."""
    u = traj[:, 0]; errs = []
    for t in range(1, traj.shape[1]):
        u = model_step(u); errs.append(rel_l2(u, traj[:, t]))
    return float(np.mean(errs)), float(errs[-1])


# ====================================================================== blocs spectraux en forme fermee
def powers_dealiased(u, m):
    """[u, u^2, ..., u^m] calcules sur une grille 2x plus fine (zero-padding spectral) puis re-tronques."""
    N = u.shape[-1]; uh = np.fft.rfft(u, axis=-1); M = 2 * N
    up = np.fft.irfft(uh, n=M, axis=-1) * (M / N)
    feats = []
    for p in range(1, m + 1):
        fh = np.fft.rfft(up ** p, axis=-1) * (N / M)
        feats.append(fh[..., :N // 2 + 1])
    return np.stack(feats, -1)                                       # (n, K_all, m) complexe


class SpectralClosedForm:
    """Par mode k : y_hat_k ~ w_k^H z_k, z_k = Fourier des puissances [u, ..., u^m] (m=1 : lineaire, gain complexe
    par mode = bloc circulant). Moindres carres complexes avec ridge ; test de permutation par mode
    (n_perm appariements aleatoires) : les modes dont la variance expliquee ne depasse pas le quantile 1-alpha
    du nul recoivent w_k = 0 (rang 0). Deroulement : re-lift a chaque pas."""
    def __init__(self, m=1, ridge=1e-6, n_perm=100, alpha=0.05, seed=0):
        self.m, self.ridge, self.n_perm, self.alpha, self.seed = m, ridge, n_perm, alpha, seed

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed); Z = powers_dealiased(X, self.m); Yh = np.fft.rfft(Y, axis=-1)
        n, K, m = Z.shape; self.N = X.shape[-1]; self.w = np.zeros((K, m), complex); self.keep = np.zeros(K, bool)
        self.r2 = np.zeros(K); self.r2_null = np.zeros(K)
        for k in range(K):
            z, y = Z[:, k, :], Yh[:, k]
            if norm(y) < 1e-12 * max(1.0, norm(Yh)): continue
            G = z.conj().T @ z + self.ridge * np.trace(z.conj().T @ z).real / m * np.eye(m)
            w = np.linalg.solve(G, z.conj().T @ y); r2 = 1 - norm(y - z @ w) ** 2 / norm(y) ** 2
            null = []
            for _ in range(self.n_perm):
                zp = z[rng.permutation(n)]; wp = np.linalg.solve(zp.conj().T @ zp + G - z.conj().T @ z, zp.conj().T @ y)
                null.append(1 - norm(y - zp @ wp) ** 2 / norm(y) ** 2)
            self.r2[k], self.r2_null[k] = r2, np.quantile(null, 1 - self.alpha)
            if r2 > self.r2_null[k]: self.w[k] = w; self.keep[k] = True
        return self

    def predict(self, X):
        Z = powers_dealiased(X, self.m); Yh = np.einsum("nkm,km->nk", Z, self.w)
        return np.fft.irfft(Yh, n=self.N, axis=-1)

    def n_params(self): return int(2 * self.keep.sum() * self.m)
    def n_modes(self): return int(self.keep.sum())


# ====================================================================== FNO 1D numpy, gradients manuels
def gelu_phi(x): return 0.5 * (1 + erf(x / np.sqrt(2)))                 # Phi(x) ; GELU = x Phi(x)
def dgelu(x, phi): return phi + x * np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)


class FNO1dNumpy:
    """FNO 1D : lift (2->C) -> L x [conv spectrale (K modes, C x C complexe) + saut lineaire + biais, GELU] ->
    projection (C->1), sortie residuelle y = u + Q(h). DFT reelle explicite par matrices (K x N) pour des
    gradients exacts et verifiables."""
    def __init__(self, N, C=16, K=12, L=2, seed=0):
        rng = np.random.default_rng(seed); self.N, self.C, self.K, self.L = N, C, K, L
        n = np.arange(N); k = np.arange(K)[:, None]
        self.Cm = np.cos(2 * np.pi * k * n / N); self.Sm = -np.sin(2 * np.pi * k * n / N)           # (K, N)
        d = np.where(np.arange(K) == 0, 1.0, 2.0)[:, None]; self.Ci = d * self.Cm / N; self.Si = d * self.Sm / N
        s = 1.0 / np.sqrt(C)
        self.p = {"P": rng.standard_normal((2, C)) * 0.5, "p0": np.zeros(C), "q": rng.standard_normal(C) * s, "q0": np.zeros(1)}
        for l in range(L):
            self.p[f"Rr{l}"] = rng.standard_normal((K, C, C)) * s; self.p[f"Ri{l}"] = rng.standard_normal((K, C, C)) * s
            self.p[f"W{l}"] = rng.standard_normal((C, C)) * s; self.p[f"b{l}"] = np.zeros(C)

    def forward(self, X, cache=False):
        p = self.p; a0 = np.stack([X, np.ones_like(X)], 1); v = p["P"].T[None] @ a0 + p["p0"][None, :, None]
        st = [(a0, v)]
        for l in range(self.L):
            vr, vi = (v @ self.Cm.T).transpose(2, 0, 1), (v @ self.Sm.T).transpose(2, 0, 1)      # (K, B, C)
            Rr, Ri = p[f"Rr{l}"], p[f"Ri{l}"]                                                   # (K, C, C)
            wr = (vr @ Rr - vi @ Ri).transpose(1, 2, 0); wi = (vr @ Ri + vi @ Rr).transpose(1, 2, 0)   # (B, C, K)
            z = wr @ self.Ci + wi @ self.Si + p[f"W{l}"].T[None] @ v + p[f"b{l}"][None, :, None]
            phi = gelu_phi(z); v_new = z * phi; st.append((vr, vi, z, v, phi)); v = v_new
        y = X + np.tensordot(v, p["q"], axes=([1], [0])) + p["q0"]
        return (y, st, v) if cache else y

    def loss_grads(self, X, Y):
        y, st, vL = self.forward(X, cache=True); p = self.p; B, N = X.shape
        loss = np.mean((y - Y) ** 2); dy = 2 * (y - Y) / (B * N); g = {}
        g["q"] = np.tensordot(vL, dy, axes=([0, 2], [0, 1])); g["q0"] = np.array([dy.sum()]); dv = dy[:, None, :] * p["q"][None, :, None]
        for l in reversed(range(self.L)):
            vr, vi, z, v, phi = st[l + 1]; dz = dv * dgelu(z, phi)
            g[f"W{l}"] = np.tensordot(v, dz, axes=([0, 2], [0, 2])); g[f"b{l}"] = dz.sum((0, 2))
            dwr, dwi = (dz @ self.Ci.T).transpose(2, 0, 1), (dz @ self.Si.T).transpose(2, 0, 1)   # (K, B, C)
            Rr, Ri = p[f"Rr{l}"], p[f"Ri{l}"]; vrT, viT = vr.transpose(0, 2, 1), vi.transpose(0, 2, 1)
            g[f"Rr{l}"] = vrT @ dwr + viT @ dwi; g[f"Ri{l}"] = -viT @ dwr + vrT @ dwi
            dvr = dwr @ Rr.transpose(0, 2, 1) + dwi @ Ri.transpose(0, 2, 1); dvi = -dwr @ Ri.transpose(0, 2, 1) + dwi @ Rr.transpose(0, 2, 1)
            dv = dvr.transpose(1, 2, 0) @ self.Cm + dvi.transpose(1, 2, 0) @ self.Sm + p[f"W{l}"][None] @ dz
        a0, _ = st[0]; g["P"] = np.tensordot(a0, dv, axes=([0, 2], [0, 2])); g["p0"] = dv.sum((0, 2))
        return loss, g

    def gradcheck(self, X, Y, eps=1e-6):
        _, g = self.loss_grads(X, Y); worst = 0.0
        for k, v in self.p.items():
            idx = np.unravel_index(np.arange(0, v.size, max(1, v.size // 7)), v.shape)
            for i in zip(*idx):
                old = v[i]; v[i] = old + eps; lp = np.mean((self.forward(X) - Y) ** 2); v[i] = old - eps
                lm = np.mean((self.forward(X) - Y) ** 2); v[i] = old
                worst = max(worst, abs((lp - lm) / (2 * eps) - g[k][i]) / (abs(g[k][i]) + 1e-8))
        return worst

    def fit(self, X, Y, steps=2000, lr=1e-2, wd=0.0, batch=200, seed=0, verbose=False):
        rng = np.random.default_rng(seed); m = {k: np.zeros_like(v) for k, v in self.p.items()}; s = {k: np.zeros_like(v) for k, v in self.p.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8; n = len(X); self.hist = []
        for t in range(1, steps + 1):
            idx = rng.choice(n, min(batch or n, n), replace=False)
            loss, g = self.loss_grads(X[idx], Y[idx]); self.hist.append(loss)
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / steps))
            for k in self.p:
                gk = g[k] + wd * self.p[k]; m[k] = b1 * m[k] + (1 - b1) * gk; s[k] = b2 * s[k] + (1 - b2) * gk ** 2
                self.p[k] -= lr_t * (m[k] / (1 - b1 ** t)) / (np.sqrt(s[k] / (1 - b2 ** t)) + eps)
            if verbose and t % 500 == 0: print(f"      step {t} loss {loss:.3e}")
        return self

    def predict(self, X): return self.forward(X)
    def n_params(self): return int(sum(v.size for v in self.p.values()))


# ====================================================================== E3 : bandes de frequence + bloc fast
def make_band_trials(C=3, p=8, T=64, n_per=20, band=(8, 12), sep=0.5, gain_sd=0.7, seed=0):
    """Essais (n, p, T) : bruit de base large bande (covariance spatiale commune) dont le gain spectral varie
    aleatoirement par essai et par bande (nuisance, exp(gain_sd N(0,1))), + composante de classe (motif spatial
    u_c, rang 2) portee UNIQUEMENT par les modes temporels de 'band'. Classes 0..C-1."""
    rng = np.random.default_rng(seed); k = np.fft.rfftfreq(T, 1.0 / T); mask = (k >= band[0]) & (k < band[1])
    Q, _ = np.linalg.qr(rng.standard_normal((p, p))); base = Q @ np.diag(np.exp(0.5 * rng.standard_normal(p))) @ Q.T
    Rb = np.linalg.cholesky(base); X, y = [], []; nb = 8; bidx = np.minimum(k.astype(int) // 4, nb - 1)
    for c in range(C):
        u = rng.standard_normal((p, 2))
        for _ in range(n_per):
            g = np.exp(gain_sd * rng.standard_normal(nb))[bidx]
            noise = np.fft.irfft(np.fft.rfft(Rb @ rng.standard_normal((p, T)), axis=-1) * g, n=T, axis=-1)
            a = np.fft.irfft(np.fft.rfft(rng.standard_normal((2, T)), axis=-1) * mask, n=T, axis=-1) * np.sqrt(T / (2 * mask.sum()))
            X.append(noise + sep * u @ a); y.append(c)
    X, y = np.stack(X), np.array(y); perm = rng.permutation(len(y)); return X[perm], y[perm]


def band_features(X, bands):
    """Pour chaque bande : matrice cross-spectrale reelle (somme des |X_k|^2 sur la bande), log-euclidienne,
    vectorisee isometriquement. Retourne une liste de (n, p(p+1)/2)."""
    Xh = np.fft.rfft(X - X.mean(2, keepdims=True), axis=-1); p = X.shape[1]; iu = np.triu_indices(p)
    wgt = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)); feats = []
    for lo, hi in bands:
        S = np.einsum("nck,ndk->ncd", Xh[:, :, lo:hi], Xh[:, :, lo:hi].conj()).real / X.shape[2]
        S += 1e-3 * np.trace(S, axis1=1, axis2=2)[:, None, None] / p * np.eye(p)
        lam, V = np.linalg.eigh(S); Lg = np.einsum("ncd,nd,ned->nce", V, np.log(lam), V)
        feats.append(Lg[:, iu[0], iu[1]] * wgt)
    return feats


def bands_classify(Xtr, ytr, Xte, yte, bands, mode="auto", seed=0):
    """mode : "broadband" (une seule covariance), "all" (rang C-1 dans chaque bande), "auto" (rang teste par bande,
    controle par bande), "fwer" (une bande n'est retenue que si elle passe le controle global min-p sur les bandes ;
    son rang est alors le rang teste)."""
    Ftr, Fte = band_features(Xtr, bands), band_features(Xte, bands); codes_tr, codes_te, ranks = [], [], []
    fits = []
    for b in range(len(bands)):
        if mode == "broadband": break
        fits.append(fit_block_fast(Ftr[b], ytr, shrink=True, rank="perm" if mode in ("auto", "fwer") else "C1", n_perm=200, seed=seed, min_rank=0))
    keep = fwer_select([f[3] for f in fits], 0.05)[0] if mode == "fwer" else np.ones(len(fits), bool)
    for b, (U, W, r, info) in enumerate(fits):
        r = int(info["r_tested"]) if keep[b] else 0                  # le test de permutation peut donner 0
        ranks.append(r)
        if r > 0: codes_tr.append(Ftr[b] @ W[:, :r]); codes_te.append(Fte[b] @ W[:, :r])
    if mode == "broadband":
        Fb_tr, Fb_te = band_features(Xtr, [(0, Xtr.shape[2] // 2 + 1)])[0], band_features(Xte, [(0, Xte.shape[2] // 2 + 1)])[0]
        U, W, r, _ = fit_block_fast(Fb_tr, ytr, shrink=True, rank="C1"); codes_tr, codes_te, ranks = [Fb_tr @ W], [Fb_te @ W], [int(r)]
    if not codes_tr: return 1.0 / len(np.unique(ytr)), ranks
    # lecture ridge sur la CONCATENATION des codes de plusieurs blocs (et, pour la comparaison a lecture egale, sur le
    # code du bloc large bande) : la regle des centroides n'a la signification de la LDA que sur le code d'UN bloc
    pr = readout_fit_predict(np.hstack(codes_tr), ytr, np.hstack(codes_te), ridge=1e-3)
    return float((pr == yte).mean()), ranks


# ====================================================================== main
if __name__ == "__main__":
    rng = np.random.default_rng(0); N = 64
    print("=== Verification du FNO numpy : gradients manuels vs differences finies ===")
    f = FNO1dNumpy(N=8, C=3, K=3, L=2, seed=1); Xg = rng.standard_normal((3, 8)); Yg = rng.standard_normal((3, 8))
    print(f"    ecart relatif max = {f.gradcheck(Xg, Yg):.1e}")

    # ---------------------------------------------------------------- E0 : correctif du test de permutation
    print("\n=== E0. Test de permutation en lot : S_W partage (variante de comparaison) vs scatter total S_T (defaut) ===")
    print("    taux de faux positifs (rang teste > 0) sur bruit pur, C=3, 200 tirages, nominal 5% :")
    for n_, d_ in [(30, 36), (60, 36), (120, 36)]:
        fp = {"within": 0, "total": 0}
        for t in range(200):
            Xn = rng.standard_normal((n_, d_)); yn = np.repeat(np.arange(3), n_ // 3)
            for pm in fp: fp[pm] += fit_block_fast(Xn, yn, rank="perm", n_perm=100, seed=t, perm_metric=pm)[3]["r_tested"] > 0
        print(f"      n={n_:4d} d={d_:3d} : S_W partage {fp['within']/2:5.1f}%   S_T {fp['total']/2:5.1f}%")

    # ---------------------------------------------------------------- E1 : advection-diffusion
    print("\n=== E1. Advection-diffusion (c=1, nu=0.02, dt=0.1, N=64) : 10 trajectoires x 20 pas en apprentissage ===")
    Xtr, Ytr, _ = make_pairs(advdiff_step, rng, 10, 20, N, 0.1); Xte, Yte, Tte = make_pairs(advdiff_step, rng, 20, 20, N, 0.1)
    k = np.fft.rfftfreq(N, 1.0 / N); true_gain = np.exp((-0.02 * k ** 2 - 1j * k) * 0.1)
    t0 = time.time(); sl = SpectralClosedForm(m=1).fit(Xtr, Ytr); t_sl = time.time() - t0
    kept = np.where(sl.keep)[0]
    print(f"    bloc spectral ferme (m=1) : {sl.n_modes()} modes retenus (indices {kept.min()}..{kept.max()}), {sl.n_params()} parametres, {t_sl*1000:.0f} ms")
    print(f"      gain estime vs propagateur exact sur les modes retenus : ecart max {np.abs(sl.w[sl.keep, 0] - true_gain[sl.keep]).max():.1e}")
    print(f"      erreur 1 pas {rel_l2(sl.predict(Xte), Yte):.2e} ; deroulement 20 pas : moyenne {rollout_err(sl.predict, Tte)[0]:.2e}, final {rollout_err(sl.predict, Tte)[1]:.2e}")
    t0 = time.time(); fno = FNO1dNumpy(N, C=16, K=12, L=2, seed=0).fit(Xtr, Ytr); t_f = time.time() - t0
    print(f"    FNO numpy (C=16, K=12, L=2, {fno.n_params()} parametres, Adam 2000 pas, {t_f:.1f} s) : erreur 1 pas {rel_l2(fno.predict(Xte), Yte):.2e} ; "
          f"deroulement moyenne {rollout_err(fno.predict, Tte)[0]:.2e}, final {rollout_err(fno.predict, Tte)[1]:.2e}")
    t0 = time.time(); mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=2000, random_state=0).fit(Xtr, Ytr); t_m = time.time() - t0
    print(f"    MLP (256, {sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)} parametres, {t_m:.1f} s) : erreur 1 pas {rel_l2(mlp.predict(Xte), Yte):.2e} ; "
          f"deroulement moyenne {rollout_err(mlp.predict, Tte)[0]:.2e}, final {rollout_err(mlp.predict, Tte)[1]:.2e}")

    # ---------------------------------------------------------------- E2 : Burgers
    for n_traj in (10, 50):
        print(f"\n=== E2. Burgers (nu=0.05, dt=0.1, N=64) : {n_traj} trajectoires x 20 pas en apprentissage ({n_traj*20} paires), test 20 trajectoires ===")
        rng = np.random.default_rng(1)
        Xtr, Ytr, _ = make_pairs(burgers_step, rng, n_traj, 20, N, 0.1); Xte, Yte, Tte = make_pairs(burgers_step, rng, 20, 20, N, 0.1)
        print(f"    {'methode':52s} {'params':>7s} {'temps':>8s} {'1 pas':>9s} {'deroul. moy':>12s} {'deroul. fin':>12s}")
        rows = []
        for m in (1, 2, 3):
            t0 = time.time(); sc = SpectralClosedForm(m=m).fit(Xtr, Ytr); t = time.time() - t0
            rows.append((f"bloc spectral ferme m={m} ({sc.n_modes()} modes auto)", sc.n_params(), t, rel_l2(sc.predict(Xte), Yte), *rollout_err(sc.predict, Tte)))
        for K in (4, 12, 24):
            t0 = time.time(); fno = FNO1dNumpy(N, C=16, K=K, L=2, seed=0).fit(Xtr, Ytr); t = time.time() - t0
            rows.append((f"FNO numpy C=16 L=2 K={K}", fno.n_params(), t, rel_l2(fno.predict(Xte), Yte), *rollout_err(fno.predict, Tte)))
        t0 = time.time(); mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=2000, random_state=0).fit(Xtr, Ytr); t = time.time() - t0
        rows.append(("MLP (256)", sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_), t, rel_l2(mlp.predict(Xte), Yte), *rollout_err(mlp.predict, Tte)))
        for name, npar, t, e1, em, ef in rows:
            print(f"    {name:52s} {npar:7d} {t:7.1f}s {e1:9.2e} {em:12.2e} {ef:12.2e}")

    # ---------------------------------------------------------------- E3 : bandes
    print("\n=== E3. Classification : information de classe confinee aux modes temporels 8..11, gain spectral de nuisance par essai (p=8, T=64, C=3, sep=0.5), 5 graines ===")
    bands = [(4 * b, 4 * b + 4) for b in range(8)]
    for n_per in (10, 30):
        res = {"large bande (rang C-1)": [], "8 bandes, rang C-1 chacune": [], "8 bandes, rang par permutation (auto)": [], "8 bandes, controle global min-p": [], "MLP(64) sur les 8 bandes": []}; sel = []; sel_fw = []
        for sd in range(5):
            X, y = make_band_trials(n_per=2 * n_per, seed=sd); n = len(y); Xtr, ytr, Xte, yte = X[:n // 2], y[:n // 2], X[n // 2:], y[n // 2:]
            res["large bande (rang C-1)"].append(bands_classify(Xtr, ytr, Xte, yte, bands, "broadband")[0])
            res["8 bandes, rang C-1 chacune"].append(bands_classify(Xtr, ytr, Xte, yte, bands, "all")[0])
            a, ranks = bands_classify(Xtr, ytr, Xte, yte, bands, "auto", seed=sd); res["8 bandes, rang par permutation (auto)"].append(a); sel.append(ranks)
            a, ranks = bands_classify(Xtr, ytr, Xte, yte, bands, "fwer", seed=sd); res["8 bandes, controle global min-p"].append(a); sel_fw.append(ranks)
            Ftr, Fte = np.hstack(band_features(Xtr, bands)), np.hstack(band_features(Xte, bands))
            res["MLP(64) sur les 8 bandes"].append(MLPClassifier((64,), max_iter=800, random_state=sd).fit(Ftr, ytr).score(Fte, yte))
        print(f"  {n_per} essais/classe :")
        for kk, v in res.items(): print(f"      {kk:42s} : {100*np.mean(v):5.1f} +- {100*np.std(v):4.1f} %")
        print(f"      rangs par bande, controle par bande  : {sel}   (bande 2 = modes 8..11 porte l'information)")
        print(f"      rangs par bande, controle global min-p : {sel_fw}")
