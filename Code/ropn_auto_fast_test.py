"""Verification d'exactitude et mesure des gains de la factorisation (ropn_auto_fast)."""
import time
import numpy as np
from numpy.linalg import norm
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.model_selection import train_test_split
from ropn_auto_nobp import make_multiclass, fit_oblique_block_closed_form, make_closed_form_gate, recurse_gated
from ropn_auto_nobp_diag import fit_block, readout_fit_predict, readout_centroid_fit_predict
from ropn_auto_fast import fit_block_fast, ledoit_wolf_coef, OnlineBlock


def proj_err(U1, W1, U2, W2):
    P1, P2 = U1 @ W1.T, U2 @ W2.T
    return norm(P1 - P2) / norm(P2)


print("=== 1. Exactitude (d=20, C=4, n=60 puis n=800) ===")
for n_per in [15, 200]:
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=n_per, kappa=50.0, sep=0.8, seed=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    classes = np.unique(ytr); mus = np.stack([Xtr[ytr == c].mean(0) for c in classes])
    Xc = Xtr - mus[np.searchsorted(classes, ytr)]
    a_fast, _ = ledoit_wolf_coef(Xc); a_sk = ledoit_wolf_shrinkage(Xc, assume_centered=True)
    print(f"  n={len(ytr):3d} | coef Ledoit-Wolf : fast={a_fast:.6f} sklearn={a_sk:.6f} | ecart {abs(a_fast-a_sk):.1e}")
    # corrige : reference fit_block(shrink=True, rank=C1) vs fast
    U1, W1, r1 = fit_block(Xtr, ytr, shrink=True, rank="C1")
    U2, W2, r2, info = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")
    print(f"        corrige : ||P_fast - P_ref||/||P_ref|| = {proj_err(U2, W2, U1, W1):.1e} | "
          f"W^T U - I : {info['WtU_err']:.1e} (sans reblanchiment) | r={r2}")
    # base : reference fit_oblique_block_closed_form (S_W brut si n>d, permutation) vs fast
    U1, W1, dg = fit_oblique_block_closed_form(Xtr, ytr, seed=0)
    U2, W2, r2, info = fit_block_fast(Xtr, ytr, shrink=False, rank="perm", seed=0, perm_metric="within", threshold="quantile")   # meme test que la reference (S_W partage) ; le defaut est "total"
    print(f"        base    : ||P_fast - P_ref||/||P_ref|| = {proj_err(U2, W2, U1, W1):.1e} | "
          f"r*: ref={dg['r_star']} fast={r2} | bartlett r={fit_block_fast(Xtr, ytr, shrink=True, rank='bartlett')[2]}")

print("\n=== 2. Forme duale (n < d) : coherence primal/dual et precision (d=200, n=15/classe) ===")
X, y, *_ = make_multiclass(n_classes=4, d=200, n_per=15, kappa=50.0, sep=1.5, seed=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
U1, W1, r1 = fit_block(Xtr, ytr, shrink=True, rank="C1")             # reference (matrice 200x200)
U2, W2, r2, info = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")  # dual (n_train=42 < 200)
print(f"  dual={info['dual']} | ||P_fast - P_ref||/||P_ref|| = {proj_err(U2, W2, U1, W1):.1e}")
for name, (U, W) in {"ref": (U1, W1), "fast": (U2, W2)}.items():
    s, b = make_closed_form_gate(U, W, Xtr)
    acc = (readout_centroid_fit_predict(recurse_gated(U, W, s, b, Xtr, 4) @ W, ytr,
                                        recurse_gated(U, W, s, b, Xte, 4) @ W) == yte).mean()
    print(f"  acc {name:4s} = {acc*100:.1f}%")

print("\n=== 3. Temps (fit du bloc seul ; corrige : min de 3 executions apres chauffe ; base : 1 execution apres chauffe ; base ref non mesuree a d=2080) ===")
print(f"  {'d':>5} {'n_train':>8} {'C':>3} | {'corr. ref (GEVD pleine)':>24} {'corr. fast':>11} {'gain':>6} | "
      f"{'base ref (+200 perm)':>21} {'base fast':>10} {'gain':>6}")
for d, n_per, C in [(20, 200, 4), (64, 120, 10), (200, 100, 4), (784, 100, 10), (2080, 30, 2), (2080, 150, 4)]:
    X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=50.0, sep=1.0, seed=0)
    def tm(fn, k=3):
        fn()                                    # chauffe (threads BLAS)
        ts = []
        for _ in range(k):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return np.min(ts)
    t_ref_c = tm(lambda: fit_block(X, y, shrink=True, rank="C1"))
    t_fast_c = tm(lambda: fit_block_fast(X, y, shrink=True, rank="C1"))
    if d <= 784:
        t_ref_b = tm(lambda: fit_oblique_block_closed_form(X, y, seed=0), k=1)
    else:
        t_ref_b = float("nan")      # ~200 GEVD 2080x2080 : plusieurs minutes, non mesure
    t_fast_b = tm(lambda: fit_block_fast(X, y, shrink=False, rank="perm", seed=0, perm_metric="within", threshold="quantile"), k=1)
    print(f"  {d:5d} {len(y):8d} {C:3d} | {t_ref_c:24.3f} {t_fast_c:11.4f} {t_ref_c/t_fast_c:6.0f}x | "
          f"{t_ref_b:21.2f} {t_fast_b:10.3f} {t_ref_b/t_fast_b:6.0f}x")


print("\n=== 4. Mise a jour en ligne (OnlineBlock) : exactitude et cout ===")
def P_of(U, W): return U @ W.T
X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=100, kappa=50.0, sep=0.8, seed=0)
perm = np.random.default_rng(0).permutation(len(y)); X, y = X[perm], y[perm]
n0, m_add = 200, 100
classes = np.unique(y[:n0]); mus = np.stack([X[:n0][y[:n0] == c].mean(0) for c in classes])
a0, mu0 = ledoit_wolf_coef(X[:n0] - mus[y[:n0]])
ob = OnlineBlock(X[:n0], y[:n0], a0, mu0, n_classes=4)
for xi, yi in zip(X[n0:n0 + m_add], y[n0:n0 + m_add]): ob.add(xi, int(yi))
Uo, Wo = ob.block()
# (a) meme (a, mu, n) geles, scatter recalcule en bloc : verifie l'exactitude des mises a jour rang-1
ob_ref = OnlineBlock(X[:n0 + m_add], y[:n0 + m_add], a0, mu0, n_classes=4); ob_ref.n = n0; ob_ref._factor(); Ur, Wr = ob_ref.block()
print(f"  d=20, {n0} exemples puis {m_add} ajouts : ecart au recalcul en bloc a (a, mu, n) geles = {norm(P_of(Uo, Wo) - P_of(Ur, Wr)) / norm(P_of(Ur, Wr)):.1e}")
# (b) ecart au reajustement complet (a, mu, n remis a jour)
Uf, Wf, _, _ = fit_block_fast(X[:n0 + m_add], y[:n0 + m_add], shrink=True, rank="C1")
print(f"  ecart au reajustement complet (Ledoit-Wolf et n remis a jour)                 = {norm(P_of(Uo, Wo) - P_of(Uf, Wf)) / norm(P_of(Uf, Wf)):.1e}")
# (c) cout par exemple a d = 2080 (n = 600, C = 4) contre un reajustement complet rapide
X, y, *_ = make_multiclass(n_classes=4, d=2080, n_per=150, kappa=50.0, sep=1.0, seed=0)
classes = np.unique(y); mus = np.stack([X[y == c].mean(0) for c in classes]); a0, mu0 = ledoit_wolf_coef(X - mus[y])
ob = OnlineBlock(X[:-5], y[:-5], a0, mu0, n_classes=4); ts = []
for xi, yi in zip(X[-5:], y[-5:]):
    t0 = time.perf_counter(); ob.add(xi, int(yi)); ob.block(); ts.append(time.perf_counter() - t0)
t0 = time.perf_counter(); fit_block_fast(X, y, shrink=True, rank="C1"); t_full = time.perf_counter() - t0
print(f"  d=2080, n=600 : ajout d'un exemple + bloc = {1000*np.median(ts):.0f} ms (boucle Python rang-1) ; reajustement complet rapide = {1000*t_full:.0f} ms")


print("\n=== 5. Contre une LDA specialisee (scikit-learn) : ajustement seul, min de 3 executions apres chauffe (s) ===")
print("    le bloc rapide et LDA(lsqr, shrinkage='auto') estiment le meme type de modele (LDA a covariance retrecie) ; LDA(svd) est sans retrecissement")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
print(f"  {'d':>5} {'n_train':>8} {'C':>3} | {'bloc rapide':>12} {'LDA lsqr+LW':>12} {'LDA eigen+LW':>13} {'LDA svd':>9} | {'lsqr / bloc':>12}")
for d, n_per, C in [(20, 200, 4), (64, 120, 10), (200, 100, 4), (784, 100, 10), (2080, 30, 2), (2080, 150, 4)]:
    X, y, *_ = make_multiclass(n_classes=C, d=d, n_per=n_per, kappa=50.0, sep=1.0, seed=0)
    def tm(fn, k=3):
        fn(); ts = []
        for _ in range(k):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return np.min(ts)
    t_b = tm(lambda: fit_block_fast(X, y, shrink=True, rank="C1"))
    t_l = tm(lambda: SkLDA(solver="lsqr", shrinkage="auto").fit(X, y))
    t_e = tm(lambda: SkLDA(solver="eigen", shrinkage="auto").fit(X, y)) if d <= 784 else float("nan")
    t_s = tm(lambda: SkLDA(solver="svd").fit(X, y))
    print(f"  {d:5d} {len(y):8d} {C:3d} | {t_b:12.4f} {t_l:12.4f} {t_e:13.4f} {t_s:9.4f} | {t_l/t_b:11.1f}x")
# accord des predictions entre le bloc (lecture par centroides, puis par moindres carres) et la LDA retrecie de
# scikit-learn (qui retrecit chaque covariance de classe separement : estimateur de covariance different)
agree, agree_ls = [], []
for sd in range(20):
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=215, kappa=50.0, sep=0.8, seed=sd)
    tr = np.concatenate([np.where(y == c)[0][:15] for c in range(4)]); te = np.setdiff1d(np.arange(len(y)), tr)
    U, W, r, _ = fit_block_fast(X[tr], y[tr], shrink=True, rank="C1")
    p_sk = SkLDA(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr]).predict(X[te])
    agree.append((readout_centroid_fit_predict(X[tr] @ W, y[tr], X[te] @ W) == p_sk).mean())
    agree_ls.append((readout_fit_predict(X[tr] @ W, y[tr], X[te] @ W) == p_sk).mean())
print(f"  accord des predictions bloc / LDA(lsqr, shrinkage='auto') sur 800 points de test, d=20, n=15/classe, 20 tirages : "
      f"lecture par centroides {100*np.mean(agree):.1f} % (min {100*np.min(agree):.1f} %) ; lecture par moindres carres {100*np.mean(agree_ls):.1f} % (min {100*np.min(agree_ls):.1f} %)")
