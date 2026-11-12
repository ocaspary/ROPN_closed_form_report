"""
Benchmark central : le bloc discriminant analytique face a des references, avec un protocole symetrique.

Question testee : le bloc (metrique de Fisher retrecie, rang C-1, lecture sur le code) reste-t-il utile quand ses
hypotheses (classes gaussiennes de covariance commune) ne sont qu'approximatives, et a petit n ?

Deux lectures analytiques du code sont comparees : par centroides (c'est exactement la regle de la LDA de meme
covariance, voir readout_centroid_fit_predict) et par moindres carres sur les indicatrices de classe (la lecture
utilisee dans les autres scripts, sujette au masquage quand les classes sont nombreuses).

Jeux de donnees
  synthetiques, tirages INDEPENDANTS, test sur 2000 points (C = 4, d = 20, n = 15 ou 60 par classe) :
    gauss   : covariance commune (hypotheses satisfaites) ;
    hetero  : une covariance par classe (covariance commune violee) ;
    student : bruit de Student a 3 degres de liberte, dispersion commune (gaussianite violee) ;
    echelle : covariances proportionnelles, facteurs 0.4, 0.8, 1.4, 2.4 (violation moderee) ;
  reel : digits de scikit-learn (1797 images 8 x 8, 10 classes), sous-echantillons stratifies repetes de 10, 30 ou
    100 exemples par classe, test sur tout le reste (les repetitions mesurent la variabilite du decoupage, ce ne
    sont pas des realisations independantes du probleme).

Deux comparaisons distinctes
  SANS reglage : une recette par defaut par methode, fixee a l'avance ;
  APRES selection : 4 configurations par methode, choisies par validation croisee interne a 3 plis sur les MEMES
    plis pour toutes les methodes (budget de selection comparable).

Statistiques : differences appariees au bloc sur les memes realisations, intervalle de confiance a 95 %, marge
d'equivalence de 1 point (ropn_auto_stats.py). Temps : pretraitement, selection, ajustement final et inference
mesures separement. numpy / scipy / sklearn seulement.
"""
import time, warnings
import numpy as np
from numpy.linalg import qr, cholesky
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from ropn_auto_fast import fit_block_fast
from ropn_auto_crossgram import fit_crossgram_block
from ropn_auto_nobp_diag import readout_fit_predict, readout_centroid_fit_predict
from ropn_auto_stats import fmt_paired

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- generateurs synthetiques
def make_synth(kind, n_per, n_test=2000, C=4, d=20, kappa=50.0, sep=0.8, seed=0):
    rng = np.random.default_rng(seed); eig = np.logspace(0, np.log10(kappa), d); eig = eig / eig.mean()
    def cov():
        Q, _ = qr(rng.standard_normal((d, d))); return Q @ np.diag(eig) @ Q.T
    mus = sep * rng.standard_normal((C, d)); mus -= mus.mean(0, keepdims=True)
    base = cov(); Ls = [cholesky(cov() if kind == "hetero" else base) for _ in range(C)]
    scale = np.array([0.4, 0.8, 1.4, 2.4]) if kind == "echelle" else np.ones(C)
    def draw(n):
        X, y = [], []
        for c in range(C):
            Z = rng.standard_normal((n, d))
            if kind == "student": Z = Z / np.sqrt(rng.chisquare(3, (n, 1)) / 3.0)
            X.append(np.sqrt(scale[c]) * Z @ Ls[c].T + mus[c]); y.append(np.full(n, c))
        return np.concatenate(X), np.concatenate(y)
    Xtr, ytr = draw(n_per); Xte, yte = draw(n_test // C)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------- methodes : (ajuste, predit) pour une configuration
def block_fit_predict(cfg, Xtr, ytr, Xte, readout="centroid"):
    if cfg == "LW": W = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")[1]
    else: W = fit_crossgram_block(Xtr, ytr, cfg)[1]
    return (readout_centroid_fit_predict if readout == "centroid" else readout_fit_predict)(Xtr @ W, ytr, Xte @ W)

def block_ls_fit_predict(cfg, Xtr, ytr, Xte): return block_fit_predict(cfg, Xtr, ytr, Xte, "ls")

def lda_fit_predict(cfg, Xtr, ytr, Xte): return LDA(solver="lsqr", shrinkage=cfg).fit(Xtr, ytr).predict(Xte)
def logreg_fit_predict(cfg, Xtr, ytr, Xte): return LogisticRegression(C=cfg, max_iter=2000).fit(Xtr, ytr).predict(Xte)
def mlp_fit_predict(cfg, Xtr, ytr, Xte, seed=0):
    h, alpha, solver = cfg
    return MLPClassifier(hidden_layer_sizes=(h,), alpha=alpha, solver=solver, max_iter=500, random_state=seed).fit(Xtr, ytr).predict(Xte)

METHODS = {   # nom : (fonction, recette par defaut, grille de 4 configurations)
    "bloc analytique": (block_fit_predict, "LW", ["LW", 0.01, 0.1, 1.0]),           # lecture par centroides (regle LDA)
    "bloc, lecture MC": (block_ls_fit_predict, "LW", ["LW", 0.01, 0.1, 1.0]),        # lecture par moindres carres
    "LDA retrecie":    (lda_fit_predict, "auto", ["auto", 0.01, 0.1, 0.5]),
    "reg. logistique": (logreg_fit_predict, 1.0, [0.01, 0.1, 1.0, 10.0]),
    "MLP":             (mlp_fit_predict, (100, 1e-4, "adam"), [(64, 1e-4, "adam"), (64, 1e-1, "adam"), (64, 1e-4, "lbfgs"), (64, 1e-1, "lbfgs")]),
}


def evaluate(Xtr, ytr, Xte, yte, seed, select):
    """retourne {methode: (precision, t_selection, t_ajustement+prediction, configuration)} ; select=False : recette par defaut."""
    out = {}
    skf = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=seed).split(Xtr, ytr)) if select else None
    for name, (fn, default, grid) in METHODS.items():
        t_sel = 0.0; cfg = default
        if select:
            t0 = time.perf_counter(); best = -1.0
            for g in grid:
                a = np.mean([(fn(g, Xtr[tr], ytr[tr], Xtr[va]) == ytr[va]).mean() for tr, va in skf])
                if a > best + 1e-12: best, cfg = a, g
            t_sel = time.perf_counter() - t0
        t0 = time.perf_counter(); pred = fn(cfg, Xtr, ytr, Xte); t_fit = time.perf_counter() - t0
        out[name] = ((pred == yte).mean(), t_sel, t_fit, cfg)
    return out


def time_phases(name, cfg, Xtr, ytr, Xte, n_rep=3):
    """ajustement final et inference chronometres SEPAREMENT (minimum de n_rep executions), en secondes."""
    best_fit, best_pred = np.inf, np.inf
    for _ in range(n_rep):
        if name.startswith("bloc"):
            t0 = time.perf_counter()
            W = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")[1] if cfg == "LW" else fit_crossgram_block(Xtr, ytr, cfg)[1]
            Ctr = Xtr @ W; classes = np.unique(ytr)
            if name == "bloc analytique":
                M = np.stack([Ctr[ytr == c].mean(0) for c in classes]); lp = np.log(np.array([(ytr == c).mean() for c in classes]))
            else:
                V, *_ = np.linalg.lstsq(np.c_[Ctr, np.ones(len(Ctr))], np.eye(len(classes))[np.searchsorted(classes, ytr)], rcond=None)
            t1 = time.perf_counter()
            Cte = Xte @ W
            if name == "bloc analytique": pred = classes[(Cte @ M.T - 0.5 * (M ** 2).sum(1) + lp).argmax(1)]
            else: pred = classes[(np.c_[Cte, np.ones(len(Cte))] @ V).argmax(1)]
            t2 = time.perf_counter()
        else:
            if name == "LDA retrecie": model = LDA(solver="lsqr", shrinkage=cfg)
            elif name == "reg. logistique": model = LogisticRegression(C=cfg, max_iter=2000)
            else: model = MLPClassifier(hidden_layer_sizes=(cfg[0],), alpha=cfg[1], solver=cfg[2], max_iter=500, random_state=0)
            t0 = time.perf_counter(); model.fit(Xtr, ytr); t1 = time.perf_counter(); pred = model.predict(Xte); t2 = time.perf_counter()
        best_fit, best_pred = min(best_fit, t1 - t0), min(best_pred, t2 - t1)
    return best_fit, best_pred


def report(title, runs_def, runs_sel):
    print(f"\n  {title}")
    print(f"    {'methode':16s} | SANS reglage ({len(runs_def)} realisations) : precision, difference appariee au bloc [IC 95 %] verdict | APRES selection ({len(runs_sel)} realisations) : idem")
    for name in METHODS:
        cells = []
        for runs in (runs_def, runs_sel):
            a = np.array([r[name][0] for r in runs]); b = np.array([r["bloc analytique"][0] for r in runs])
            cells.append(f"{100*a.mean():5.1f} +/- {100*a.std():4.1f}   " + (fmt_paired(a, b) if name != "bloc analytique" else " " * 34))
        print(f"    {name:16s} | {cells[0]:62s} | {cells[1]}")


if __name__ == "__main__":
    R_DEF, R_SEL = 20, 10
    print(f"=== 1. Donnees synthetiques : tirages independants, test sur 2000 points ; {R_DEF} tirages sans reglage, {R_SEL} avec selection ===")
    for kind, label in [("gauss", "gauss (hypotheses satisfaites)"), ("hetero", "hetero (covariance commune violee)"),
                        ("student", "student t3 (gaussianite violee)"), ("echelle", "echelle (covariances proportionnelles)")]:
        for n_per in (15, 60):
            runs_def, runs_sel = [], []
            for rep in range(R_DEF):
                Xtr, ytr, Xte, yte = make_synth(kind, n_per, seed=rep)
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                runs_def.append(evaluate(Xtr, ytr, Xte, yte, rep, False))
                if rep < R_SEL: runs_sel.append(evaluate(Xtr, ytr, Xte, yte, rep, True))
            report(f"{label}, n = {n_per}/classe", runs_def, runs_sel)

    print(f"\n=== 2. Cas reel central : digits (d = 64 dont 61 variables non constantes, C = 10), sous-echantillons stratifies repetes, test sur le reste ===")
    D = load_digits(); X0, y0 = D.data, D.target
    timing = None
    for n_per in (10, 30, 100):
        runs_def, runs_sel = [], []
        for rep in range(R_DEF):
            rng = np.random.default_rng(rep); tr = np.concatenate([rng.choice(np.where(y0 == c)[0], n_per, replace=False) for c in range(10)])
            te = np.setdiff1d(np.arange(len(y0)), tr)
            t0 = time.perf_counter(); keep = X0[tr].std(0) > 1e-8; sc = StandardScaler().fit(X0[tr][:, keep])
            Xtr, Xte = sc.transform(X0[tr][:, keep]), sc.transform(X0[te][:, keep]); t_pre = time.perf_counter() - t0
            runs_def.append(evaluate(Xtr, y0[tr], Xte, y0[te], rep, False))
            if rep < R_SEL: runs_sel.append(evaluate(Xtr, y0[tr], Xte, y0[te], rep, True))
            if n_per == 100 and rep == 0: timing = (t_pre, len(te), runs_sel[0], Xtr, y0[tr], Xte)
        report(f"digits, n = {n_per}/classe (n_train = {10*n_per}, n_test = {len(te)})", runs_def, runs_sel)
    t_pre, n_te, r0, Xtr_t, ytr_t, Xte_t = timing
    print(f"\n  Temps par phase sur digits, n = 100/classe, une realisation (ms) : pretraitement commun (standardisation) {1000*t_pre:.1f}")
    print(f"    {'methode':16s} {'selection (4 config. x 3 plis)':>32s} {'ajustement final':>18s} {'inference de ' + str(n_te) + ' points':>26s}   (configuration selectionnee ; ajustement et inference : min de 3)")
    for name in METHODS:
        t_fit, t_pred = time_phases(name, r0[name][3], Xtr_t, ytr_t, Xte_t)
        print(f"    {name:16s} {1000*r0[name][1]:32.1f} {1000*t_fit:18.2f} {1000*t_pred:26.2f}   {r0[name][3]}")

    # variabilite d'entrainement contre variabilite du decoupage (MLP par defaut, digits, n = 30/classe)
    rng = np.random.default_rng(0); tr = np.concatenate([rng.choice(np.where(y0 == c)[0], 30, replace=False) for c in range(10)]); te = np.setdiff1d(np.arange(len(y0)), tr)
    keep = X0[tr].std(0) > 1e-8; sc = StandardScaler().fit(X0[tr][:, keep]); Xtr, Xte = sc.transform(X0[tr][:, keep]), sc.transform(X0[te][:, keep])
    a_seed = [(mlp_fit_predict((100, 1e-4, "adam"), Xtr, y0[tr], Xte, seed=s) == y0[te]).mean() for s in range(10)]
    print(f"\n  MLP par defaut, digits n = 30/classe : ecart-type sur 10 graines d'entrainement a decoupage fixe = {100*np.std(a_seed):.2f} point(s) (moyenne {100*np.mean(a_seed):.1f}) ; "
          f"le bloc analytique est deterministe a decoupage fixe.")

    print("\n=== 2 bis. Equivalence bloc + lecture par centroides = LDA de meme covariance (digits, 10 sous-echantillons) ===")
    from ropn_auto_fast import ledoit_wolf_coef
    for n_per in (10, 30, 100):
        ag_own, ag_sk = [], []
        for rep in range(10):
            rng = np.random.default_rng(rep); tr = np.concatenate([rng.choice(np.where(y0 == c)[0], n_per, replace=False) for c in range(10)]); te = np.setdiff1d(np.arange(len(y0)), tr)
            keep = X0[tr].std(0) > 1e-8; sc = StandardScaler().fit(X0[tr][:, keep]); Xtr, Xte, ytr = sc.transform(X0[tr][:, keep]), sc.transform(X0[te][:, keep]), y0[tr]
            W = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")[1]; pc = readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W)
            mus = np.stack([Xtr[ytr == k].mean(0) for k in range(10)]); Xc = Xtr - mus[ytr]; al, mu_ = ledoit_wolf_coef(Xc)
            Si = np.linalg.inv((1 - al) * Xc.T @ Xc / len(Xtr) + al * mu_ * np.eye(Xtr.shape[1]))     # meme S que le bloc
            sco = Xte @ Si @ mus.T - 0.5 * np.einsum("kd,de,ke->k", mus, Si, mus)[None, :] + np.log(np.bincount(ytr) / len(ytr))
            ag_own.append((pc == sco.argmax(1)).mean()); ag_sk.append((pc == LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).predict(Xte)).mean())
        print(f"  n = {n_per:3d}/classe : predictions identiques a la LDA calculee avec le meme S : {100*np.mean(ag_own):.1f} % ; a LDA(lsqr, shrinkage='auto') de scikit-learn : {100*np.mean(ag_sk):.1f} %")

    print("\n=== 3. Rang teste sur digits (regle sequentielle, 200 permutations) et precision (lecture par centroides) avec le rang teste ou avec C-1 = 9, 20 sous-echantillons ===")
    for n_per in (5, 10, 30):
        rt, a_t, a_c = [], [], []
        for rep in range(R_DEF):
            rng = np.random.default_rng(rep); tr = np.concatenate([rng.choice(np.where(y0 == c)[0], n_per, replace=False) for c in range(10)]); te = np.setdiff1d(np.arange(len(y0)), tr)
            keep = X0[tr].std(0) > 1e-8; sc = StandardScaler().fit(X0[tr][:, keep]); Xtr, Xte = sc.transform(X0[tr][:, keep]), sc.transform(X0[te][:, keep])
            U, W, r, info = fit_block_fast(Xtr, y0[tr], shrink=True, rank="seq", n_perm=200, seed=rep, min_rank=0); Wf = fit_block_fast(Xtr, y0[tr], shrink=True, rank="C1")[1]
            rt.append(info["r_tested"]); k = max(1, info["r_tested"])
            a_t.append((readout_centroid_fit_predict(Xtr @ Wf[:, :k], y0[tr], Xte @ Wf[:, :k]) == y0[te]).mean()); a_c.append((readout_centroid_fit_predict(Xtr @ Wf, y0[tr], Xte @ Wf) == y0[te]).mean())
        print(f"  n = {n_per:3d}/classe (n_train = {10*n_per}) : rang teste moyen {np.mean(rt):.2f} (min {min(rt)}, max {max(rt)}) | precision au rang utilise {100*np.mean(a_t):.1f} | au rang C-1 {100*np.mean(a_c):.1f} | difference appariee {fmt_paired(a_t, a_c)}")
