"""
Banc d'essai : orthogonal vs bloc ferme (base) vs bloc ferme (corrige) vs MLP (backprop)
sur des jeux synthetiques et reels.

Pour chaque (jeu, methode) on rapporte :
  - precision test (et balanced accuracy pour les jeux desequilibres)
  - convergence : nb d'iterations (MLP : n_iter_ et perte finale ; forme fermee : 1 passe,
    ou 1 + n_perm resolutions GEVP pour le test de permutation de la variante 'base')
  - temps de calcul mesure (fit + predict, wall-clock) et cout asymptotique
  - nombre de parametres stockes, et parmi eux ceux optimises par gradient

Methodes (definitions strictement identiques a celles des autres scripts) :
  orthogonal      : plus proche centroide (n'exploite pas S_W)                 [sklearn]
  bloc ferme base : S_W brut (shrink si n<=d), rang par permutation,
                    lecture LS sur h_T (dim d), T=4                            [ropn_auto_nobp]
  bloc ferme corr.: shrinkage Ledoit-Wolf, rang structurel C-1,
                    lecture par centroides sur le code c = W^T h_T (regle LDA)  [ropn_auto_nobp_diag]
  MLP backprop    : 1 couche cachee (largeur adaptee a d), Adam               [sklearn]

Jeux : synthetiques (gauss kappa=10 n=15/cl ; gauss kappa=200 n=200/cl ; heteroscedastique),
       reels sklearn (digits 8x8 ~ MNIST reduit ; breast cancer Wisconsin = medical ; wine),
       MNIST 784 via OpenML si le reseau le permet (sinon ignore avec un message).
"""
import time
import warnings
import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning

from ropn_auto_nobp import (make_multiclass, fit_oblique_block_closed_form,
                             make_closed_form_gate, recurse_gated)
from ropn_auto_nobp_diag import fit_block, readout_fit_predict, readout_centroid_fit_predict
from ropn_auto_finetune import make_hetero

warnings.filterwarnings("ignore", category=ConvergenceWarning)
T_REC = 4


# ---------------------------------------------------------------- jeux de donnees
def dataset_registry(seed=0, mnist_n_train=10000, mnist_n_test=2000):
    reg = {}
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=15, kappa=10.0, sep=0.8, seed=seed)
    reg["synth gauss k=10, n=15/cl"] = (X, y)
    X, y, *_ = make_multiclass(n_classes=4, d=20, n_per=200, kappa=200.0, sep=0.8, seed=seed)
    reg["synth gauss k=200, n=200/cl"] = (X, y)
    X, y, *_ = make_hetero(n_classes=4, d=20, n_per=60, kappa=50.0, sep=0.8, seed=seed)
    reg["synth hetero k=50, n=60/cl"] = (X, y)
    d = load_digits();        reg["digits 8x8 (10 cl.)"] = (d.data, d.target)
    b = load_breast_cancer(); reg["breast cancer (medical, 2 cl.)"] = (b.data, b.target)
    w = load_wine();          reg["wine (3 cl.)"] = (w.data, w.target)
    try:
        from sklearn.datasets import fetch_openml
        Xm, ym = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)
        rng = np.random.default_rng(seed); idx = rng.permutation(len(ym))
        Xm = Xm[idx[:mnist_n_train + mnist_n_test]].astype(np.float64)
        ym = ym[idx[:mnist_n_train + mnist_n_test]].astype(int)
        reg[f"MNIST 784 (sous-ech. {mnist_n_train}+{mnist_n_test})"] = (Xm, ym)
    except Exception as e:  # reseau indisponible : on l'indique, on ne fabrique rien
        print(f"[info] MNIST (OpenML) indisponible ici : {type(e).__name__}. "
              f"Lancer ce script avec acces reseau pour l'inclure.")
    return reg


def preprocess(X, y, seed, test_size=0.3):
    """Standardisation + suppression des colonnes constantes (pixels toujours nuls),
    ajustees sur le train uniquement."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    keep = Xtr.std(0) > 1e-8
    sc = StandardScaler().fit(Xtr[:, keep])
    return sc.transform(Xtr[:, keep]), sc.transform(Xte[:, keep]), ytr, yte


# ---------------------------------------------------------------- methodes
def run_orthogonal(Xtr, ytr, Xte):
    m = NearestCentroid().fit(Xtr, ytr)
    C, d = len(np.unique(ytr)), Xtr.shape[1]
    return m.predict(Xte), dict(params=C * d, params_grad=0, iters="1 passe (moyennes)",
                                cost="O(n d)")


def run_block_base(Xtr, ytr, Xte, seed):
    U, W, diag = fit_oblique_block_closed_form(Xtr, ytr, seed=seed)
    s, b = make_closed_form_gate(U, W, Xtr)
    Htr = recurse_gated(U, W, s, b, Xtr, T_REC); Hte = recurse_gated(U, W, s, b, Xte, T_REC)
    C = len(np.unique(ytr))
    pred = readout_fit_predict(Htr, ytr, Hte)
    return pred, dict(params=U.size + (Xtr.shape[1] + 1) * C, params_grad=0,
                      iters=f"1 GEVP + 200 GEVP (permutation), r*={diag['r_star']}",
                      cost="O(n d^2 + d^3) x (1+n_perm)")


def run_block_corr(Xtr, ytr, Xte, seed):
    U, W, r = fit_block(Xtr, ytr, shrink=True, rank="C1", seed=seed)
    s, b = make_closed_form_gate(U, W, Xtr)
    Ctr = recurse_gated(U, W, s, b, Xtr, T_REC) @ W; Cte = recurse_gated(U, W, s, b, Xte, T_REC) @ W
    C = len(np.unique(ytr))
    pred = readout_centroid_fit_predict(Ctr, ytr, Cte)              # lecture par centroides = regle LDA
    return pred, dict(params=U.size + (r + 1) * C, params_grad=0,
                      iters=f"1 GEVP (Ledoit-Wolf), r*={r}", cost="O(n d^2 + d^3)")


def run_mlp(Xtr, ytr, Xte, seed):
    d = Xtr.shape[1]; h = int(np.clip(2 * d, 32, 256))
    m = MLPClassifier(hidden_layer_sizes=(h,), max_iter=500, random_state=seed,
                      early_stopping=False, n_iter_no_change=20, tol=1e-4)
    m.fit(Xtr, ytr)
    n_par = sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)
    return m.predict(Xte), dict(params=n_par, params_grad=n_par,
                                iters=f"{m.n_iter_} epoques Adam (perte finale {m.loss_:.3f})"
                                      + (" [max_iter atteint]" if m.n_iter_ >= 500 else ""),
                                cost=f"O(epoques x n x d x h), h={h}")


METHODS = {
    "orthogonal (centroides)": lambda Xtr, ytr, Xte, s: run_orthogonal(Xtr, ytr, Xte),
    "bloc ferme base":         run_block_base,
    "bloc ferme corrige":      run_block_corr,
    "MLP (backprop)":          run_mlp,
}


# ---------------------------------------------------------------- boucle principale
def benchmark(seeds=(0, 1, 2, 3, 4), verbose=True, csv_path="ropn_auto_benchmark_results.csv"):
    results = {}
    reg = dataset_registry(seed=0)
    for dname, (X, y) in reg.items():
        results[dname] = {}
        for mname, fn in METHODS.items():
            accs, baccs, times, info = [], [], [], None
            for sd in seeds:
                Xtr, Xte, ytr, yte = preprocess(X, y, sd)
                t0 = time.perf_counter()
                try:
                    pred, info = fn(Xtr, ytr, Xte, sd)
                    ok = True
                except Exception as e:
                    ok = False; info = dict(params=-1, params_grad=-1, iters=f"ECHEC: {type(e).__name__}", cost="-")
                times.append(time.perf_counter() - t0)
                if ok:
                    accs.append((pred == yte).mean()); baccs.append(balanced_accuracy_score(yte, pred))
            results[dname][mname] = dict(acc=np.mean(accs) if accs else np.nan,
                                         acc_std=np.std(accs) if accs else np.nan,
                                         bacc=np.mean(baccs) if baccs else np.nan,
                                         time=np.mean(times), **info)
        if verbose:
            n, d, C = X.shape[0], X.shape[1], len(np.unique(y))
            print(f"\n=== {dname}  (n={n}, d={d}, C={C}, {len(seeds)} graines, split 70/30) ===")
            print(f"  {'methode':26s} {'acc':>13s} {'bal.acc':>8s} {'temps(s)':>9s} {'#param':>8s} {'#grad':>7s}  convergence | cout")
            for mname, r in results[dname].items():
                print(f"  {mname:26s} {r['acc']*100:6.1f} +/-{r['acc_std']*100:4.1f} {r['bacc']*100:7.1f} "
                      f"{r['time']:9.3f} {r['params']:8d} {r['params_grad']:7d}  {r['iters']} | {r['cost']}")
    # ---- export CSV + synthese ----
    with open(csv_path, "w") as f:
        f.write("dataset,methode,acc,acc_std,bal_acc,temps_s,n_params,n_params_grad,convergence,cout\n")
        for dname, per in results.items():
            for mname, r in per.items():
                f.write(f'"{dname}","{mname}",{r["acc"]:.4f},{r["acc_std"]:.4f},{r["bacc"]:.4f},'
                        f'{r["time"]:.4f},{r["params"]},{r["params_grad"]},"{r["iters"]}","{r["cost"]}"\n')
    if verbose:
        print(f"\n[resultats ecrits dans {csv_path}]")
        print("\n=== Synthese : meilleure methode par jeu (precision moyenne) ===")
        for dname, per in results.items():
            best = max(per.items(), key=lambda kv: kv[1]["acc"])
            corr, mlp = per["bloc ferme corrige"], per["MLP (backprop)"]
            print(f"  {dname:36s} -> {best[0]:24s} ({best[1]['acc']*100:.1f}%) | "
                  f"corrige vs MLP : {corr['acc']*100:.1f} vs {mlp['acc']*100:.1f}, "
                  f"params {corr['params']} vs {mlp['params']}, temps x{mlp['time']/max(corr['time'],1e-6):.0f}")
    return results


if __name__ == "__main__":
    benchmark()
