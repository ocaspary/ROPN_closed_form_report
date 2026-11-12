"""Balayage kappa x graines : accuracy moyenne +/- std, comparaison
bloc unique auto-ROPN (sans backprop) vs MLP (backprop), et regime n<=d (shrinkage)."""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from ropn_auto_nobp import (make_multiclass, fit_oblique_block_closed_form,
                             make_closed_form_gate, recurse_gated)
from ropn_auto_nobp_diag import fit_block, readout_fit_predict, readout_centroid_fit_predict

def run_once(kappa, n_per, d=20, n_classes=4, T=4, seed=0, sep=0.8):
    X, y, *_ = make_multiclass(n_classes=n_classes, d=d, n_per=n_per, kappa=kappa, sep=sep, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    nc = NearestCentroid().fit(Xtr, ytr)
    acc_ortho = nc.score(Xte, yte)

    U, W, diag = fit_oblique_block_closed_form(Xtr, ytr, seed=seed)
    s, b = make_closed_form_gate(U, W, Xtr)
    Htr = recurse_gated(U, W, s, b, Xtr, T)
    Hte = recurse_gated(U, W, s, b, Xte, T)
    classes = np.unique(ytr)
    Ytr = np.eye(len(classes))[ytr]
    Wro, *_ = np.linalg.lstsq(np.c_[Htr, np.ones(len(Htr))], Ytr, rcond=None)
    pred = classes[(np.c_[Hte, np.ones(len(Hte))] @ Wro).argmax(1)]
    acc_auto = (pred == yte).mean()
    n_par_auto = U.size + Wro.size

    # pipeline corrige (section petits echantillons) : shrinkage systematique, rang structurel C-1,
    # lecture par centroides sur le code oblique c = W^T h_T (dim r) = regle LDA ; la lecture par
    # moindres carres sur le meme code est conservee pour comparaison
    Ub, Wb, _ = fit_block(Xtr, ytr, shrink=True, rank="C1", seed=seed)
    sb, bb = make_closed_form_gate(Ub, Wb, Xtr)
    Ctr = recurse_gated(Ub, Wb, sb, bb, Xtr, T) @ Wb; Cte = recurse_gated(Ub, Wb, sb, bb, Xte, T) @ Wb
    acc_best = (readout_centroid_fit_predict(Ctr, ytr, Cte) == yte).mean()
    acc_ls = (readout_fit_predict(Ctr, ytr, Cte) == yte).mean()
    n_par_best = Ub.size + (Wb.shape[1] + 1) * len(classes)       # centroides r C + a priori C = (r+1) C

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=seed)
    mlp.fit(Xtr, ytr)
    acc_mlp = mlp.score(Xte, yte)
    n_par_mlp = sum(w.size for w in mlp.coefs_) + sum(bb.size for bb in mlp.intercepts_)

    return acc_auto, n_par_auto, acc_mlp, n_par_mlp, diag['r_star'], diag['shrink'], acc_ortho, acc_best, n_par_best, acc_ls


if __name__ == "__main__":
    print(f"{'kappa':>7} {'n/classe':>9} {'acc ortho':>9} {'acc base':>9} {'acc corr':>9} {'acc MLP':>8} "
          f"{'#par base':>10} {'#par corr':>10} {'#par MLP':>9} {'r*':>3} {'shrink':>7}")
    SEEDS = 20
    from ropn_auto_stats import fmt_paired
    for kappa in [1.0, 10.0, 50.0, 200.0, 1000.0]:
        for n_per in [200, 15]:
            accs_a, accs_m, accs_o, accs_b, npa, npm, npb, rst, accs_l = [], [], [], [], [], [], [], [], []
            shrink_used = False
            for s in range(SEEDS):
                aa, na, am, nm, r, sh, ao, ab, nb, al = run_once(kappa, n_per, seed=s, T=4)
                accs_a.append(aa); accs_m.append(am); accs_o.append(ao); accs_b.append(ab); accs_l.append(al)
                npa.append(na); npm.append(nm); npb.append(nb); rst.append(r)
                shrink_used = shrink_used or sh
            print(f"{kappa:7.0f} {n_per:9d} "
                  f"{np.mean(accs_o)*100:8.1f}% {np.mean(accs_a)*100:8.1f}% {np.mean(accs_b)*100:8.1f}% {np.mean(accs_m)*100:7.1f}% "
                  f"{int(np.mean(npa)):10d} {int(np.mean(npb)):10d} {int(np.mean(npm)):9d} "
                  f"{np.mean(rst):3.1f} {'oui' if shrink_used else 'non':>7}"
                  f"   (std base={np.std(accs_a)*100:.1f}, std corr={np.std(accs_b)*100:.1f}, std MLP={np.std(accs_m)*100:.1f})")
            print(f"        differences appariees [IC 95 %] : corrige - MLP {fmt_paired(accs_b, accs_m)} | corrige - orthogonal {fmt_paired(accs_b, accs_o)}"
                  f" | corrige (centroides) - corrige (MC) {fmt_paired(accs_b, accs_l)} [MC : {np.mean(accs_l)*100:.1f}%]")
