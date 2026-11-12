"""
Selection du rang du bloc discriminant analytique : ce qui est garanti, ce qui est mesure.

Trois rangs sont distingues : r_struct = min(C-1, d) (borne exacte), r_tested (regle de test, peut valoir 0),
r_used = max(1, r_tested) (rang effectivement utilise par le classifieur). Regles comparees :
  perm     : chaque valeur propre contre le quantile 95 % du MAXIMUM sous permutation (metrique S_T) ;
  seq      : test de permutation sequentiel avec deflation (metrique S_T) ;
  bartlett : test sequentiel de Bartlett sur le Lambda de Wilks (asymptotique) ;
  C-1      : borne structurelle, aucun test.

Experiences (donnees gaussiennes a covariance commune, C = 4, d = 20, kappa = 10) :
  0. calibration sous l'hypothese nulle globale (2000 tirages) : metrique S_T, metrique S_W partagee, Bartlett,
     en bruit isotrope puis anisotrope ;
  A. alternatives de rang vrai q = 0, 1, 2, 3 avec des forces egales ou graduees par direction : loi du rang
     teste (sous-estimation, exactitude, sur-estimation), separement du risque predictif (precision avec le rang
     utilise, avec C-1 et avec le rang vrai) ;
  B. famille de tests (8 blocs independants, par exemple des bandes de frequence) : probabilite de retenir au
     moins un bloc sous l'hypothese nulle globale avec un test par bloc, et avec le controle global min-p ;
     puissance quand un seul bloc porte l'information.
numpy / scipy seulement.
"""
import numpy as np
from numpy.linalg import qr, cholesky
from ropn_auto_fast import fit_block_fast, fwer_select
from ropn_auto_nobp_diag import readout_centroid_fit_predict
from ropn_auto_stats import fmt_paired

RULES = ("perm", "seq", "bartlett")


def make_rank_data(q, strengths, n_per, C=4, d=20, kappa=10.0, seed=0, n_test=2000):
    """Moyennes de classe de rang q exactement : mu_c = sqrt(C) * sum_j H[c, j] s_j v_j, H base orthonormee des
    contrastes entre classes, v_j directions orthonormees ; s_j = ecart-type inter-classes le long de v_j, en
    unites de l'ecart-type moyen du bruit. Bruit gaussien de covariance commune (conditionnement kappa)."""
    rng = np.random.default_rng(seed)
    eig = np.logspace(0, np.log10(kappa), d); eig = eig / eig.mean()
    Qm, _ = qr(rng.standard_normal((d, d))); L = cholesky(Qm @ np.diag(eig) @ Qm.T)
    H = qr(np.eye(C) - 1.0 / C)[0][:, :C - 1]; V, _ = qr(rng.standard_normal((d, C - 1)))
    mus = np.sqrt(C) * (H[:, :q] * np.asarray(strengths[:q], float)) @ V[:, :q].T if q > 0 else np.zeros((C, d))
    def draw(n):
        y = np.repeat(np.arange(C), n); return rng.standard_normal((C * n, d)) @ L.T + mus[y], y
    Xtr, ytr = draw(n_per); Xte, yte = draw(n_test // C)
    return Xtr, ytr, Xte, yte


def acc_rank(W, r, Xtr, ytr, Xte, yte):
    # lecture par centroides dans l'espace du code = regle LDA restreinte aux r premieres directions
    r = max(1, int(r)); return float((readout_centroid_fit_predict(Xtr @ W[:, :r], ytr, Xte @ W[:, :r]) == yte).mean())


if __name__ == "__main__":
    C, d, REPS = 4, 20, 200
    print("=== 0. Calibration sous l'hypothese nulle globale (bruit pur), 2000 tirages, 199 permutations, niveau nominal 5 % (erreur-type 0.5 point) ===")
    print(f"    {'n':>4s} {'d':>4s} {'C':>3s} | {'permutation, S_T':>17s} | {'permutation, S_W partage':>25s} | {'Bartlett':>9s}")
    for n_per, d0, C0 in [(10, 36, 3), (20, 36, 3), (40, 36, 3), (10, 20, 4), (30, 20, 4), (100, 20, 4)]:
        fp = {"total": 0, "within": 0, "bartlett": 0}; R0 = 2000
        for rep in range(R0):
            rng = np.random.default_rng(900000 + rep); y0 = np.repeat(np.arange(C0), n_per); X0 = rng.standard_normal((C0 * n_per, d0))
            fp["total"] += fit_block_fast(X0, y0, rank="perm", n_perm=199, seed=rep, min_rank=0)[3]["r_tested"] > 0
            fp["within"] += fit_block_fast(X0, y0, rank="perm", n_perm=199, seed=rep, min_rank=0, perm_metric="within")[3]["r_tested"] > 0
            fp["bartlett"] += fit_block_fast(X0, y0, rank="bartlett", min_rank=0)[3]["r_tested"] > 0
        print(f"    {C0*n_per:4d} {d0:4d} {C0:3d} | {100*fp['total']/R0:16.1f}% | {100*fp['within']/R0:24.1f}% | {100*fp['bartlett']/R0:8.1f}%")
    # meme verification avec le bruit ANISOTROPE (kappa = 10) et les tailles de la partie A : l'echangeabilite ne
    # depend pas de la covariance du bruit, le niveau doit donc rester <= 5 % ; 2000 tirages au lieu de 200
    print("    bruit anisotrope de la partie A (C = 4, d = 20, kappa = 10), 2000 tirages, 200 permutations :")
    for n_per in (10, 30, 100):
        fp = {"perm": 0, "seq": 0}; R0 = 2000
        for rep in range(R0):
            X0, y0, _, _ = make_rank_data(0, (1.2, 1.2, 1.2), n_per, seed=50000 + rep, n_test=8)
            for rule in fp:
                fp[rule] += fit_block_fast(X0, y0, rank=rule, n_perm=200, seed=rep, min_rank=0)[3]["r_tested"] > 0
        print(f"    {4*n_per:4d} {20:4d} {4:3d} | regle perm {100*fp['perm']/R0:4.1f}% | regle seq {100*fp['seq']/R0:4.1f}%")
    print()
    print(f"=== A. Recuperation du rang sous des alternatives de rang q (C={C}, d={d}, kappa=10, {REPS} tirages independants, 200 permutations, niveau 5 %) ===")
    print("    colonnes : P(rang teste < q) / P(= q) / P(> q), rang teste moyen ; precision (%) sur 2000 points de test, lecture par centroides, avec le rang UTILISE max(1, rang teste)")
    for prof_name, strengths in [("forces egales (1.2, 1.2, 1.2)", (1.2, 1.2, 1.2)), ("forces graduees (1.6, 0.8, 0.4)", (1.6, 0.8, 0.4))]:
        for n_per in (10, 30, 100):
            print(f"\n  {prof_name}, n = {n_per}/classe (n_train = {C*n_per}) :")
            print(f"    {'q':>2s} | " + " | ".join(f"{r_:^34s}" for r_ in RULES) + " | precision : " + " ".join(f"{r_:>8s}" for r_ in RULES) + f" {'C-1':>7s} {'rang q':>7s}")
            for q in range(C):
                rt = {r_: [] for r_ in RULES}; ac = {k: [] for k in RULES + ("C1", "oracle")}
                for rep in range(REPS):
                    Xtr, ytr, Xte, yte = make_rank_data(q, strengths, n_per, C, d, seed=1000 * q + rep)
                    U, W, _, _ = fit_block_fast(Xtr, ytr, shrink=True, rank="C1")
                    for r_ in RULES:
                        info = fit_block_fast(Xtr, ytr, shrink=True, rank=r_, n_perm=200, seed=rep, min_rank=0)[3]
                        rt[r_].append(info["r_tested"])
                        if q > 0: ac[r_].append(acc_rank(W, info["r_tested"], Xtr, ytr, Xte, yte))
                    if q > 0:
                        ac["C1"].append(acc_rank(W, C - 1, Xtr, ytr, Xte, yte)); ac["oracle"].append(acc_rank(W, q, Xtr, ytr, Xte, yte))
                cells = []
                for r_ in RULES:
                    v = np.array(rt[r_]); cells.append(f"{100*np.mean(v < q):5.1f} /{100*np.mean(v == q):5.1f} /{100*np.mean(v > q):5.1f}  moy {v.mean():4.2f}")
                accs = " ".join(f"{100*np.mean(ac[k]):8.1f}" if q > 0 else f"{'--':>8s}" for k in RULES) + (f" {100*np.mean(ac['C1']):7.1f} {100*np.mean(ac['oracle']):7.1f}" if q > 0 else f" {'--':>7s} {'--':>7s}")
                print(f"    {q:2d} | " + " | ".join(cells) + " |             " + accs)
                if q > 0:    # risque predictif, differences appariees sur les memes tirages
                    print(f"         appariees [IC 95 %] : rang teste (seq) - (C-1)  {fmt_paired(ac['seq'], ac['C1'])} | (C-1) - rang vrai q  {fmt_paired(ac['C1'], ac['oracle'])}")
    print("\n  (q = 0 : P(> q) est le taux de faux positifs de l'hypothese nulle globale, niveau nominal 5 % ; erreur-type binomiale <= 3.5 points.)")

    print("\n=== B. Famille de 8 tests (blocs independants de dimension 36, C = 3, n = 30) : controle par bloc contre controle global min-p, 200 tirages ===")
    for label, sep in [("hypothese nulle globale (aucun bloc informatif)", 0.0), ("un bloc informatif sur 8 (bloc 2, force 1.5)", 1.5)]:
        any_pb, any_fw, hit_pb, hit_fw, fp_pb, fp_fw = [], [], [], [], [], []
        for rep in range(200):
            rng = np.random.default_rng(5000 + rep); y = np.repeat(np.arange(3), 10); infos = []
            for b in range(8):
                Xb = rng.standard_normal((30, 36))
                if b == 2 and sep > 0:
                    Hc = qr(np.eye(3) - 1.0 / 3)[0][:, :2]; Vb, _ = qr(rng.standard_normal((36, 2))); Xb = Xb + np.sqrt(3) * sep * (Hc @ Vb.T)[y]
                infos.append(fit_block_fast(Xb, y, shrink=True, rank="perm", n_perm=200, seed=rep, min_rank=0)[3])
            pb = np.array([i["r_tested"] > 0 for i in infos]); fw, _ = fwer_select(infos, 0.05)
            null_idx = np.array([b != 2 or sep == 0 for b in range(8)])
            any_pb.append(pb[null_idx].any()); any_fw.append(fw[null_idx].any()); fp_pb.append(pb[null_idx].mean()); fp_fw.append(fw[null_idx].mean())
            hit_pb.append(pb[2]); hit_fw.append(fw[2])
        line = f"  {label} :\n      test par bloc : P(au moins un bloc nul retenu) = {100*np.mean(any_pb):5.1f} %, taux par bloc nul = {100*np.mean(fp_pb):4.1f} %"
        line += f"\n      controle min-p : P(au moins un bloc nul retenu) = {100*np.mean(any_fw):5.1f} %, taux par bloc nul = {100*np.mean(fp_fw):4.1f} %"
        if sep > 0: line += f"\n      puissance (bloc informatif retenu) : test par bloc {100*np.mean(hit_pb):5.1f} %, controle min-p {100*np.mean(hit_fw):5.1f} %"
        print(line)
    print("  (attendu sous l'hypothese nulle globale pour 8 tests independants au niveau 5 % : 1 - 0.95^8 = 33.7 %.)")
