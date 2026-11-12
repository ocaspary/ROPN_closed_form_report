"""
Factorisation a cout reduit de la GEVD des blocs fermes (base et corrige).

Idee centrale (exacte, sans perte) : S_B = D D^T est de rang <= C-1, avec
    D = [sqrt(n_c) (mu_c - mu)]_c / sqrt(C-1)   (d x C).
Les paires propres NON NULLES de S_B v = lambda S_W v se lisent sur le petit probleme C x C
    M = D^T S_W^{-1} D = Q Lambda Q^T,
    W = S_W^{-1} D Q_r Lambda_r^{-1/2}    (mesure),
    U = D Q_r Lambda_r^{-1/2}             (representation),
et l'on a W^T U = I_r AUTOMATIQUEMENT : le reblanchiment devient inutile.
Cout : un seul systeme S_W Z = D (Cholesky d^3/3 + C solves) + eigh C x C, au lieu d'une GEVD
pleine d x d (~10 d^3). Aucune approximation.

Forme duale (n < d) : avec le shrinkage de Ledoit-Wolf, S = (1-a) Xc^T Xc / n + a mu I est
"identite + bas rang", donc S^{-1} D s'obtient par Woodbury avec un systeme n x n
(cout O(n^2 d + n^3)) sans jamais former de matrice d x d. Le coefficient de Ledoit-Wolf
lui-meme se calcule a partir de la matrice de Gram n x n et des normes ||x_i||^4 (identites
exactes ||X^T X||_F^2 = ||X X^T||_F^2 et sum(X2^T X2) = sum_i ||x_i||^4).

Test de permutation : D_p = Xbar^T Y_p N^{-1/2} (sommes par groupe permute), UNE factorisation
de la metrique partagee par toutes les permutations, et les n_perm problemes C x C resolus en
lot (eigvalsh sur une pile) : O(n_perm (n d + d^2 C)) au lieu de O(n_perm d^3), en BLAS-3.
Metrique du test : par defaut le scatter TOTAL S_T (independant des etiquettes, donc
echangeable sous permutation) ; perm_metric="within" donne le test a S_W partage
(calcule avec les vraies etiquettes), anti-conservateur des que n n'est pas grand devant d.
Le rang retourne est borne inferieurement par 1 ; info["r_tested"] donne le rang teste (0 possible).
Chaque valeur propre est comparee au quantile 95 % du MAXIMUM sous permutation : le niveau n'est
garanti que pour l'hypothese nulle globale (aucune structure), c'est une regle conservatrice au-dela.

Option supplementaire : test de Bartlett/Wilks (analytique, sans permutation) pour le rang.
"""
import numpy as np
from numpy.linalg import cholesky, eigh, norm
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2


# ---------------------------------------------------------------- Ledoit-Wolf sans matrice d x d
def ledoit_wolf_coef(Xc, gram=None):
    """Coefficient de Ledoit-Wolf (formule sklearn) calcule sans former Xc^T Xc.
    Xc : residus centres (n, d). gram : Xc Xc^T optionnel (n, n)."""
    n, d = Xc.shape
    sq = np.sum(Xc ** 2, axis=1)                      # ||x_i||^2
    emp_trace = np.sum(Xc ** 2, axis=0) / n
    mu = emp_trace.sum() / d
    if gram is not None:
        delta_ = np.sum(gram ** 2) / n ** 2           # ||Xc Xc^T||_F^2 = ||Xc^T Xc||_F^2
    elif n < d:
        delta_ = np.sum((Xc @ Xc.T) ** 2) / n ** 2    # dual : Gram n x n
    else:
        delta_ = np.sum((Xc.T @ Xc) ** 2) / n ** 2    # primal : d x d
    beta_ = np.sum(sq ** 2)                           # = sum(X2^T X2)
    beta = (beta_ / n - delta_) / (d * n)
    delta = (delta_ - 2.0 * mu * emp_trace.sum() + d * mu ** 2) / d
    beta = min(beta, delta)
    return (0.0 if beta == 0 else beta / delta), mu


# ---------------------------------------------------------------- solveur S^{-1} B (primal / dual)
class ShrunkSolver:
    """Applique S^{-1} avec S = (1-a) Xc^T Xc / n + a mu I  (a = 0 : S_W brut / m).
    Primal (n >= d) : Cholesky de S (d x d).  Dual (n < d) : Woodbury sur n x n."""

    def __init__(self, Xc, alpha, mu, denom=None, ridge_rel=1e-10):
        n, d = Xc.shape
        self.Xc, self.alpha, self.mu = Xc, alpha, mu
        denom = n if denom is None else denom
        self.scale = (1.0 - alpha) / denom            # S = scale * Xc^T Xc + alpha mu I
        self.diag = alpha * mu
        self.dual = n < d
        if self.dual:
            if self.diag <= 0:                        # S_W brut singulier en n < d : petit ridge
                self.diag = ridge_rel * self.scale * np.sum(Xc ** 2) / d
            K = self.scale * (Xc @ Xc.T) + self.diag * np.eye(n)
            self.cf = cho_factor(K, lower=True)
        else:
            S = self.scale * (Xc.T @ Xc) + self.diag * np.eye(d)
            if self.diag <= 0:
                S += ridge_rel * np.trace(S) / d * np.eye(d)
            self.cf = cho_factor(S, lower=True)

    def solve(self, B):
        if not self.dual:
            return cho_solve(self.cf, B)
        # Woodbury : (c I + s X^T X)^{-1} B = B/c - (s/c) X^T (c I + s X X^T)^{-1} X B
        c, s, X = self.diag, self.scale, self.Xc
        return B / c - (s / c) * (X.T @ cho_solve(self.cf, X @ B))


# ---------------------------------------------------------------- bloc ferme rapide
def fit_block_fast(X, y, shrink=True, rank="C1", n_perm=200, seed=0, alpha_bartlett=0.05, perm_metric="total",
                   min_rank=1, alpha=0.05, threshold="exact"):
    """Meme bloc que fit_block (ropn_auto_nobp_diag) / fit_oblique_block_closed_form,
    via la factorisation bas rang. Retourne U, W, r, info.

    Trois rangs sont distingues (tous dans info) :
      r_struct : borne structurelle min(C-1, d) ;
      r_tested : rang donne par la regle de test (peut valoir 0) ;
      r_used   : rang effectivement utilise = max(min_rank, r_tested) (= r retourne).
    Regles (parametre rank) :
      "C1"       : r_tested = r_struct (aucun test) ;
      "perm"     : chaque valeur propre est comparee au seuil exact de niveau alpha du MAXIMUM sous permutation
                   (niveau garanti pour l'hypothese nulle globale seulement) ;
      "seq"      : test de permutation sequentiel avec deflation : a l'etape k, la k+1-ieme valeur propre est
                   comparee a la plus grande valeur propre obtenue en permutant les etiquettes des donnees
                   dont les k premieres directions ont ete retirees ; arret au premier non-rejet ;
      "bartlett" : test sequentiel de Bartlett sur le Lambda de Wilks (asymptotique)."""
    rng = np.random.default_rng(seed)
    classes, yi = np.unique(y, return_inverse=True)
    C = len(classes); n, d = X.shape
    ns = np.bincount(yi, minlength=C).astype(float)
    mu = X.mean(0)
    Y = np.zeros((n, C)); Y[np.arange(n), yi] = 1.0
    mus = (Y.T @ X) / ns[:, None]
    Xbar = X - mu
    D = ((mus - mu).T * np.sqrt(ns)) / np.sqrt(max(C - 1, 1))       # S_B = D D^T
    Xc = X - mus[yi]

    if shrink:
        a, m = ledoit_wolf_coef(Xc)
        solver = ShrunkSolver(Xc, a, m, denom=n)                     # convention sklearn LedoitWolf
    else:
        solver = ShrunkSolver(Xc, 0.0, 0.0, denom=max(n - C, 1))     # S_W brut / (n-C)

    Z = solver.solve(D)                                              # S^{-1} D   (d x C)
    M = D.T @ Z; M = 0.5 * (M + M.T)                                 # C x C
    lam, Q = eigh(M); order = np.argsort(lam)[::-1]; lam, Q = lam[order], Q[:, order]
    r_struct = min(C - 1, d)

    if rank == "C1":
        r = r_struct
    elif rank in ("perm", "seq"):
        # D_p = Xbar^T Y_p N^{-1/2} / sqrt(C-1) pour chaque permutation, en lot.
        # Metrique du test : perm_metric="total" (defaut) utilise S_T = scatter TOTAL (retreci), qui ne
        # depend pas des etiquettes : sans retrecissement, S_T = S_W + (C-1)/n S_B (normalisations 1/n, 1/n, 1/(C-1)),
        # donc memes vecteurs propres et lam_T = lam_W / (1 + (C-1) lam_W / n), fonction croissante de lam_W ;
        # avec deux retrecissements separes l'egalite n'est qu'approchee. Le test est echangeable avec une
        # seule factorisation. "within" (variante de comparaison) partage S_W calcule avec les VRAIES etiquettes : S_W exclut
        # la direction inter-classes vraie et non les directions permutees -> anti-conservateur si n <~ d.
        Yall = np.zeros((n, n_perm * C))
        for p in range(n_perm):
            yp = rng.permutation(yi)
            Yall[np.arange(n), p * C + yp] = 1.0
        Dall = (Xbar.T @ Yall) / np.sqrt(np.tile(ns, n_perm)) / np.sqrt(max(C - 1, 1))
        if perm_metric == "total":
            aT, mT = ledoit_wolf_coef(Xbar) if shrink else (0.0, 0.0)
            solver_T = ShrunkSolver(Xbar, aT, mT, denom=n if shrink else max(n - 1, 1))
            Z_test = solver_T.solve(D); Zall = solver_T.solve(Dall)
        else:
            Z_test = Z; Zall = solver.solve(Dall)                     # une seule factorisation (S_W)
        MT = D.T @ Z_test; MT = 0.5 * (MT + MT.T); lam_test, Q_test = eigh(MT)
        o_t = np.argsort(lam_test)[::-1]; lam_test, Q_test = lam_test[o_t], Q_test[:, o_t]
        Dp = Dall.T.reshape(n_perm, C, d); Zp = Zall.T.reshape(n_perm, C, d)
        Mp = Dp @ Zp.transpose(0, 2, 1); Mp = 0.5 * (Mp + Mp.transpose(0, 2, 1))
        null_max = np.linalg.eigvalsh(Mp)[:, -1]                      # lot de problemes C x C
        # seuil EXACT du test de permutation : avec m = floor(alpha (n_perm + 1)), on rejette si moins de m valeurs
        # permutees atteignent la valeur observee, c'est-a-dire si p = (1 + #{nul >= obs}) / (n_perm + 1) <= alpha.
        # (Le quantile empirique interpole de numpy serait decale d'un rang et donnerait un niveau reel d'environ 6 %.)
        # threshold="quantile" utilise le quantile empirique, convention de l'implementation de reference du bloc de base.
        m_rej = int(np.floor(alpha * (n_perm + 1)))
        if threshold == "quantile": null_q95 = float(np.quantile(null_max, 1.0 - alpha))
        else: null_q95 = float(np.sort(null_max)[::-1][m_rej - 1]) if m_rej >= 1 else np.inf
        p_value = float((1 + np.sum(null_max >= lam_test[0])) / (n_perm + 1))   # hypothese nulle globale
        if rank == "perm":
            r0 = int(min(r_struct, np.sum(lam_test[:r_struct] > null_q95)))  # rang teste (peut valoir 0)
        else:
            # test sequentiel avec deflation. Avec (U_k, W_k) la paire biorthogonale des k premieres directions
            # dans la metrique du test, (I-P_k)^T S^{-1} (I-P_k) = S^{-1} - W_k W_k^T : la statistique nulle de
            # l'etape k est la plus grande valeur propre de M_p - B_p^T B_p, B_p = W_k^T D_p (k x C), sans
            # nouvelle factorisation. La valeur observee est lam_test[k].
            Ap = Z_test.T @ Dall                                         # (C, n_perm C) = D^T S^{-1} D_p
            Ap = Ap.reshape(C, n_perm, C).transpose(1, 0, 2)             # (n_perm, C, C)
            r0, p_seq = 0, []
            for k in range(r_struct):
                if k == 0:
                    null_k = null_max
                else:
                    Bk = (Q_test[:, :k] / np.sqrt(np.clip(lam_test[:k], 1e-12, None))).T @ Ap   # (n_perm, k, C)
                    Mk = Mp - Bk.transpose(0, 2, 1) @ Bk
                    null_k = np.linalg.eigvalsh(0.5 * (Mk + Mk.transpose(0, 2, 1)))[:, -1]
                pk = float((1 + np.sum(null_k >= lam_test[k])) / (n_perm + 1)); p_seq.append(pk)
                if pk > alpha: break
                r0 = k + 1
        r = max(min_rank, r0)
    elif rank == "bartlett":
        # test sequentiel de Bartlett sur Wilks' Lambda (valeurs propres de W^{-1} B, echelle scatter)
        lam_sc = np.clip(lam[:r_struct], 0, None) * (C - 1) / max(n - C, 1)
        r = r_struct
        for k in range(r_struct):
            stat = (n - 1 - (d + C) / 2.0) * np.sum(np.log1p(lam_sc[k:]))
            df = (d - k) * (C - 1 - k)
            if chi2.sf(stat, df) > alpha_bartlett:
                r = k; break
        r0 = r; r = max(min_rank, r0)
    else:
        raise ValueError(rank)

    lam_r = np.clip(lam[:r], 1e-12, None)
    Qr = Q[:, :r]
    W = Z @ Qr / np.sqrt(lam_r)                                       # mesure : S^{-1} u
    U = D @ Qr / np.sqrt(lam_r)                                       # representation
    info = dict(r=r, eigenvalues=lam, dual=solver.dual, WtU_err=norm(W.T @ U - np.eye(r)) if r > 0 else 0.0,
                r_struct=r_struct, r_tested=r_struct if rank == "C1" else r0, r_used=r)
    if rank in ("perm", "seq"):
        info["null_q95"] = null_q95; info["eigenvalues_test"] = lam_test; info["null_max"] = null_max; info["p_value"] = p_value
        if rank == "seq": info["p_seq"] = p_seq
    return U, W, r, info


# ---------------------------------------------------------------- controle global d'une famille de tests
def fwer_select(infos, alpha=0.05):
    """Controle GLOBAL (family-wise) d'une famille de B tests de permutation menes avec les MEMES permutations
    (meme graine, memes etiquettes, par exemple un bloc par bande de frequence) : procedure min-p a un pas.
    Chaque statistique (observee ou permutee) est convertie en p-valeur dans sa propre famille, ce qui rend les
    blocs comparables ; la p-valeur ajustee d'un bloc est la proportion des tirages (observe inclus) dont la plus
    petite p-valeur sur les B blocs est <= la sienne. Sous l'hypothese nulle globale, la probabilite de retenir
    au moins un bloc est <= alpha. infos : liste des dictionnaires info de fit_block_fast(rank="perm" ou "seq").
    Retourne (masque des blocs retenus, p-valeurs ajustees)."""
    N = np.stack([i["null_max"] for i in infos]); obs = np.array([i["eigenvalues_test"][0] for i in infos])
    A = np.concatenate([obs[:, None], N], axis=1)                    # (B, n_perm + 1), colonne 0 = observe
    P = (A[:, None, :] >= A[:, :, None]).sum(2) / A.shape[1]         # P[b, j] = p-valeur du tirage j dans le bloc b
    minP = P.min(0)
    p_adj = np.array([(minP <= P[b, 0]).mean() for b in range(len(infos))])
    return p_adj <= alpha, p_adj


# ---------------------------------------------------------------- mise a jour en ligne (BCI)
def chol_rank1_update(L, x):
    """L L^T <- L L^T + x x^T en O(d^2) (L triangulaire inferieure, modifiee en place)."""
    x = x.copy(); d = len(x)
    for k in range(d):
        r = np.hypot(L[k, k], x[k]); c = r / L[k, k]; s = x[k] / L[k, k]
        L[k, k] = r
        if k + 1 < d:
            L[k+1:, k] = (L[k+1:, k] + s * x[k+1:]) / c
            x[k+1:] = c * x[k+1:] - s * L[k+1:, k]
    return L


class OnlineBlock:
    """Bloc ferme corrige mis a jour echantillon par echantillon, a shrinkage FIXE (a, mu) :
    S = (1-a)/n * sum_i (x_i - mu_{y_i})(...)^T + a mu I. Chaque nouvel exemple coute
    O(d^2) (mise a jour de Cholesky) + O(d^2 C) (C solves) + O(C^3) : pas de GEVD.
    La mise a jour rang-1 du scatter intra-classe est EXACTE : si la classe c a n_c exemples de
    moyenne m_c, ajouter x donne S_raw <- S_raw + n_c/(n_c+1) (x - m_c)(x - m_c)^T. La seule
    approximation par rapport a un reajustement complet est le gel de (a, mu) et du n de la
    normalisation entre deux reajustements."""

    def __init__(self, X, y, alpha, mu, n_classes=None):
        self.C = int(y.max()) + 1 if n_classes is None else n_classes; n, d = X.shape
        self.alpha, self.mu, self.n = alpha, mu, n
        self.sum_x = np.zeros((self.C, d)); self.cnt = np.zeros(self.C)
        self.S_raw = np.zeros((d, d))                  # sum_i (x_i - mu_{y_i})(x_i - mu_{y_i})^T
        for xi, yi in zip(X, y):
            self.sum_x[yi] += xi; self.cnt[yi] += 1
        mus = self.sum_x / self.cnt[:, None]
        Xc = X - mus[y]; self.S_raw = Xc.T @ Xc
        self._factor()

    def _S(self):
        d = self.S_raw.shape[0]
        return (1 - self.alpha) / self.n * self.S_raw + self.alpha * self.mu * np.eye(d)

    def _factor(self):
        self.L = cholesky(self._S())

    def add(self, x, c):
        """Ajoute un exemple (x, classe c) : rang-1 sur S_raw + rang-1 sur le Cholesky (echelle)."""
        mus_c = self.sum_x[c] / max(self.cnt[c], 1)
        r = np.sqrt(self.cnt[c] / (self.cnt[c] + 1.0)) * (x - mus_c)      # facteur n_c/(n_c+1) : mise a jour exacte
        self.S_raw += np.outer(r, r); self.sum_x[c] += x; self.cnt[c] += 1
        # S = k * S_raw + a mu I avec k = (1-a)/n ; on ne remet pas n a jour entre deux refit
        chol_rank1_update(self.L, np.sqrt((1 - self.alpha) / self.n) * r)

    def block(self):
        mu = self.sum_x.sum(0) / self.cnt.sum()
        mus = self.sum_x / self.cnt[:, None]
        D = ((mus - mu).T * np.sqrt(self.cnt)) / np.sqrt(max(self.C - 1, 1))
        Z = cho_solve((self.L, True), D)
        M = D.T @ Z; M = 0.5 * (M + M.T)
        lam, Q = eigh(M); o = np.argsort(lam)[::-1]; lam, Q = lam[o], Q[:, o]
        r = self.C - 1; lr = np.clip(lam[:r], 1e-12, None)
        return D @ Q[:, :r] / np.sqrt(lr), Z @ Q[:, :r] / np.sqrt(lr)
