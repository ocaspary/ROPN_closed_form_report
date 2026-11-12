"""
Mini-experience "modele de langage" : prediction du caractere suivant a partir des k precedents,
pour situer les blocs fermes (LDA oblique = bloc fast, treillis) face a un MLP entraine par
retropropagation, sur une tache de type LM (C = taille de l'alphabet, entree = contexte one-hot).

Corpus : fichier fige ropn_corpus.txt (environ 1,5 x 10^5 caracteres : code source Python et commentaires
en francais, alphabet reduit a 35 symboles : lettres minuscules, espace, ponctuation courante). Le corpus
est fige pour que les resultats ne dependent pas de l'etat des scripts ; build_corpus() le reconstruit a
partir de fichiers sources. Metriques : precision top-1, perplexite (exp de la log-perte) quand le modele
fournit des probabilites.
"""
import re, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from ropn_auto_fast import fit_block_fast
from ropn_auto_lattice import fit_lattice_auto


def load_corpus(path="ropn_corpus.txt"):
    """corpus fige (deja normalise)."""
    return open(path, encoding="utf-8").read()


def build_corpus(paths):
    """normalisation utilisee pour construire ropn_corpus.txt a partir d'une liste de fichiers texte."""
    txt = ""
    for pth in paths:
        txt += open(pth, encoding="utf-8").read().lower() + " "
    txt = re.sub(r"\\[a-z]+", " ", txt)                    # commandes LaTeX -> espace
    txt = re.sub("[^a-z\u00e0\u00e2\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00fb\u00f9\u00fc\u00ff\u0153 ,.;:'()-]+", " ", txt)  # accents en echappements unicode (source ASCII)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def make_dataset(txt, k=4, n_max=30000, seed=0):
    chars = sorted(set(txt)); idx = {c: i for i, c in enumerate(chars)}; V = len(chars)
    ids = np.array([idx[c] for c in txt])
    n = len(ids) - k
    rng = np.random.default_rng(seed); pos = rng.choice(n, size=min(n, n_max), replace=False)
    X = np.zeros((len(pos), k * V), dtype=np.float32)
    for j in range(k):
        X[np.arange(len(pos)), j * V + ids[pos + j]] = 1.0
    y = ids[pos + k]
    keep = np.isin(y, np.where(np.bincount(y, minlength=V) >= 30)[0])   # classes rares retirees
    X, y = X[keep], y[keep]
    classes, y = np.unique(y, return_inverse=True)                       # etiquettes contigues
    return X, y, len(classes), V


def perplexity(P, y):
    p = np.clip(P[np.arange(len(y)), y], 1e-12, None)
    return float(np.exp(-np.mean(np.log(p))))


if __name__ == "__main__":
    txt = load_corpus(); print(f"corpus : {len(txt)} caracteres, alphabet de {len(set(txt))} symboles")
    for k in [2, 4]:
        X, y, V, Va = make_dataset(txt, k=k)   # V = classes de sortie, Va = taille de l'alphabet (one-hot)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        # aligner les classes (certaines lettres rares peuvent manquer du train)
        print(f"\n=== contexte k={k} : d = {X.shape[1]} (one-hot), C = {V} classes, n_train = {len(ytr)} ===")
        # 1. n-gramme (comptage) : baseline classique
        t0 = time.perf_counter()
        ctx_tr = Xtr.reshape(len(Xtr), k, Va).argmax(2); ctx_te = Xte.reshape(len(Xte), k, Va).argmax(2)
        table = {}
        for c, t in zip(map(tuple, ctx_tr), ytr): table.setdefault(c, np.zeros(V))[t] += 1
        uni = np.bincount(ytr, minlength=V) + 1.0
        P = np.stack([(table.get(tuple(c), np.zeros(V)) + 0.5 * uni / uni.sum()) for c in ctx_te]); P /= P.sum(1, keepdims=True)
        print(f"  {'n-gramme (comptage, lissage)':34s} acc = {(P.argmax(1) == yte).mean()*100:5.1f}  ppl = {perplexity(P, yte):6.2f}  | {time.perf_counter()-t0:5.1f} s")
        # 2. bloc fast = LDA oblique (lecture par centroides sur le code)
        t0 = time.perf_counter()
        U, W, r, info = fit_block_fast(Xtr.astype(float), ytr, shrink=True, rank="C1")
        from ropn_auto_nobp_diag import readout_centroid_fit_predict, readout_fit_predict
        pred = readout_centroid_fit_predict(Xtr @ W, ytr, Xte @ W)
        print(f"  {'bloc fast (LDA oblique, rang C-1)':34s} acc = {(pred == yte).mean()*100:5.1f}  ppl =    --   | {time.perf_counter()-t0:5.1f} s  (r={r}, dual={info['dual']})")
        pred_ls = readout_fit_predict(Xtr @ W, ytr, Xte @ W)            # meme code, lecture par moindres carres (comparaison)
        print(f"  {'bloc fast, lecture MC (comparaison)':34s} acc = {(pred_ls == yte).mean()*100:5.1f}  ppl =    --   |")
        # 3. LDA-shrinkage sklearn (probabilites -> perplexite)
        t0 = time.perf_counter()
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
        P = np.zeros((len(Xte), V)); P[:, lda.classes_] = lda.predict_proba(Xte)
        print(f"  {'LDA-shrinkage (proba)':34s} acc = {(P.argmax(1) == yte).mean()*100:5.1f}  ppl = {perplexity(P, yte):6.2f}  | {time.perf_counter()-t0:5.1f} s")
        # 4. regression logistique (lineaire, gradient convexe)
        t0 = time.perf_counter()
        lr = LogisticRegression(max_iter=300, C=1.0).fit(Xtr, ytr)
        P = np.zeros((len(Xte), V)); P[:, lr.classes_] = lr.predict_proba(Xte)
        print(f"  {'regression logistique (lineaire)':34s} acc = {(P.argmax(1) == yte).mean()*100:5.1f}  ppl = {perplexity(P, yte):6.2f}  | {time.perf_counter()-t0:5.1f} s")
        # 5. treillis AUTO (ferme, non lineaire fixe)
        if k == 2:
            t0 = time.perf_counter()
            m, cfg = fit_lattice_auto(Xtr.astype(float), ytr, n_groups=k, seed=0)
            print(f"  {'treillis AUTO (ferme)':34s} acc = {(m.predict(Xte.astype(float)) == yte).mean()*100:5.1f}  ppl =    --   | {time.perf_counter()-t0:5.1f} s  (cfg={cfg['k_quad'] if cfg['use_resid_quad'] else 0}, c^2={cfg['use_codes_sq']})")
        # 6. MLP (backprop)
        t0 = time.perf_counter()
        mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=60, random_state=0).fit(Xtr, ytr)   # 60 epoques au plus (budget fixe, non converge)
        P = np.zeros((len(Xte), V)); P[:, mlp.classes_] = mlp.predict_proba(Xte)
        print(f"  {'MLP 256 (backprop)':34s} acc = {(P.argmax(1) == yte).mean()*100:5.1f}  ppl = {perplexity(P, yte):6.2f}  | {time.perf_counter()-t0:5.1f} s")
