"""
Comparaisons appariees entre methodes evaluees sur les MEMES realisations (memes tirages ou memes decoupages).

paired_summary(a, b) : difference moyenne a - b, intervalle de confiance de Student a 95 % sur les differences
appariees, et verdict par rapport a une marge d'equivalence pratique fixee a l'avance (1 point de precision par
defaut) :
    "equivalent"   : l'intervalle est contenu dans ]-marge, +marge[ ;
    "A > B", "A < B" : l'intervalle exclut 0 (et n'est pas contenu dans la marge) ;
    "indetermine"  : l'intervalle contient 0 et deborde de la marge (les donnees ne permettent pas de conclure).
Une difference inferieure a l'ecart-type n'est donc jamais declaree nulle : elle est "indeterminee" tant que
l'intervalle n'est pas dans la marge. Les decoupages repetes d'un meme jeu de donnees ne sont pas des realisations
independantes du probleme : l'intervalle mesure alors la variabilite du decoupage, pas celle de la population.
"""
import numpy as np
from scipy.stats import t as student_t

MARGIN = 0.01          # marge d'equivalence pratique : 1 point de precision, fixee avant les experiences


def paired_summary(a, b, margin=MARGIN, level=0.95):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if not np.isfinite(se) or se == 0.0:
        lo = hi = m
    else:
        q = float(student_t.ppf(0.5 + level / 2, n - 1)); lo, hi = m - q * se, m + q * se
    if -margin < lo and hi < margin: verdict = "equivalent"
    elif lo > 0: verdict = "A > B"
    elif hi < 0: verdict = "A < B"
    else: verdict = "indetermine"
    return dict(mean=m, lo=lo, hi=hi, n=n, verdict=verdict)


def fmt_paired(a, b, margin=MARGIN, scale=100.0):
    r = paired_summary(a, b, margin)
    return f"{scale*r['mean']:+5.1f} [{scale*r['lo']:+5.1f}, {scale*r['hi']:+5.1f}] {r['verdict']}"
