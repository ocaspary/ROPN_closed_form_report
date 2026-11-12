"""Exemples travailles (petites dimensions, nombres explicites) pour les theoremes de collapse, du determinant, de reduction de la profondeur gatee et de capacite du triceptron."""
import numpy as np
from numpy.linalg import norm

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def num_jac(F, h, eps=1e-6):
    d = len(h); J = np.zeros((d, d))
    for j in range(d):
        e = np.zeros(d); e[j] = eps; J[:, j] = (F(h + e) - F(h - e)) / (2 * eps)
    return J

print("=== Exemples : collapse et determinant du bloc gate (d=3, r=1, u=(1,0,0), w=(1,1,0)) ===")
u = np.array([1., 0, 0]); w = np.array([1., 1, 0]); P = np.outer(u, w); I3 = np.eye(3)
M = I3 + P / 2.0
print("T1   L=2, T=2 : (I+P/2)^2 - I = c_2 P avec c_2 = (3/2)^2 - 1 = 1.25 :", np.allclose(M @ M - I3, 1.25 * P),
      "| T=3 : c_3 = (3/2)^3 - 1 =", (1.5 ** 3 - 1), np.allclose(M @ M @ M - I3, (1.5 ** 3 - 1) * P))
print("     h=(1,0.5,2) : (I+P/2)^2 h =", np.round(M @ M @ np.array([1, .5, 2.]), 4), " = h + 1.25 * (1.5, 0, 0)")
s2 = np.array([0., 0, 1]); h2 = np.array([1.0, 0.5, 2.0])
for L in (1.0, 2.0):
    F = lambda h: h + (sigmoid(s2 @ h) / L) * (P @ h - h)
    a = sigmoid(s2 @ h2); t = s2 @ (h2 - P @ h2)
    det_th = (1 - a / L) ** 2 * (1 - a * (1 - a) / (L - a) * t)
    print(f"T2   L={L:.0f}, s=(0,0,1), h=(1,0.5,2) : a = sigma(2) = {a:.4f}, s^T(I-P)h = {t:.1f} -> det (formule) = {det_th:+.6f}, det (diff. finies) = {np.linalg.det(num_jac(F, h2)):+.6f}")
lo, hi = 0.5, 2.0                                    # L=1 : seuil a * h3 = 1
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if sigmoid(mid) * mid < 1: lo = mid
    else: hi = mid
hc = np.array([1.0, 0.5, 0.5 * (lo + hi)]); F1 = lambda h: h + sigmoid(s2 @ h) * (P @ h - h)
print(f"     L=1 : det = 0 quand sigma(h3) h3 = 1, soit h3* = {hc[2]:.4f} ; det (diff. finies) en h3* = {np.linalg.det(num_jac(F1, hc)):.1e} ; h3 = 2 > h3* -> det < 0 (orientation renversee)")
a = sigmoid(2.0)
print(f"     L=2 : det = 0 quand s^T(I-P)h = (L-a)/(a(1-a)) = {(2-a)/(a*(1-a)):.2f} ; condition suffisante du texte |s^T(I-P)h| < L/a = {2/a:.2f} ; ici 2 -> inversible.")

print("\n=== Exemples : reduction de la profondeur gatee ===")
# --- A.1 : d=3, r=1, u=(1,0,0), w=(1,1,0) -> P = u w^T, ker P = {h1+h2=0}
u = np.array([1., 0, 0]); w = np.array([1., 1, 0]); P = np.outer(u, w); Q = np.eye(3) - P
print("A.1  P = u w^T =\n", P, "\n     (I-P) =\n", Q, "\n     P^2 - P =", norm(P @ P - P))
L, T, b = 2.0, 3, 0.0
def step(h, s):                                      # bloc gate par coordonnee, r=1
    a = sigmoid(s @ h + b); return h + (a / L) * u * (w @ h)
def iterate(h, s, T=T):
    for _ in range(T): h = step(h, s)
    return h
s_ker = np.array([0., 0, 1])                         # lit h3 : s^T (I-P) = (0,0,1) != 0  -> m = 1, A = s^T u = 0
print("     s=(0,0,1) : s^T(I-P) =", s_ker @ Q, " A = s^T u =", s_ker @ u)
h  = np.array([1.0, 0.5, 2.0]); hp = np.array([0.2, 1.3, 2.0])    # meme code c = h1+h2 = 1.5, meme beta = h3 = 2
for hh in (h, hp):
    hT = iterate(hh, s_ker)
    print(f"     h0 = {hh} : c0 = {w@hh:.2f}, beta = {hh[2]:.1f} -> h_T = {np.round(hT, 4)}, c_T = {w@hT:.4f}, residu (I-P)h_T = {np.round(Q@hT,4)}")
cT_formula = (w @ h) * (1 + sigmoid(h[2] + b) / L) ** T
print(f"     forme close c_T = c0 (1 + sigma(beta)/L)^T = 1.5 * (1 + {sigmoid(2.0):.4f}/2)^3 = {cT_formula:.4f}")
Fk = lambda hh: step(hh, s_ker); a0 = sigmoid(s_ker @ h + b)
print(f"     det nabla F (determinant r x r, A=0) = 1 + a/L = 1 + {a0:.4f}/2 = {1 + a0 / L:.4f} ; diff. finies : {np.linalg.det(num_jac(Fk, h)):.4f}")
s_span = w.copy()                                    # s dans span(W) : s^T (I-P) = 0 -> m = 0
print("     s = w=(1,1,0) : s^T(I-P) =", s_span @ Q)
h3a = np.array([1.0, 0.5, 2.0]); h3b = np.array([1.0, 0.5, -7.0])  # meme code, h3 tres different
print(f"     s=w : h3=2 -> c_T = {w@iterate(h3a, s_span):.4f} ; h3=-7 -> c_T = {w@iterate(h3b, s_span):.4f}  (le noyau est ignore)")

# --- A.2 : r=1, A != 0 : monotonie et une seule courbe de frontiere psi_T(beta)
s_mix = np.array([1., 0, 1])                         # A = s^T u = 1, beta = s^T (I-P) h = h3 - h2
def cT_of(c, beta, T):
    for _ in range(T): c = c * (1 + sigmoid(1.0 * c + beta + b) / L)
    return c
print("A.2  s=(1,0,1) : A = 1. Table c_T(c0, beta=0.5) :")
for c0 in [-2, -1, 0, 1, 2]:
    print(f"       c0={c0:+d} : T=1 -> {cT_of(c0, 0.5, 1):+.4f}   T=6 -> {cT_of(c0, 0.5, 6):+.4f}")
# frontiere v c_T = theta  <=>  c0 > psi_T(beta) : resolution par bissection
def psi(beta, T, theta=1.0):
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cT_of(mid, beta, T) < theta: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)
print("     seuil c_T = 1 :  beta      psi_1(beta)   psi_6(beta)")
for beta in [-2.0, 0.0, 2.0]:
    print(f"                     {beta:+.1f}      {psi(beta,1):+.4f}       {psi(beta,6):+.4f}")

# --- A.3 : d=4, r=2, m=1 vs m=2
U = np.array([[1., 0], [0, 1], [0, 0], [0, 0]]); W = np.array([[1., 0], [0, 1], [1, 0], [0, 1]])
P4 = U @ W.T; Q4 = np.eye(4) - P4
print("A.3  W^T U = I ?", np.allclose(W.T @ U, np.eye(2)), "| P h = (h1+h3, h2+h4, 0, 0)")
def iter4(h, S, T=4, L=2.0):
    for _ in range(T):
        a = sigmoid(S.T @ h); h = h + (1 / L) * U @ (a * (W.T @ h))
    return h
h  = np.array([1.0, 0.5, 0.3, 0.8]); hp = np.array([1.0, 3.3, 0.3, -2.0])   # meme code (1.3, 1.3), meme h3, h4 different
S_m1 = np.array([[0., 0], [0, 0], [1, 1], [0, 0]])   # les deux portes lisent h3 -> m = rang(S^T(I-P)) = 1
S_m2 = np.array([[0., 0], [0, 0], [1, 0], [0, 1]])   # portes lisent h3 et h4 -> m = 2
for name, S in [("m=1 (S lit h3 seulement)", S_m1), ("m=2 (S lit h3 et h4)", S_m2)]:
    print(f"     {name} : rang S^T(I-P) = {np.linalg.matrix_rank(S.T @ Q4)} | c_T(h) = {np.round(W.T@iter4(h,S),4)}  c_T(h') = {np.round(W.T@iter4(hp,S),4)}")
# ce que la profondeur change et ne change pas (m = 1) : pour tout T, h et h' restent indiscernables par le code ;
# la fonction calculee sur les coordonnees accessibles, elle, depend de T.
h2 = np.array([2.0, 0.5, -1.0, 0.8])                 # autre entree : code (1.0, 1.3), h3 = -1.0
for T in (4, 8, 40):
    cT, cTp, cT2 = W.T @ iter4(h, S_m1, T), W.T @ iter4(hp, S_m1, T), W.T @ iter4(h2, S_m1, T)
    print(f"     m=1, T={T:2d} : c_T(h) = c_T(h') ? {np.allclose(cT, cTp)} | c_T(h)[0] / c_T(h2)[0] = {cT[0] / cT2[0]:.4f}  (rapport initial des codes : {1.3 / 1.0:.4f})")

print("\n=== Exemples : capacite quadrique du triceptron ===")
# --- B.1 inclusion
z = 2.0
print(f"B.1  s=w, kappa=1 : sigma(2)*2 = {sigmoid(2.0)*2:.4f} = SiLU(2) ; kappa=50 : {sigmoid(100)*2:.4f} = ReLU(2) ; z=-2 : {sigmoid(-100)*(-2):.2e} = ReLU(-2)")
# --- B.2 XOR : 1 unite + saut
k = 0.05
f_xor = lambda x1, x2: sigmoid(k * x1) * x2 - 0.5 * x2
print("B.2  XOR, f = sigma(0.05 x1) x2 - x2/2  (= 0.0125 x1 x2 + O(kappa^3)) :")
for x1, x2 in [(1, 1), (1, -1), (-1, 1), (-1, -1), (2, 0.5), (-2, 0.5)]:
    print(f"       f({x1:+d},{x2:+.1f}) = {f_xor(x1, x2):+.5f}  (0.0125 x1 x2 = {0.0125*x1*x2:+.5f})  signe {'OK' if np.sign(f_xor(x1,x2))==np.sign(x1*x2) else 'KO'}")
# --- B.3 cercle : 2 unites + saut, seuil kappa/4 R^2
f_circ = lambda x1, x2: sigmoid(k * x1) * x1 + sigmoid(k * x2) * x2 - 0.5 * (x1 + x2)
print("B.3  cercle R=1, f = sum_i [sigma(0.05 x_i) x_i - x_i/2] (= 0.0125 (x1^2+x2^2)), seuil 0.0125 :")
for x1, x2 in [(0.9, 0), (1.1, 0), (0.7, 0.7), (0.6, 0.9), (-0.5, 0.5)]:
    v = f_circ(x1, x2); print(f"       f({x1:+.1f},{x2:+.1f}) = {v:.5f} -> {'exterieur' if v > 0.0125 else 'interieur'} (vrai : {'exterieur' if x1**2+x2**2>1 else 'interieur'})")
# --- B.4 inertie : hyperbole, ellipse, selle
def symm(s, w): return 0.5 * (np.outer(s, w) + np.outer(w, s))
print("B.4  hyperbole x1^2 - x2^2 : s=(1,1), w=(1,-1) : sym(s w^T) =\n", symm(np.array([1.,1]), np.array([1.,-1])))
print("     ellipse 2x1^2 + x2^2 : s=w=(sqrt2,0) et s=w=(0,1) : somme =\n", symm(np.array([2**0.5,0]), np.array([2**0.5,0])) + symm(np.array([0.,1]), np.array([0.,1])))
Qs = symm(np.array([1.,0,1]), np.array([1.,0,-1])) + symm(np.array([0.,1,0]), np.array([0.,1,0]))
print("     selle x1^2 + x2^2 - x3^2 (inertie (2,1)) : 2 unites, s=(1,0,1),w=(1,0,-1) + s=w=e2 :\n", Qs, "\n     valeurs propres :", np.round(np.linalg.eigvalsh(Qs), 6))
# --- B.5 borne ReLU : erreur du N-gone
print("B.5  cercle R=1 : erreur de Hausdorff du N-gone inscrit, N(m) = 1+m+m(m-1)/2 :")
for m in [2, 3, 4, 6, 10, 20]:
    N = 1 + m + m * (m - 1) // 2; print(f"       m={m:2d} -> N<={N:3d} -> erreur >= 1-cos(pi/N) = {1-np.cos(np.pi/N):.6f}")
for eps in [1e-2, 1e-3, 1e-4]:
    N = int(np.ceil(np.pi / np.arccos(1 - eps))); m = int(np.ceil((-1 + np.sqrt(1 + 8 * (N - 1))) / 2))
    print(f"       epsilon={eps:.0e} : N >= {N} cotes -> m >= {m} unites ReLU (N({m}) = {1+m+m*(m-1)//2}) ; triceptrons doux : 2")
# --- B.6 regime dur : le commutateur 1[x1>0] x2
print("B.6  commutateur 1[x1>0]*x2 : triceptron dur s=e1, w=e2 exact ; un reseau ReLU (continu) ne peut que l'approcher.")
for x in [(1, 3), (-1, 3), (0.01, 3), (-0.01, 3)]:
    print(f"       x={x} : sigma(50 x1) x2 = {sigmoid(50*x[0])*x[1]:.4f}")
