# -*- coding: utf-8 -*-
"""
Mjerne velicine iz nacrta v3 §7.  Sve rucno, bez scipy -- v3 §12.4 trazi da
NumPy bude jedina tvrda zavisnost, da bi se .deb sigurno instalirao na Mint 4 GB.

  norm_ppf      inverzna normalna (Acklam), za d'
  dprime_*      d' iz detekcione teorije, s log-linearnom korekcijom
  roc_auc       Mann-Whitney U -- tacno hvata izjednacenja, kojih ima mnogo
                jer je odlucivacka varijabla cio broj spike-ova
  auc_log       glavni odziv: povrsina ispod d'(log dr)
  dr_star       najvece kasnjenje nagrade pri kojem se zadatak jos uci
  participation_ratio   provjera tvrdnje o vektorskom theta (v3 §6.10)
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------
# Inverzna normalna -- Acklam, |greska| < 1.15e-9
# ------------------------------------------------------------------
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_PLOW, _PHIGH = 0.02425, 1.0 - 0.02425


def norm_ppf(p: float) -> float:
    """Inverzna standardna normalna CDF."""
    if not (0.0 < p < 1.0):
        return float("-inf") if p <= 0.0 else float("inf")
    if p < _PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if p > _PHIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ------------------------------------------------------------------
# d' po fiksnom kriteriju
# ------------------------------------------------------------------
def dprime_fixed(sig_counts, noise_counts, k_crit: int) -> dict:
    """
    d' pri kriteriju "broj spike-ova >= k_crit".

    Log-linearna korekcija (+0.5 / +1) je OBAVEZNA -- inace z(1) = inf cim
    jedan run bude savrsen, i cijeli sweep pukne na jednoj tacki.  v3 §7.1
    """
    sig = np.asarray(sig_counts)
    noi = np.asarray(noise_counts)
    n_s, n_n = int(sig.size), int(noi.size)
    if n_s == 0 or n_n == 0:
        return dict(dprime=0.0, hit=0.0, fa=0.0, n_signal=n_s, n_noise=n_n)
    h = int(np.count_nonzero(sig >= k_crit))
    f = int(np.count_nonzero(noi >= k_crit))
    hit = (h + 0.5) / (n_s + 1.0)
    fa = (f + 0.5) / (n_n + 1.0)
    return dict(dprime=norm_ppf(hit) - norm_ppf(fa),
                hit=hit, fa=fa, hit_raw=h / n_s, fa_raw=f / n_n,
                n_signal=n_s, n_noise=n_n)


# ------------------------------------------------------------------
# ROC preko Mann-Whitney U
# ------------------------------------------------------------------
def roc_auc(sig_counts, noise_counts) -> float:
    """
    P(count_signal > count_noise) + 0.5*P(jednako).

    Racuna se preko rangova jer je odlucivacka varijabla cio broj spike-ova,
    pa je izjednacenja mnogo; naivno prebrojavanje bi ih tiho bacilo na jednu
    stranu i napravilo pristrasnost u d'.
    """
    sig = np.asarray(sig_counts, dtype=np.float64)
    noi = np.asarray(noise_counts, dtype=np.float64)
    n_s, n_n = sig.size, noi.size
    if n_s == 0 or n_n == 0:
        return 0.5
    allv = np.concatenate([sig, noi])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(allv.size, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < sv.size:                       # srednji rang za svaku grupu jednakih
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_s = ranks[:n_s].sum()
    u = r_s - n_s * (n_s + 1.0) / 2.0
    return float(u / (n_s * n_n))


def dprime_roc(sig_counts, noise_counts) -> dict:
    """
    d' izveden iz ROC povrsine: d'_a = sqrt(2) * z(AUC).

    Ovo je mjera koja NE visi o proizvoljnom kriteriju -- razlog zbog kojeg je
    v3 §6.10 izabrao gradirani readout umjesto OR.  Uz OR bi ovdje postojala
    tacno jedna (hit, FA) tacka i ova funkcija ne bi imala sta da integrise.
    """
    auc = roc_auc(sig_counts, noise_counts)
    auc_c = min(max(auc, 1e-6), 1.0 - 1e-6)
    return dict(auc=auc, dprime=math.sqrt(2.0) * norm_ppf(auc_c))


# ------------------------------------------------------------------
# Glavni odziv:  AUC nad log dr
# ------------------------------------------------------------------
def auc_log(dr_values, dprimes) -> float:
    """
    Trapezno po log dr, normalizovano rasponom.  v3 §7.2

    Log je prirodna varijabla jer je mehanizam eksponencijalno curenje traga;
    ravnomjeran raspored u logu daje posten prosjek.
    """
    dr = np.asarray(dr_values, dtype=np.float64)
    dp = np.asarray(dprimes, dtype=np.float64)
    if dr.size < 2:
        return float(dp[0]) if dp.size else 0.0
    order = np.argsort(dr)
    x = np.log(dr[order])
    y = dp[order]
    return float(np.trapezoid(y, x) / (x[-1] - x[0])) if hasattr(np, "trapezoid") \
        else float(np.trapz(y, x) / (x[-1] - x[0]))


# ------------------------------------------------------------------
# dr*  -- prag, izvjestava se ali se NE fituje  (v3 §7.2)
# ------------------------------------------------------------------
def _sigmoid(x, dmax, x0, s):
    return dmax / (1.0 + np.exp((x - x0) / max(s, 1e-6)))


def _nelder_mead(fun, x0, iters=400):
    """Minimalni NM -- 3 parametra, bez scipy.  Deterministican."""
    n = len(x0)
    pts = [np.array(x0, dtype=np.float64)]
    for i in range(n):
        p = np.array(x0, dtype=np.float64)
        p[i] = p[i] + (0.1 * abs(p[i]) if p[i] != 0 else 0.1)
        pts.append(p)
    pts = np.array(pts)
    val = np.array([fun(p) for p in pts])
    for _ in range(iters):
        idx = np.argsort(val, kind="stable")
        pts, val = pts[idx], val[idx]
        if abs(val[-1] - val[0]) < 1e-12:
            break
        cen = pts[:-1].mean(axis=0)
        xr = cen + (cen - pts[-1]); fr = fun(xr)
        if fr < val[0]:
            xe = cen + 2.0 * (cen - pts[-1]); fe = fun(xe)
            pts[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < val[-2]:
            pts[-1], val[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = fun(xc)
            if fc < val[-1]:
                pts[-1], val[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                val[1:] = [fun(p) for p in pts[1:]]
    i = int(np.argmin(val))
    return pts[i], float(val[i])


def dr_star(dr_values, dprimes, crit: float = 1.0) -> dict:
    """
    Kasnjenje nagrade pri kojem d' pada na `crit`.

    Pokusava sigmoidni fit; ako ne uspije ili je tacaka premalo, vraca se na
    monotonu interpolaciju u log prostoru.  Cenzura se prijavljuje EKSPLICITNO
    umjesto da se sakrije kao rubna vrijednost rjesetke -- to je bio jedan od
    razloga zasto dr* nije glavni odziv (v3 §7.2).
    """
    dr = np.asarray(dr_values, dtype=np.float64)
    dp = np.asarray(dprimes, dtype=np.float64)
    order = np.argsort(dr)
    x, y = np.log(dr[order]), dp[order]

    if y.max() < crit:
        return dict(dr_star=float(dr[order][0]), censored="left", method="none")
    if y.min() >= crit:
        return dict(dr_star=float(dr[order][-1]), censored="right", method="none")

    fit = None
    if x.size >= 4:
        def loss(p):
            return float(np.sum((y - _sigmoid(x, p[0], p[1], p[2])) ** 2))
        p0 = [max(y.max(), 1e-3), float(np.median(x)), 0.5]
        p, err = _nelder_mead(loss, p0)
        if p[0] > crit and p[2] > 1e-4 and np.isfinite(err):
            # d' = crit  =>  x = x0 + s*ln(dmax/crit - 1)
            ratio = p[0] / crit - 1.0
            if ratio > 0:
                fit = float(math.exp(p[1] + p[2] * math.log(ratio)))
    if fit is not None and dr[order][0] * 0.5 <= fit <= dr[order][-1] * 2.0:
        return dict(dr_star=fit, censored=None, method="sigmoid")

    for i in range(len(y) - 1):               # monotona interpolacija u logu
        if (y[i] - crit) * (y[i + 1] - crit) <= 0 and y[i] != y[i + 1]:
            t = (y[i] - crit) / (y[i] - y[i + 1])
            return dict(dr_star=float(math.exp(x[i] + t * (x[i + 1] - x[i]))),
                        censored=None, method="interp")
    return dict(dr_star=float(dr[order][-1]), censored="right", method="interp")


# ------------------------------------------------------------------
# Participation ratio -- v3 §6.10 / §7.3 / §11
# ------------------------------------------------------------------
def participation_ratio(rates) -> float:
    """
    PR = (sum rho)^2 / sum rho^2.

    PR ~ 1  -> jedan neuron radi sve (winner-take-all)
    PR ~ n  -> ravnomjerno

    Ovo je broj kojim tvrdnja "theta mora biti vektor" postaje provjerljiva
    umjesto da se vjeruje na rijec.  Ako skalarni prag da PR ~ 1 uz JEDNAK d',
    tvrdnja pada -- i to je zapisano kao falsifikator u v3 §11.
    """
    r = np.asarray(rates, dtype=np.float64)
    s2 = float(np.sum(r * r))
    if s2 <= 0.0:
        return 0.0
    return float(np.sum(r) ** 2 / s2)


# ------------------------------------------------------------------
# Sitno
# ------------------------------------------------------------------
def bimodality(w, w_max: float = 1.0, n_bins: int = 20) -> dict:
    """
    Histogram tezina + dva broja koja hvataju kvarove iz v3 §7.3:
    udio zalijepljen uz w_max (premali A_minus ili obrnut predznak, §6.9)
    i udio na nuli.
    """
    w = np.asarray(w, dtype=np.float64).ravel()
    hist, edges = np.histogram(w, bins=n_bins, range=(0.0, w_max))
    return dict(hist=hist.tolist(), edges=edges.tolist(),
                frac_at_max=float(np.mean(w >= 0.98 * w_max)),
                frac_at_zero=float(np.mean(w <= 0.02 * w_max)),
                mean=float(w.mean()), sd=float(w.std()))
