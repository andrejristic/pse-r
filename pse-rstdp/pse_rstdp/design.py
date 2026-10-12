# -*- coding: utf-8 -*-
"""
Mixture dizajn: Scheffe model, lack-of-fit, najbrzi uspon.   v3 §8

Care Invariant je SMJESA (w_s + w_o + w_f = 1), pa se koristi Scheffe model
bez intercepta -- ogranicenje ga apsorbuje.  To nije trik nego posljedica
aksioma koji je vec postavljen.

    y = b_s*w_s + b_o*w_o + b_f*w_f + b_so*w_s*w_o + b_sf*w_s*w_f + b_of*w_o*w_f

Clanovi b_so, b_sf, b_of su tacno ono sto jednodimenzionalna skeniranja iz v1
NE mogu vidjeti: da li kanali saradjuju ili se ponistavaju preko onoga sto
svaki nosi sam.

Sve rucno, bez scipy (v3 §12.4).
"""
from __future__ import annotations

import math

import numpy as np

TERMS = ("b_s", "b_o", "b_f", "b_so", "b_sf", "b_of")
TERMS_CUBIC = TERMS + ("b_sof",)


# ------------------------------------------------------------------
# F raspodjela preko regularizovane nepotpune beta funkcije
# ------------------------------------------------------------------
def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    """Regularizovana nepotpuna beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_sf(f: float, d1: int, d2: int) -> float:
    """P(F > f).  Vraca 1.0 kad test nije definisan."""
    if d1 <= 0 or d2 <= 0 or f <= 0 or not math.isfinite(f):
        return 1.0
    return float(betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f)))


# ------------------------------------------------------------------
# Model matrica
# ------------------------------------------------------------------
def _row(c, cubic=False):
    s, o, f = c
    r = [s, o, f, s * o, s * f, o * f]
    if cubic:
        r.append(s * o * f)
    return r


def design_matrix(cs, cubic=False):
    return np.array([_row(c, cubic) for c in cs], dtype=np.float64)


# ------------------------------------------------------------------
# Fit + lack-of-fit
# ------------------------------------------------------------------
def fit_scheffe(points, cubic: bool = False, alpha: float = 0.05) -> dict:
    """
    points: lista (c, [y_po_sjemenu, ...]).

    Lack-of-fit trazi PURE ERROR iz replikata -- zato sjemena nisu luksuz nego
    uslov da se pretpostavka glatkoce uopste moze provjeriti (v3 §8.3).
    Bez replikata test ne postoji i to se prijavljuje eksplicitno.
    """
    cs = [p[0] for p in points]
    ys = [np.asarray(p[1], dtype=np.float64) for p in points]
    if not cs:
        return dict(ok=False, reason="nema tacaka")

    x_all, y_all = [], []
    for c, yy in zip(cs, ys):
        for v in yy:
            x_all.append(_row(c, cubic))
            y_all.append(v)
    X = np.array(x_all)
    y = np.array(y_all)
    k = X.shape[1]
    if X.shape[0] <= k:
        return dict(ok=False, reason="premalo opservacija za %d parametara" % k)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    n = X.shape[0]
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

    # pure error / lack of fit
    ss_pure, df_pure = 0.0, 0
    ss_lof, df_lof = 0.0, 0
    for c, yy in zip(cs, ys):
        m = yy.mean()
        ss_pure += float(np.sum((yy - m) ** 2))
        df_pure += yy.size - 1
        pred = float(np.array(_row(c, cubic)) @ beta)
        ss_lof += yy.size * (m - pred) ** 2
    df_lof = len(cs) - k

    lof = dict(available=bool(df_pure > 0 and df_lof > 0))
    if lof["available"]:
        ms_lof = ss_lof / df_lof
        ms_pe = ss_pure / df_pure
        F = ms_lof / ms_pe if ms_pe > 0 else float("inf")
        pval = f_sf(F, df_lof, df_pure)
        lof.update(F=float(F), df_lof=int(df_lof), df_pure=int(df_pure),
                   p_value=float(pval), significant=bool(pval < alpha))
    else:
        lof["reason"] = ("nema replikata -- pure error se ne moze ocijeniti; "
                         "test lack-of-fit NE postoji (v3 §8.3)")

    # optimum nad simpleksom (barycentricna rjesetka)
    g = _simplex_grid(0.02)
    vals = design_matrix(g, cubic) @ beta
    i = int(np.argmax(vals))
    best_c = tuple(float(v) for v in g[i])
    on_edge = min(best_c) < 0.03

    # reziduali po koordinati -- sistematski isti znak blizu ivice je dijagnoza
    # i kad F nije znacajan (v3 §8.3)
    res_by_coord = {}
    resid = y - yhat
    for j, nm in enumerate(("w_s", "w_o", "w_f")):
        cv = np.array([c[j] for c, yy in zip(cs, ys) for _ in yy])
        hi = resid[cv >= np.median(cv)]
        lo = resid[cv < np.median(cv)]
        res_by_coord[nm] = dict(mean_resid_high=float(hi.mean()) if hi.size else 0.0,
                                mean_resid_low=float(lo.mean()) if lo.size else 0.0)

    names = TERMS_CUBIC if cubic else TERMS
    return dict(ok=True, cubic=bool(cubic),
                coefficients={nm: float(b) for nm, b in zip(names, beta)},
                r2=float(r2), r2_adj=float(r2adj), n_obs=int(n), n_params=int(k),
                lack_of_fit=lof,
                lack_of_fit_significant=bool(lof.get("significant", False)),
                optimum_c=list(best_c), optimum_value=float(vals[i]),
                optimum_on_edge=bool(on_edge),
                residual_by_coordinate=res_by_coord,
                interactions={nm: float(b) for nm, b in
                              zip(names, beta) if nm.startswith("b_") and len(nm) > 3})


def escalate(points, alpha: float = 0.05) -> dict:
    """
    Kvadrat -> ako lack-of-fit puca, specijalni kubni clan -> ako i dalje puca,
    povrsina NIJE glatka i ne smije se fitovati: izvijesti sirove tacke.  (v3 §8.3)
    """
    q = fit_scheffe(points, cubic=False, alpha=alpha)
    if not q.get("ok") or not q.get("lack_of_fit_significant"):
        q["verdict"] = "kvadratni model prolazi"
        return q
    cu = fit_scheffe(points, cubic=True, alpha=alpha)
    if cu.get("ok") and not cu.get("lack_of_fit_significant"):
        cu["verdict"] = "potreban specijalni kubni clan"
        cu["quadratic"] = q
        return cu
    cu = cu if cu.get("ok") else q
    cu["verdict"] = ("povrsina nije glatka -- mjesavinski model se NE smije "
                     "koristiti kao opis; vrijede samo sirove tacke (v3 §8.3)")
    cu["quadratic"] = q
    return cu


# ------------------------------------------------------------------
# Pomocno
# ------------------------------------------------------------------
def _simplex_grid(step: float):
    n = int(round(1.0 / step))
    out = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            out.append((i / n, j / n, (n - i - j) / n))
    return out


def steepest_ascent_points(fit: dict, n: int = 6, span: float = 0.35):
    """
    Ako je optimum na ivici, kvadrat daje PRAVAC najbrzeg uspona; dodaje se
    nekoliko tacaka duz njega.  Sekvencijalno i informisano, umjesto naslijepo.
    (v3 §8.2, faza 2)
    """
    if not fit.get("ok"):
        return []
    b = fit["coefficients"]
    c0 = np.array(fit.get("optimum_c", [1 / 3, 1 / 3, 1 / 3]), dtype=np.float64)
    s, o, f = c0
    grad = np.array([b["b_s"] + b["b_so"] * o + b["b_sf"] * f,
                     b["b_o"] + b["b_so"] * s + b["b_of"] * f,
                     b["b_f"] + b["b_sf"] * s + b["b_of"] * o])
    grad = grad - grad.mean()                      # projekcija na simpleks
    nrm = np.linalg.norm(grad)
    if nrm < 1e-12:
        return []
    grad /= nrm
    pts = []
    for i in range(1, n + 1):
        c = c0 + grad * (span * i / n)
        c = np.clip(c, 0.0, 1.0)
        t = c.sum()
        if t <= 0:
            continue
        c = c / t
        cand = tuple(round(float(v), 4) for v in c)
        if cand not in pts:
            pts.append(cand)
    return pts
