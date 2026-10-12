# -*- coding: utf-8 -*-
"""
Zadatak i faza R0 -- validnost zadatka.   v3 §2, §10 R0

R0 postoji zato sto sve dalje visi o jednoj tvrdnji iz §2: "Marginalne
statistike su identicne ... jedina razlika je u relativnom tajmingu."
v1 je tu tvrdnju PRETPOSTAVIO.  Ovdje se dokazuje, i to prije nego se napise
ijedan neuron.

VAZNA ISPRAVKA KOJA JE ISPALA IZ IMPLEMENTACIJE R0
--------------------------------------------------
Kako je NOISE bio opisan u v1-v3 ("t_B ~ U(0,400), nezavisno"), marginale NISU
bile identicne:

    SIGNAL: t_B = t_A + lag        ->  t_B ~ U(0,W) (+) lag
    NOISE:  t_B = U(0,W)           ->  t_B ~ U(0,W)

Srednje vrijeme B burst-a bi se razlikovalo za ~20 ms izmedju klasa, pa bi
obican detektor srednjeg vremena rijesio zadatak bez ikakvog tajminga.  To je
tacno kvar koji R0 treba da uhvati -- i uhvatio ga je.

Ispravka: u NOISE se t_B gradi iz NEZAVISNOG izvlacenja t_A' i NEZAVISNOG
izvlacenja lag-a:

    SIGNAL: t_A ~ U(0,W),  t_B = t_A  + lag
    NOISE:  t_A ~ U(0,W),  t_B = t_A' + lag'      (t_A' _|_ t_A,  lag' ~ ista raspodjela)

Sada su obje marginale identicne po konstrukciji (ista konvolucija), a razlikuje
se samo ZAVISNOST.  Posljedica koja cini zadatak onim sto tvrdi da jeste:

    SIGNAL:  t_B - t_A ~ N(20, 3)              sd ~   3 ms
    NOISE:   t_B - t_A ~ (U-U) (+) N(20,3)     sd ~ 163 ms

Iste srednje vrijednosti, drasticno razlicite varijanse.  Zato nijedan LINEARAN
model nad marginalama (ni nad njihovom razlikom) ne moze nista, a sve sto gleda
|t_B - t_A - 20| rjesava odmah -- a to je doslovno koincidencijski detektor.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Params, N_A, N_B
from .rng import make_rng

SIGNAL, NOISE = 1, 0


# ------------------------------------------------------------------
# Generator trialova
# ------------------------------------------------------------------
def make_trials(rng: np.random.Generator, p: Params, n: int) -> dict:
    """n trialova.  Vraca klasu i vremena burst-a za oba kanala."""
    kind = (rng.random(n) < 0.5).astype(np.int8)          # 1 = SIGNAL

    t_a = rng.uniform(0.0, p.t_max_burst, n)
    lag_sig = p.lag_true + rng.normal(0.0, p.jitter, n)

    # NOISE: nezavisno t_A' i nezavisan lag' -> ista marginala kao SIGNAL
    t_a_alt = rng.uniform(0.0, p.t_max_burst, n)
    lag_noi = p.lag_true + rng.normal(0.0, p.jitter, n)

    t_b = np.where(kind == SIGNAL, t_a + lag_sig, t_a_alt + lag_noi)
    return dict(kind=kind, t_a=t_a, t_b=t_b)


def build_raster(rng: np.random.Generator, p: Params,
                 t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """
    Rasteri burst-ova za kanale A i B.  Oblik (n, trial_len, N_A + N_B) bool.

    Burst (v3 §2.1): 5 spike-ova na 200 Hz, svaki sa ziterom +-1 ms.
    Identicno za oba kanala i obje klase, pa marginale ostaju identicne po
    konstrukciji.

    burst_frac = 1.0 (svi neuroni kanala), ne 0.5 kako je v3 §2.1 predlagao:
    detektor iz R1 ima tacno jednu A-sinapsu i jednu B-sinapsu, pa bi pri 0.5
    oba bas njegova neurona bila u burstu tek u 25% trialova.  v. config.py.
    """
    n = t_a.size
    n_in = N_A + N_B
    raster = np.zeros((n, p.trial_len, n_in), dtype=bool)
    k = max(1, int(round(p.burst_frac * N_A)))

    for ch, (t0, off, npop) in enumerate(((t_a, 0, N_A), (t_b, N_A, N_B))):
        # nasumicnih k neurona po trialu, bez ponavljanja
        sel = np.argsort(rng.random((n, npop)), axis=1, kind="stable")[:, :k]
        steps = np.arange(p.burst_n, dtype=np.float64)[None, :, None] * p.burst_isi
        jit = rng.uniform(-p.burst_jitter, p.burst_jitter, (n, p.burst_n, k))
        tt = t0[:, None, None] + steps + jit
        ti = np.clip(np.rint(tt).astype(np.int64), 0, p.trial_len - 1)

        trial_i = np.repeat(np.arange(n), p.burst_n * k)
        neur_i = np.tile(sel[:, None, :], (1, p.burst_n, 1)).reshape(-1) + off
        raster[trial_i, ti.reshape(-1), neur_i] = True
    return raster


def shuffle_raster(rng: np.random.Generator, raster: np.ndarray) -> np.ndarray:
    """
    Razbija A-B korelaciju drzeci marginale netaknutima: permutuje trialove
    nezavisno po kanalima.  Koristi kalibracijski run u R2 (v3 §6.9).
    """
    n = raster.shape[0]
    out = raster.copy()
    out[:, :, N_A:] = raster[rng.permutation(n)][:, :, N_A:]
    return out


# ------------------------------------------------------------------
# Obiljezja za R0
# ------------------------------------------------------------------
def marginal_features(raster: np.ndarray) -> dict:
    """
    Samo MARGINALNA obiljezja -- ona koja detektor stope stvarno ima.
    Nista sto poravnava kanale u vremenu.
    """
    n, tlen, _ = raster.shape
    tvec = np.arange(tlen, dtype=np.float64)
    out = {}
    for name, sl in (("a", slice(0, N_A)), ("b", slice(N_A, N_A + N_B))):
        ch = raster[:, :, sl]
        cnt = ch.sum(axis=(1, 2)).astype(np.float64)
        per_t = ch.sum(axis=2).astype(np.float64)
        tot = np.maximum(cnt, 1.0)
        mean_t = (per_t * tvec[None, :]).sum(axis=1) / tot
        var_t = (per_t * (tvec[None, :] - mean_t[:, None]) ** 2).sum(axis=1) / tot
        out["count_" + name] = cnt
        out["rate_" + name] = cnt / tlen
        out["mean_t_" + name] = mean_t
        out["sd_t_" + name] = np.sqrt(np.maximum(var_t, 0.0))
        out["peak_" + name] = per_t.max(axis=1)
    out["count_total"] = out["count_a"] + out["count_b"]
    # razlika srednjih vremena: LINEARNA kombinacija dvije marginale, pa je
    # logisticka regresija ionako moze formirati.  Ukljucena eksplicitno da
    # test bude stroziji, ne blazi.
    out["dmean"] = out["mean_t_b"] - out["mean_t_a"]
    return out


# ------------------------------------------------------------------
# Logisticka regresija (IRLS, bez scipy)
# ------------------------------------------------------------------
def _logistic_fit(z: np.ndarray, y: np.ndarray, iters: int = 60) -> np.ndarray:
    w = np.zeros(z.shape[1])
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
        wt = np.maximum(pr * (1.0 - pr), 1e-6)
        grad = z.T @ (y - pr)
        hess = (z * wt[:, None]).T @ z + 1e-6 * np.eye(z.shape[1])
        try:
            w = w + np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
    return w


def logistic_accuracy(x: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """
    Tacnost UNAKRSNOM VALIDACIJOM, ne na istim podacima.

    In-sample tacnost je pristrasna navise za ~sqrt(k/n): sa 12 obiljezja i
    600 trialova daje 0.53-0.57 i na cistom sumu, pa gate [0.45, 0.55] pada
    bez ikakvog razloga.  Uz CV je procjena nepristrasna i gate radi na svakoj
    velicini uzorka.
    """
    n, k = x.shape
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    z = np.hstack([(x - mu) / sd, np.ones((n, 1))])
    folds = max(2, min(folds, n // 20))
    idx = np.arange(n)
    correct = 0
    for f in range(folds):
        te = idx % folds == f
        tr = ~te
        if tr.sum() < k + 2 or te.sum() == 0:
            continue
        w = _logistic_fit(z[tr], y[tr])
        pr = 1.0 / (1.0 + np.exp(-np.clip(z[te] @ w, -30, 30)))
        correct += int(np.sum((pr >= 0.5).astype(np.int8) == y[te]))
    return float(correct / n)


def chance_band(n: int, k_sd: float = 4.0) -> float:
    """Polusirina pojasa oko 0.5 koja jos znaci 'na slucajnom nivou'."""
    return max(0.03, k_sd * math.sqrt(0.25 / max(n, 1)))


# ------------------------------------------------------------------
# R0
# ------------------------------------------------------------------
def run_r0(p: Params, seed: int, n_trials: int = 4000) -> dict:
    """
    Tri provjere.  Prva dvije moraju PROCI (biti na slucajnom nivou),
    treca mora PASTI (zadatak mora biti rjesiv) -- inace R1 nema sta da trazi.
    """
    rng = make_rng(seed, "r0")
    tr = make_trials(rng, p, n_trials)
    raster = build_raster(rng, p, tr["t_a"], tr["t_b"])
    y = tr["kind"].astype(np.float64)
    f = marginal_features(raster)

    # --- R0a: identicnost marginala, standardizovana razlika po obiljezju ---
    marg = {}
    worst_name, worst_d = None, 0.0
    for name in ("count_a", "count_b", "rate_a", "rate_b", "mean_t_a", "mean_t_b",
                 "sd_t_a", "sd_t_b", "peak_a", "peak_b", "count_total", "dmean"):
        v = f[name]
        s, nz = v[y == 1], v[y == 0]
        pooled = np.sqrt(0.5 * (s.var() + nz.var()))
        d = 0.0 if pooled < 1e-12 else abs(s.mean() - nz.mean()) / pooled
        marg[name] = dict(cohen_d=float(d), mean_signal=float(s.mean()),
                          mean_noise=float(nz.mean()))
        if d > worst_d:
            worst_d, worst_name = d, name

    # --- R0b: detektor stope + marginale.  Mora biti ~50%. ---
    rate_only = np.column_stack([f["count_a"], f["count_b"], f["count_total"],
                                 f["rate_a"], f["rate_b"], f["peak_a"], f["peak_b"]])
    acc_rate = logistic_accuracy(rate_only, y)

    full_marg = np.column_stack([rate_only,
                                 f["mean_t_a"], f["mean_t_b"],
                                 f["sd_t_a"], f["sd_t_b"], f["dmean"]])
    acc_marg = logistic_accuracy(full_marg, y)

    # --- R0c: koincidencijski orakl.  Mora RIJESITI, inace zadatak nije rjesiv. ---
    coincidence = np.abs(f["dmean"] - p.lag_true)[:, None]
    acc_oracle = logistic_accuracy(coincidence, y)

    band = chance_band(n_trials)
    gate_rate = abs(acc_rate - 0.5) <= band
    gate_marg = abs(acc_marg - 0.5) <= band
    gate_oracle = acc_oracle >= 0.80

    return dict(
        phase="R0", seed=seed, n_trials=n_trials,
        acc_rate_detector=acc_rate,
        acc_marginal_detector=acc_marg,
        acc_coincidence_oracle=acc_oracle,
        worst_marginal=worst_name, worst_cohen_d=worst_d, chance_band=band,
        marginals=marg,
        gate_rate_at_chance=bool(gate_rate),
        gate_marginals_at_chance=bool(gate_marg),
        gate_task_solvable=bool(gate_oracle),
        gate=bool(gate_rate and gate_marg and gate_oracle),
    )
