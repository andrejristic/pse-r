# -*- coding: utf-8 -*-
"""
Mjerenja i rucno ozicavanje.   v3 §7, §10 R1

  handwire          rucno postavi tezine da detektor RADI -- srz faze R1
  evaluate          d', ROC, PR, stope; uvijek sa zamrznutom plasticnoscu
  delay_scan        sanity check: pik odgovora nad razmakom A-B
  stdp_window       empirijski STDP prozor kroz PRAVI kod
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .config import Params
from .net import Network
from .rng import make_rng
from .stats import (bimodality, dprime_fixed, dprime_roc, participation_ratio)
from .task import SIGNAL, build_raster, make_trials


# ==================================================================
# R1 -- rucno ozicavanje
# ==================================================================
def handwire(net: Network, w_hi: float = 0.9, tol: int = 3,
             out_drive: float = 16.0, n_pairs: int = 3) -> dict:
    """
    Postavi tezine tako da mreza detektuje koincidenciju BEZ ucenja.

    Logika supstrata (v3 §4): A ispali u t_A, B u t_A + 20.  Dolasci se
    poklapaju kad je

        t_A + d_A  ==  t_A + 20 + d_B     =>     d_A - d_B == 20

    Dakle detektor je H neuron koji ima A-sinapsu s dugim kasnjenjem i
    B-sinapsu s kratkim, razlike ~20 ms.  Zato kasnjenja MORAJU biti i na
    ulaznim projekcijama, ne samo na H->H -- inace se ovaj par ne moze
    formirati u jednom skoku (v3 §4.1).

    TRI USLOVA KOJA JE PRVA VERZIJA PREKRSILA, sva nadjena u R1:

    1. Marza koincidencije je samo 2x: jedan kanal (burst od 5 spike-ova kroz
       tau_m = 20 ms) ne smije preci prag, a dva poklopljena moraju.  Prva
       verzija je postavljala SVE odgovarajuce parove punom tezinom, cime je
       jednokanalni pogon narastao na 3x i selektivnost nestala (d' = 0.07).

    1a. Ali JEDAN par je opet premalo: neuron tada sjedi tacno na pragu
       (izmjereno V_max = 12.29 mV od potrebnih 13.0), pa ga vec zbir zitera
       lag-a (sd 3 ms), zitera burst-a (+-1 ms) i neslaganja para (do tol ms)
       obara ispod praga -- hit je bio 0.53 i d' = 1.03.
       Rjesenje: do `n_pairs` parova, ali svaki s tezinom w_hi / n_pairs.
       Jednokanalni pogon ostaje w_hi, koincidentni 2*w_hi -- marza je
       NETAKNUTA -- a neuron sad integrise vise nezavisnih pomaka, pa se
       ziter usrednjava umjesto da obara.

    2. H->OUT se NORMALIZUJE brojem detektora.  Jedno globalno g_syn sluzi oba
       sloja; ako 43 detektora gadjaju OUT punom tezinom, OUT je zasicen davno
       prije nego H uopste postane aktivan.  Zato je zbir ulaznih tezina po
       OUT neuronu fiksiran na out_drive * w_hi -- pa OUT prelazi prag kad
       ispali otprilike out_drive/2 udio detektora, isto kao sto H prelazi na
       dva poklopljena dolaska.

    Ako ovo ne razdvoji SIGNAL od NOISE, nikakvo ucenje nece, i problem je u
    supstratu a ne u pravilu.  To je cijela svrha R1.
    """
    p = net.p
    lag = int(round(p.lag_true))
    net.w[:] = 0.0
    n_det = np.zeros(net.B, dtype=int)

    for b in range(net.B):
        pre, post, d = net.pre[b], net.post[b], net.d[b]
        is_a = pre < C.N_A
        is_b = (pre >= C.N_A) & (pre < C.N_A + C.N_B)
        to_h = (post >= C.SL_H.start) & (post < C.SL_H.stop)
        detectors = []
        for h in range(C.SL_H.start, C.SL_H.stop):
            m = to_h & (post == h)
            ia = np.flatnonzero(m & is_a)
            ib = np.flatnonzero(m & is_b)
            if ia.size == 0 or ib.size == 0:
                continue
            diff = np.abs(d[ia][:, None] - d[ib][None, :] - lag)
            flat = np.argsort(diff, axis=None, kind="stable")
            chosen = [np.unravel_index(int(k), diff.shape) for k in flat
                      if diff.flat[int(k)] <= tol][:n_pairs]
            if not chosen:
                continue
            # tezina podijeljena brojem parova -> marza 2x ostaje netaknuta
            wp = w_hi / len(chosen)
            for r, c_ in chosen:
                net.w[b, ia[r]] += wp
                net.w[b, ib[c_]] += wp
            detectors.append(h)
        n_det[b] = len(detectors)
        if detectors:
            det = np.asarray(detectors)
            m_out = np.isin(pre, det) & (post >= C.SL_OUT.start)
            # zbir po OUT neuronu fiksiran, bez obzira na broj detektora
            net.w[b, m_out] = out_drive * w_hi / max(len(detectors), 1)

    return dict(n_detectors=n_det.tolist(), out_drive=out_drive,
                w_target=float(np.mean([
                    net.w[b][net.post[b] >= C.SL_OUT.start].sum() / C.N_OUT
                    for b in range(net.B)])))


# ==================================================================
# Evaluacija
# ==================================================================
def evaluate(net: Network, n_trials: int = 400, rng_name: str = "eval") -> dict:
    """
    Uvijek sa zamrznutom plasticnoscu i homeostazom -- inace mjeris sondu,
    ne mrezu (v3 §7.1).  Odlucivacka varijabla je broj OUT spike-ova u prozoru
    odgovora, pa se kriterij prevlaci naknadno i dobija se puna ROC.
    """
    p = net.p
    saved = (p.plastic, p.homeostasis)
    net.p = p.with_(plastic=False, homeostasis=False)
    try:
        r = net.run_trials(n_trials, rng_name, learn=False, collect=True)
    finally:
        net.p = p

    counts, kind = r["counts"], r["kind"]
    out = []
    for b in range(net.B):
        sig = counts[b][kind[b] == SIGNAL]
        noi = counts[b][kind[b] != SIGNAL]
        fixed = dprime_fixed(sig, noi, p.k_crit)
        roc = dprime_roc(sig, noi)
        out.append(dict(
            dprime_roc=roc["dprime"], auc=roc["auc"],
            dprime_fixed=fixed["dprime"], hit=fixed["hit"], fa=fixed["fa"],
            mean_count_signal=float(np.mean(sig)) if sig.size else 0.0,
            mean_count_noise=float(np.mean(noi)) if noi.size else 0.0,
            pr_out=participation_ratio(r["rate_out_per_neuron_hz"]),
            rate_out_hz=float(r["rate_out_hz"]),
            rate_h_hz=float(r["rate_h_hz"]),
        ))
    return dict(per_batch=out,
                dprime_roc=float(np.mean([o["dprime_roc"] for o in out])),
                auc=float(np.mean([o["auc"] for o in out])),
                pr_out=float(np.mean([o["pr_out"] for o in out])),
                rate_out_hz=float(r["rate_out_hz"]),
                rate_h_hz=float(r["rate_h_hz"]))


def sweep_gain(net: Network, g_lo: float = 40.0, g_hi: float = 240.0, n_g: int = 6,
               od_lo: float = 4.0, od_hi: float = 64.0, n_od: int = 5,
               n_trials: int = 60) -> dict:
    """
    Zajednicki sweep (g_syn, out_drive).   Faza R1.

    Dva sloja imaju razlicite radne tacke i JEDNO globalno g_syn ih ne moze
    oba posluziti:

      A/B -> H : prag mora sjeci izmedju jednog kanala i dva poklopljena.
                 Marza je samo 2x, pa je prozor za g uzak.
      H -> OUT : detektori ne ispaljuju istovremeno nego rasuti kroz ~30 ms,
                 pa OUT vidi zbir oslabljen membranskim curenjem.  Potreban
                 pogon zavisi i od BROJA detektora, koji se ne zna unaprijed.

    Zato se ne pogadja analiticki (prva verzija je uzela out_drive = 4 i OUT
    nikad nije ispalio) nego se trazi na gruboj 2-D rjesetci.  Jednokratno --
    rezultat ide u measured.json i sve dalje ga koristi.
    """
    gs = np.exp(np.linspace(np.log(g_lo), np.log(g_hi), n_g))
    ods = np.exp(np.linspace(np.log(od_lo), np.log(od_hi), n_od))
    best = (-1e9, float(gs[0]), float(ods[0]), None)
    curve = []
    for g in gs:
        for od in ods:
            net.reset()
            hw = handwire(net, out_drive=float(od))
            net.g_syn = float(g)
            ev = evaluate(net, n_trials, "gainsweep")
            curve.append(dict(g=float(g), out_drive=float(od),
                              dprime=ev["dprime_roc"], auc=ev["auc"],
                              rate_out_hz=ev["rate_out_hz"],
                              rate_h_hz=ev["rate_h_hz"]))
            if ev["dprime_roc"] > best[0]:
                best = (ev["dprime_roc"], float(g), float(od), ev)
    net.reset()
    hw = handwire(net, out_drive=best[2])
    net.g_syn = best[1]
    on_edge = (best[1] in (float(gs[0]), float(gs[-1])) or
               best[2] in (float(ods[0]), float(ods[-1])))
    return dict(g_syn=best[1], out_drive=best[2], dprime=best[0],
                w_target=hw["w_target"], n_detectors=hw["n_detectors"],
                # optimum na ivici znaci da rjesetka nije obuhvatila maksimum;
                # izvjestaj to prijavljuje umjesto da tiho uzme rubnu tacku
                optimum_on_grid_edge=bool(on_edge),
                g_range=[float(gs[0]), float(gs[-1])],
                out_drive_range=[float(ods[0]), float(ods[-1])],
                curve=curve, eval=best[3])


# ==================================================================
# Sanity check: pik naucenog kasnjenja
# ==================================================================
def delay_scan(net: Network, lags=None, n_trials: int = 120,
               step_ms: int = 10, lo_ms: int = -60, hi_ms: int = 140) -> dict:
    """
    Odgovor OUT populacije kao funkcija razmaka A-B, sa zamrznutom
    plasticnoscu.  Ocekuje se pik oko 20 ms.

    Ovo je SANITY CHECK, ne glavno mjerenje -- pik postavljaju tau_m i
    raspodjela kasnjenja u supstratu, a ne duzina eligibility traga (v3 §1).

    OPSEG MORA BITI SIRI OD SIRINE PODESENOSTI.  Prva verzija je skenirala
    0-40 ms i dobila potpuno ravnu krivu (34, 29, 31, 31, 31, 34) -- ne zato
    sto tuninga nema, nego zato sto je cijeli opseg lezao UNUTAR njega:
    prozor koincidencije je ~+-40 ms (tau_m = 20 ms plus trajanje burst-a od
    20 ms).  U NOISE klasi je |t_B - t_A - 20| tipicno ~130 ms, dakle daleko
    izvan tog opsega -- zato d' moze biti visok a uzak scan ravan.
    Odatle podrazumijevani opseg -60..+140 ms.

    Kontrola sa zamijenjenim identitetom kanala ide uz svako mjerenje: ako se
    pik ne pomjeri simetricno, mjeri se nesto drugo (v3 §7.3).
    """
    p = net.p
    if lags is None:
        lags = list(range(lo_ms, hi_ms + 1, step_ms))
    saved_w = net.w.copy()
    saved_g = net.g_syn

    def _one(lag, swap):
        net.reset()
        net.w[:] = saved_w
        net.g_syn = saved_g
        rng = make_rng(int(net.seeds[0]), "delayscan_%d_%d" % (int(lag), int(swap)))
        tot = np.zeros(net.B)
        done = 0
        while done < n_trials:
            m = min(24, n_trials - done)
            # t_a je centriran (ne U(0,400)) da bi i negativna kasnjenja stala
            # u trial bez odsijecanja na 0
            t_a = rng.uniform(120.0, 300.0, m)
            t_b = t_a + lag + rng.normal(0.0, p.jitter, m)
            if swap:
                t_a, t_b = t_b, t_a
            R = build_raster(rng, p, t_a, t_b)
            R = np.broadcast_to(R, (net.B,) + R.shape)
            for j in range(m):
                net.n_out[:] = 0
                for tp in range(p.trial_len):
                    net.step(R[:, j, tp, :], tp, False, None, False)
                tot += net.n_out
            done += m
        return tot / n_trials

    resp = np.array([_one(l, False) for l in lags])          # (n_lags, B)
    resp_sw = np.array([_one(l, True) for l in lags])
    net.reset()
    net.w[:] = saved_w
    net.g_syn = saved_g

    mean = resp.mean(axis=1)
    peak = float(lags[int(np.argmax(mean))]) if mean.size else float("nan")
    # sirina na pola visine iznad baze -- govori koliko je detektor ostar
    base, top = float(mean.min()), float(mean.max())
    half = base + 0.5 * (top - base)
    above = [l for l, v in zip(lags, mean) if v >= half]
    width = float(max(above) - min(above)) if len(above) > 1 else 0.0
    return dict(lags=list(map(float, lags)),
                response=mean.tolist(),
                response_swapped=resp_sw.mean(axis=1).tolist(),
                peak_lag_ms=peak, fwhm_ms=width,
                modulation=float((top - base) / top) if top > 0 else 0.0,
                peak_lag_swapped_ms=float(lags[int(np.argmax(resp_sw.mean(axis=1)))]))


# ==================================================================
# Empirijski STDP prozor  (v3 §7.3)
# ==================================================================
def stdp_window(p: Params, seed: int = 1, dt_range: int = 25) -> dict:
    """
    Izmjeri stvarni STDP prozor kroz PRAVI kod mreze.

    Sve tezine na nulu (nema spontane aktivnosti), pa se forsira jedan
    pre-spike i jedan post-spike na kontrolisanoj razlici, i mjeri promjena
    eligibility-ja na jednoj sinapsi.

    Ovo je jedina provjera koja hvata gresku u knjigovodstvu tragova: ako
    namjeravani i izmjereni prozor ne izgledaju isto, svi rezultati do te
    tacke su nevazeci (v3 §11).  Konkretno hvata i konvenciju na Dt = 0:

        DT0_MODE = 'LTP'   ->   Dt = 0  daje  +A_plus     (opcija B, izabrana)
        DT0_MODE = 'ZERO'  ->   Dt = 0  daje   0          (opcija A)
        greska iz v1       ->   Dt = 0  daje  -0.1        (opcija C)
    """
    net = Network(p.with_(plastic=False, homeostasis=False), [(seed, 20)],
                  run_name="stdp")
    net.w[:] = 0.0
    net.g_syn = 0.0

    # jedna sinapsa A -> H
    k = int(np.flatnonzero((net.pre[0] < C.N_A) &
                           (net.post[0] >= C.SL_H.start) &
                           (net.post[0] < C.SL_H.stop))[0])
    i_pre, i_post, d0 = int(net.pre[0, k]), int(net.post[0, k]), int(net.d[0, k])

    ext_zero = np.zeros((1, net.n_in), dtype=bool)
    out = {}
    for dtv in range(-dt_range, dt_range + 1):
        net.reset()
        net.g_syn = 0.0
        net.w[:] = 0.0
        t_pre = 60                        # dovoljno daleko od pocetka
        t_arr = t_pre + d0                # Dt se mjeri od DOLASKA, ne od slanja
        t_post = t_arr + dtv
        t_read = max(t_arr, t_post)       # tek kad su se OBA desila
        e0 = 0.0
        for t in range(200):
            f = np.zeros((1, C.N), dtype=bool)
            if t == t_pre:
                f[0, i_pre] = True
            if t == t_post:
                f[0, i_post] = True
            net.step(ext_zero, t % p.trial_len, False, None, False, force=f)
            if t == t_read:
                e0 = float(net.e[0, k])
        out[dtv] = e0

    ks = sorted(out)
    vals = [out[k_] for k_ in ks]
    at0 = out.get(0, 0.0)
    peak_i = int(np.argmax(np.abs(vals)))
    return dict(delta_t=ks, e=vals, e_at_zero=at0,
                peak_delta_t=ks[peak_i], peak_value=vals[peak_i],
                dt0_mode=p.dt0_mode,
                sign_at_zero=("LTP" if at0 > 1e-9 else
                              "LTD" if at0 < -1e-9 else "ZERO"))


def weight_report(net: Network) -> dict:
    return bimodality(net.w, net.p.w_max)
