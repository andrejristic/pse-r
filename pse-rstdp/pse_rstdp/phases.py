# -*- coding: utf-8 -*-
"""
Faze R0-R5.   v3 §10

Svaka faza ima GATE koji mora proci da bi sljedeca krenula, i ishod gate-a se
upisuje u JSONL zajedno s podacima na kojima je donesen (v3 §9).  Falsifikatori
se izvrsavaju PRIJE gledanja rezultata -- inace se citaju kao potvrda.

Lanac mjerenih velicina (v3 §5.5):
    R1  ->  g_syn, W_target
    R2  ->  A_minus
    R3+ ->  koriste oboje
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .config import (DESIGN_POINTS, DR_LADDER, GRID_TAU_E, GRID_WF, Q_LADDER,
                     VERTEX_NAMES, Params)
from .design import escalate, fit_scheffe, steepest_ascent_points
from .evaluate import (delay_scan, evaluate, handwire, stdp_window, sweep_gain,
                       weight_report)
from .net import Network
from .runner import Journal, Timer, chunk_specs
from .stats import auc_log, dr_star
from .task import run_r0


# ==================================================================
def _apply_measured(p: Params, jr: Journal) -> Params:
    kw = {}
    if "a_minus" in jr.measured:
        kw["a_minus"] = float(jr.measured["a_minus"])
    if "w_target" in jr.measured:
        kw["w_target"] = float(jr.measured["w_target"])
    return p.with_(**kw) if kw else p


def _net(p: Params, specs, jr: Journal) -> Network:
    net = Network(_apply_measured(p, jr), specs)
    if "g_syn" in jr.measured:
        net.g_syn = float(jr.measured["g_syn"])
    elif p.calibrate_gain:
        net.calibrate_gain()
    return net


# ==================================================================
# R0
# ==================================================================
def phase_r0(p: Params, jr: Journal, seeds, scale: float = 1.0) -> dict:
    n = max(400, int(4000 * scale))
    ok = True
    for seed in seeds:
        key = "r0_s%d" % seed
        if jr.done("r0", key):
            continue
        rec = run_r0(p, seed, n_trials=n)
        rec["key"] = key
        jr.write("r0", rec)
        ok = ok and rec["gate"]
    recs = jr.read("r0")
    gate = all(r["gate"] for r in recs) if recs else False
    return dict(phase="R0", gate=gate, n=len(recs),
                acc_rate=float(np.mean([r["acc_rate_detector"] for r in recs])),
                acc_marginal=float(np.mean([r["acc_marginal_detector"] for r in recs])),
                acc_oracle=float(np.mean([r["acc_coincidence_oracle"] for r in recs])))


# ==================================================================
# R1 -- supstrat, bez ucenja.  Proizvodi g_syn i W_target.
# ==================================================================
def phase_r1(p: Params, jr: Journal, seeds, scale: float = 1.0) -> dict:
    t = Timer()
    n_ev = max(80, int(300 * scale))
    n_gain = max(60, int(150 * scale))
    p1 = p.with_(plastic=False, homeostasis=False)
    specs = [(int(s), DR_LADDER[0]) for s in seeds]
    net = Network(p1, specs)

    # zajednicki sweep (g_syn, out_drive): dva sloja imaju razlicite radne
    # tacke i jedno globalno g_syn ih ne moze oba posluziti (v. evaluate.sweep_gain)
    n_g = 6 if scale >= 0.5 else 4
    n_od = 4 if scale >= 0.5 else 3
    sw = sweep_gain(net, g_lo=30.0, g_hi=300.0, n_g=n_g,
                    od_lo=4.0, od_hi=64.0, n_od=n_od, n_trials=n_gain)
    ev = evaluate(net, n_ev, "r1eval")
    scan = delay_scan(net, n_trials=max(30, int(100 * scale)),
                      step_ms=10 if scale >= 0.5 else 20)
    win = stdp_window(p)

    gate = ev["dprime_roc"] > 1.5
    rec = dict(key="r1", phase="R1", gate=bool(gate),
               n_detectors=sw["n_detectors"], w_target=sw["w_target"],
               g_syn=sw["g_syn"], out_drive=sw["out_drive"], gain_curve=sw["curve"],
               dprime_roc=ev["dprime_roc"], auc=ev["auc"],
               per_batch=ev["per_batch"],
               peak_lag_ms=scan["peak_lag_ms"], fwhm_ms=scan["fwhm_ms"],
               modulation=scan["modulation"],
               peak_lag_swapped_ms=scan["peak_lag_swapped_ms"],
               optimum_on_grid_edge=sw["optimum_on_grid_edge"],
               delay_scan=scan,
               stdp_window=win, secs=t.s())
    jr.reset_phase("r1")
    jr.write("r1", rec)
    # upisuje se i kad gate padne: sljedece faze ih trebaju da bi se uopste
    # mogle pokrenuti pod --no-gate.  Gate se i dalje prijavljuje kao pao.
    jr.set_measured(g_syn=sw["g_syn"], out_drive=sw["out_drive"],
                    w_target=sw["w_target"], r1_gate=bool(gate))
    return dict(phase="R1", gate=bool(gate), dprime=ev["dprime_roc"],
                g_syn=sw["g_syn"], out_drive=sw["out_drive"],
                w_target=sw["w_target"], n_detectors=sw["n_detectors"],
                peak_lag_ms=scan["peak_lag_ms"], fwhm_ms=scan["fwhm_ms"],
                modulation=scan["modulation"],
                optimum_on_grid_edge=sw["optimum_on_grid_edge"],
                stdp_dt0=win["sign_at_zero"], secs=t.s())


# ==================================================================
# R2 -- plasticnost + kalibracija A_minus + PR provjera
# ==================================================================
def calibrate_a_minus(p: Params, jr: Journal, seed: int, n_trials: int,
                      lo: float = 0.20, hi: float = 3.00,
                      iters: int = 12) -> dict:
    """
    v3 §6.9.  Zamrzni M na konstantu, pusti shuffle-ovan ulaz (razbijena A-B
    korelacija), mjeri srednji drift tezina, podesi A_minus dok ne bude ~0.

    Zatvorena formula je ODBACENA jer pretpostavlja RAVNU raspodjelu Dt, a
    stvarna je oblikovana stopama, refrakcijom, kasnjenjima i time sto post
    neuron ispaljuje ZBOG dolazaka.  Ovdje se mjeri sta mreza stvarno radi.
    """
    g = float(jr.measured["g_syn"])
    trace = []
    mid = 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        # Homeostaza ostaje UKLJUCENA: mjeri se raspodjela Dt koju mreza stvarno
        # ima u radnom rezimu.  Sa iskljucenom bi mreza otisla u zasicenje i
        # kalibrisali bismo drift koji se nikad ne desava.
        net = Network(p.with_(a_minus=mid), [(seed, DR_LADDER[0])])
        net.g_syn = g
        w0 = net.w.copy()                   # obicna inicijalizacija, ne 0.5
        net.run_trials(n_trials, "aminuscal", learn=True, shuffle=True,
                       mod_override=1.0)
        # samo sinapse koje NISU udarile u granicu (clip bi napravio laznu
        # asimetriju), i bez OUT sinapsi (njih mijenja i sinapticko skaliranje)
        free = (w0 > 0.02 * p.w_max) & (w0 < 0.98 * p.w_max) & \
               (net.w > 0.0) & (net.w < p.w_max) & (~net.is_out_syn)
        drift = float(np.mean((net.w - w0)[free])) if free.any() else 0.0
        trace.append(dict(a_minus=mid, drift=drift))
        if drift > 0.0:
            lo = mid                        # treba jaca depresija
        else:
            hi = mid
        if abs(drift) < 1e-7:
            break
    # Ako je rjesenje na ivici opsega, drift nije nulovan nego je bisekcija
    # istrcala.  To se prijavljuje, ne prescutkuje -- inace bi A_minus izgledao
    # kao izmjeren a bio bi granicna vrijednost.
    lo0, hi0 = 0.20, 3.00
    on_edge = mid <= lo0 * 1.02 or mid >= hi0 * 0.98
    return dict(a_minus=mid, trace=trace, final_drift=trace[-1]["drift"],
                on_range_edge=bool(on_edge), search_range=[lo0, hi0])


def phase_r2(p: Params, jr: Journal, seeds, scale: float = 1.0) -> dict:
    t = Timer()
    jr.need("g_syn", "R1")
    n_train = max(60, int(600 * scale))
    n_eval = max(80, int(400 * scale))
    n_cal = max(20, int(120 * scale))
    p2 = p.with_(w_s=1.0, w_o=0.0, w_f=0.0)      # cist R-STDP

    # ---- 1. kalibracija A_minus ----
    if "a_minus" not in jr.measured:
        cal = calibrate_a_minus(p2, jr, int(seeds[0]), n_cal)
        jr.write("r2", dict(key="r2_aminus", step="calibrate_a_minus", **cal))
        jr.set_measured(a_minus=cal["a_minus"])
    a_minus = float(jr.measured["a_minus"])

    # ---- 2. poredjenje konvencija na Dt = 0, SEED-UPARENO ----
    # Neuparen test je potceenjen snagom i lako daje lazno "ispod suma" (v3 §10 R2).
    conv = []
    for name, mode, am in (("A_1.1", "ZERO", 1.1),
                           ("B_1.1", "LTP", 1.1),
                           ("B_cal", "LTP", a_minus)):
        key = "r2_conv_%s" % name
        if jr.done("r2", key):
            continue
        pc = p2.with_(dt0_mode=mode, a_minus=am)
        net = _net(pc, [(int(s), DR_LADDER[0]) for s in seeds], jr)
        net.run_trials(n_train, "r2conv", learn=True)
        ev = evaluate(net, n_eval, "r2conveval")
        rec = dict(key=key, step="convention", name=name, dt0_mode=mode,
                   a_minus=am, dprime_roc=ev["dprime_roc"], auc=ev["auc"],
                   pr_out=ev["pr_out"], weights=weight_report(net),
                   per_batch=ev["per_batch"])
        jr.write("r2", rec)
    conv = [r for r in jr.read("r2") if r.get("step") == "convention"]

    # ---- 3. PR provjera: vektorski vs skalarni prag ----
    # Ako skalar da PR ~ 1 uz JEDNAK d', tvrdnja iz v3 §6.10 pada -- po
    # podacima, ne po raspravi.  Falsifikator je zapisan u v3 §11.
    for vec in (True, False):
        key = "r2_pr_%s" % ("vector" if vec else "scalar")
        if jr.done("r2", key):
            continue
        net = _net(p2.with_(theta_vector=vec, a_minus=a_minus),
                   [(int(s), DR_LADDER[0]) for s in seeds], jr)
        net.run_trials(n_train, "r2pr", learn=True)
        ev = evaluate(net, n_eval, "r2preval")
        jr.write("r2", dict(key=key, step="pr", theta_vector=vec,
                            dprime_roc=ev["dprime_roc"], pr_out=ev["pr_out"],
                            rate_out_hz=ev["rate_out_hz"],
                            per_batch=ev["per_batch"]))
    pr = {r["theta_vector"]: r for r in jr.read("r2") if r.get("step") == "pr"}

    # ---- 4. glavni R2 run ----
    key = "r2_main"
    if not jr.done("r2", key):
        net = _net(p2.with_(a_minus=a_minus),
                   [(int(s), DR_LADDER[0]) for s in seeds], jr)
        w_before = weight_report(net)
        net.run_trials(n_train, "r2main", learn=True)
        ev = evaluate(net, n_eval, "r2maineval")
        scan = delay_scan(net, n_trials=max(30, int(100 * scale)),
                          step_ms=10 if scale >= 0.5 else 20)
        w_after = weight_report(net)
        gate = (ev["dprime_roc"] > 1.0 and w_after["frac_at_max"] < 0.5)
        jr.write("r2", dict(key=key, step="main", gate=bool(gate),
                            dprime_roc=ev["dprime_roc"], auc=ev["auc"],
                            pr_out=ev["pr_out"], peak_lag_ms=scan["peak_lag_ms"],
                            weights_before=w_before, weights_after=w_after,
                            per_batch=ev["per_batch"], secs=t.s()))
    main = [r for r in jr.read("r2") if r.get("step") == "main"][-1]

    pr_verdict = None
    if True in pr and False in pr:
        d_vec, d_sca = pr[True]["dprime_roc"], pr[False]["dprime_roc"]
        pr_sca = pr[False]["pr_out"]
        pr_verdict = ("skalar dovoljan (tvrdnja o vektoru PADA)"
                      if pr_sca < 1.5 and d_sca >= d_vec - 0.15
                      else "vektor opravdan")

    return dict(phase="R2", gate=bool(main["gate"]), a_minus=a_minus,
                dprime=main["dprime_roc"], pr_out=main["pr_out"],
                peak_lag_ms=main["peak_lag_ms"],
                frac_at_max=main["weights_after"]["frac_at_max"],
                conventions={c["name"]: c["dprime_roc"] for c in conv},
                pr_vector=(pr[True]["pr_out"] if True in pr else None),
                pr_scalar=(pr[False]["pr_out"] if False in pr else None),
                pr_verdict=pr_verdict, secs=t.s())


# ==================================================================
# Zajednicki: jedna tacka dizajna preko cijele Dr ljestvice
# ==================================================================
def _equalize_eta(p: Params, jr: Journal, seed: int, n_probe: int) -> float:
    """
    v3 §6.7 (2): eta se reskalira po tacki dizajna tako da eta_eff * E[|M|]
    bude konstantno kroz cijeli dizajn.  Bez toga mijenjanje c mijenja
    EFEKTIVNU STOPU UCENJA, pa se pomjeranje odziva moze pripisati brizi za
    buducnost a zapravo je manji eta.  Najzilaviji konfund.
    """
    net = _net(p.with_(plastic=False), [(seed, DR_LADDER[0])], jr)
    r = net.run_trials(n_probe, "etaprobe", learn=False)
    m = float(r["mean_abs_M"])
    return 1.0 / max(m, 1e-6)


def run_design_point(p: Params, jr: Journal, phase: str, name: str,
                     c: tuple, seeds, scale: float, ladder=DR_LADDER,
                     extra: dict | None = None) -> dict:
    """Trenira i evaluira jednu tacku dizajna preko cijele Dr ljestvice."""
    n_train = max(60, int(600 * scale))
    n_eval = max(80, int(400 * scale))
    n_probe = max(10, int(40 * scale))
    w_s, w_o, w_f = c
    base = p.with_(w_s=w_s, w_o=w_o, w_f=w_f)
    scl = _equalize_eta(base, jr, int(seeds[0]), n_probe)
    base = base.with_(eta_eff_scale=scl)

    specs = [(int(s), int(dr)) for s in seeds for dr in ladder]
    batch = jr.profile.batch if jr.profile else 16
    rows, weights = [], []
    for group in chunk_specs(specs, batch):
        net = _net(base, group, jr)
        net.run_trials(n_train, "train_%s" % name, learn=True)
        ev = evaluate(net, n_eval, "eval_%s" % name)
        weights.append(weight_report(net))
        for i, (seed, dr) in enumerate(group):
            b = ev["per_batch"][i]
            rows.append(dict(seed=seed, delta_r=dr, **b))

    out = []
    for seed in seeds:
        sub = sorted([r for r in rows if r["seed"] == seed],
                     key=lambda r: r["delta_r"])
        drs = [r["delta_r"] for r in sub]
        dps = [r["dprime_roc"] for r in sub]
        star = dr_star(drs, dps)
        out.append(dict(seed=int(seed), delta_r=drs, dprime=dps,
                        auc_log=auc_log(drs, dps), **star))

    rec = dict(key="%s_%s" % (phase, name), phase=phase.upper(), point=name,
               c=[w_s, w_o, w_f], eta_eff_scale=scl,
               tau_e=base.tau_e, lam=base.lam,
               response=float(np.mean([o["auc_log"] for o in out])),
               response_sd=float(np.std([o["auc_log"] for o in out])),
               per_seed=out, rows=rows, weights=weights,
               frac_at_max=float(np.mean([w["frac_at_max"] for w in weights])))
    if extra:
        rec.update(extra)
    return rec


# ==================================================================
# R3 -- mikser: vrhovi kao gate + ablacija kriticara
# ==================================================================
def phase_r3(p: Params, jr: Journal, seeds, scale: float = 1.0) -> dict:
    t = Timer()
    jr.need("g_syn", "R1")
    jr.need("a_minus", "R2")
    p3 = _apply_measured(p, jr)

    for name, c in DESIGN_POINTS:
        if name not in VERTEX_NAMES:
            continue
        key = "r3_%s" % name
        if jr.done("r3", key):
            continue
        rec = run_design_point(p3, jr, "r3", name, c, seeds, scale)
        rec["key"] = key
        jr.write("r3", rec)

    verts = {r["point"]: r for r in jr.read("r3") if r["point"] in VERTEX_NAMES}
    vals = [verts[v]["response"] for v in VERTEX_NAMES if v in verts]
    sds = [verts[v]["response_sd"] for v in VERTEX_NAMES if v in verts]
    spread = float(max(vals) - min(vals)) if vals else 0.0
    noise = float(np.mean(sds)) if sds else 1.0
    # GATE: bar dva vrha se razlikuju iznad varijanse po sjemenu.  Ako se sva
    # tri ponasaju isto, mikser je suvisan -- v3 §11, prvi falsifikator.
    gate = spread > max(noise, 1e-9)

    # --- ablacija kriticara bez faznih RBF obiljezja (v3 §6.6) ---
    if not jr.done("r3", "r3_ablation_norbf"):
        rec = run_design_point(p3.with_(use_rbf=False), jr, "r3",
                               "vertex_f_norbf", (0.0, 0.0, 1.0), seeds, scale)
        rec["key"] = "r3_ablation_norbf"
        jr.write("r3", rec)
    abl = [r for r in jr.read("r3") if r.get("point") == "vertex_f_norbf"]

    res = dict(phase="R3", gate=bool(gate), spread=spread, seed_noise=noise,
               vertices={v: verts[v]["response"] for v in verts}, secs=t.s())
    if abl and "vertex_f" in verts:
        res["ablation_norbf"] = abl[-1]["response"]
        res["vertex_f_full"] = verts["vertex_f"]["response"]
        res["ablation_drop"] = verts["vertex_f"]["response"] - abl[-1]["response"]
    return res


# ==================================================================
# R4 -- povrsina i grid
# ==================================================================
def phase_r4(p: Params, jr: Journal, seeds, scale: float = 1.0,
             do_grid: bool = True, do_ascent: bool = True) -> dict:
    # najbrzi uspon je sekvencijalna faza -- na maloj skali se preskace,
    # jer bi fit iz kojeg izvodi pravac ionako bio bez snage
    do_ascent = do_ascent and scale >= 0.3
    t = Timer()
    jr.need("g_syn", "R1")
    jr.need("a_minus", "R2")
    p4 = _apply_measured(p, jr)

    # ---- 1. augmentirani simplex-centroid, 10 tacaka ----
    for name, c in DESIGN_POINTS:
        key = "r4_%s" % name
        if jr.done("r4", key):
            continue
        prev = [r for r in jr.read("r3") if r.get("point") == name]
        if prev:                                    # vrhovi su vec odvrceni u R3
            rec = dict(prev[-1])
            rec["key"] = key
            rec["phase"] = "R4"
            rec["reused_from"] = "R3"
        else:
            rec = run_design_point(p4, jr, "r4", name, c, seeds, scale)
            rec["key"] = key
        jr.write("r4", rec)

    pts = {r["point"]: r for r in jr.read("r4") if r.get("point")}
    # per-seed vrijednosti, ne srednje -- lack-of-fit trazi PURE ERROR iz
    # replikata, a on postoji samo ako sjemena udju pojedinacno (v3 §8.3)
    design = [(tuple(pts[n]["c"]), [o["auc_log"] for o in pts[n]["per_seed"]])
              for n, _ in DESIGN_POINTS if n in pts]
    fit = escalate(design)

    # ---- 2. najbrzi uspon ako je optimum na ivici ----
    ascent = None
    if do_ascent and fit.get("optimum_on_edge"):
        for i, c in enumerate(steepest_ascent_points(fit, n=6)):
            key = "r4_ascent_%d" % i
            if jr.done("r4", key):
                continue
            rec = run_design_point(p4, jr, "r4", "ascent_%d" % i, c, seeds, scale)
            rec["key"] = key
            jr.write("r4", rec)
        ascent = [r for r in jr.read("r4") if str(r.get("point", "")).startswith("ascent")]

    # ---- 3. grid w_f x tau_e na tri najinformativnije c tacke ----
    grid = []
    if do_grid:
        n_base = 3 if scale >= 0.3 else 1
        # samo tacke osnovnog dizajna -- inace bi se pri nastavljanju grid
        # tacke rangirale i generisale ugnijezdena imena tipa grid_grid_...
        names = {n for n, _ in DESIGN_POINTS}
        base_pts = [r for k, r in pts.items() if k in names]
        ranked = sorted(base_pts, key=lambda r: -r["response"])[:n_base]
        for r in ranked:
            for wf in GRID_WF:
                for te in GRID_TAU_E:
                    nm = "grid_%s_wf%.1f_te%.0f" % (r["point"], wf, te)
                    key = "r4_" + nm
                    if jr.done("r4", key):
                        continue
                    ws, wo, _ = r["c"]
                    tot = ws + wo
                    # ostatak mase se dijeli u istom odnosu w_s : w_o kao u
                    # baznoj tacki; ako je bazna tacka cist w_f, ostatak ide na
                    # w_s (primarni kanal), ne na peer
                    c = ((ws / tot) * (1 - wf), (wo / tot) * (1 - wf), wf) \
                        if tot > 0 else (1 - wf, 0.0, wf)
                    rec = run_design_point(
                        p4.with_(tau_e_override=te), jr, "r4", nm, c, seeds, scale,
                        extra=dict(grid_wf=wf, grid_tau_e=te, base_point=r["point"]))
                    rec["key"] = key
                    jr.write("r4", rec)
        grid = [r for r in jr.read("r4") if r.get("grid_wf") is not None]

    grid_verdict = None
    if grid:
        gw = np.array([r["grid_wf"] for r in grid])
        gt = np.array([np.log(r["grid_tau_e"]) for r in grid])
        gy = np.array([r["response"] for r in grid])
        X = np.column_stack([np.ones_like(gw), gw, gt, gw * gt])
        try:
            beta, *_ = np.linalg.lstsq(X, gy, rcond=None)
            grid_verdict = dict(b_wf=float(beta[1]), b_tau_e=float(beta[2]),
                                b_interaction=float(beta[3]))
        except np.linalg.LinAlgError:
            pass

    gate = bool(fit.get("r2", 0.0) >= 0.7 and not fit.get("lack_of_fit_significant", True))
    return dict(phase="R4", gate=gate, fit=fit,
                n_points=len(pts), n_ascent=(len(ascent) if ascent else 0),
                n_grid=len(grid), grid_verdict=grid_verdict, secs=t.s())


# ==================================================================
# R5 -- druga tvrdnja: optimalno w_o raste s q?
# ==================================================================
def phase_r5(p: Params, jr: Journal, seeds, scale: float = 1.0) -> dict:
    t = Timer()
    jr.need("g_syn", "R1")
    jr.need("a_minus", "R2")
    p5 = _apply_measured(p, jr)
    w_o_grid = (0.0, 0.25, 0.5, 0.75) if scale >= 0.3 else (0.0, 0.5)

    for q in Q_LADDER:
        for wo in w_o_grid:
            nm = "q%.1f_wo%.2f" % (q, wo)
            key = "r5_" + nm
            if jr.done("r5", key):
                continue
            c = (1.0 - wo, wo, 0.0)
            rec = run_design_point(p5.with_(q=q), jr, "r5", nm, c, seeds, scale,
                                   extra=dict(q=q, w_o=wo))
            rec["key"] = key
            jr.write("r5", rec)

    rows = [r for r in jr.read("r5") if r.get("q") is not None]
    best = {}
    for q in Q_LADDER:
        sub = [r for r in rows if abs(r["q"] - q) < 1e-9]
        if sub:
            b = max(sub, key=lambda r: r["response"])
            best[q] = dict(w_o=b["w_o"], response=b["response"])
    qs = sorted(best)
    monotone = all(best[qs[i]]["w_o"] <= best[qs[i + 1]]["w_o"]
                   for i in range(len(qs) - 1)) if len(qs) > 1 else False
    return dict(phase="R5", gate=bool(monotone), best_w_o_by_q=best,
                monotone=bool(monotone), secs=t.s())


PHASES = {"r0": phase_r0, "r1": phase_r1, "r2": phase_r2,
          "r3": phase_r3, "r4": phase_r4, "r5": phase_r5}
