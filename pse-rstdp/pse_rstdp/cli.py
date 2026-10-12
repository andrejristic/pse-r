# -*- coding: utf-8 -*-
"""Komandna linija.   pse-rstdp <komanda>"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .analysis import report
from .config import DR_LADDER, DR_LADDER_SHORT, PROFILES, Params, get_profile
from .net import Network
from .phases import PHASES
from .rng import state_hash
from .runner import Journal

DEFAULT_OUT = os.path.expanduser("~/pse-rstdp-results")
PHASE_ORDER = ("r0", "r1", "r2", "r3", "r4", "r5")

# Hash kratke deterministicke simulacije.  Isti seed -> ista vrijednost na
# svakoj masini (v3 §12.1).  Ako se razlikuje, rezultati s dvije masine se NE
# smiju spajati -- masina bi postala konfund u dizajnu koji je cijeli o
# kontrastima.  Referenca se generise sa `selftest --update`.
REF_PATH = os.path.join(os.path.dirname(__file__), "selftest_ref.json")


# ------------------------------------------------------------------
def _params_from_args(a) -> Params:
    kw = {}
    if getattr(a, "steps", None):
        kw["T"] = int(a.steps)
    if getattr(a, "dt0", None):
        kw["dt0_mode"] = a.dt0
    return Params(**kw)


def _seeds(a):
    return [int(s) for s in str(a.seeds).split(",") if s.strip()]


def _ladder(a):
    return DR_LADDER_SHORT if getattr(a, "short_ladder", False) else DR_LADDER


# ------------------------------------------------------------------
def cmd_selftest(a) -> int:
    p = Params(plastic=True, homeostasis=True, w_s=1.0, w_o=0.0, w_f=0.0)
    net = Network(p, [(1, 20), (2, 80)])
    net.g_syn = 100.0
    net.run_trials(4, "selftest", learn=True)
    h = state_hash(w=net.w, V=net.V, theta=net.theta, rho=net.rho,
                   e=net.e, x=net.x, y=net.y)
    print("hash stanja : %s" % h)
    print("numpy       : %s" % np.__version__)
    print("paket       : %s" % __version__)

    if getattr(a, "update", False):
        with open(REF_PATH, "w") as f:
            json.dump(dict(hash=h, numpy=np.__version__,
                           version=__version__), f, indent=2)
        print("referenca upisana -> %s" % REF_PATH)
        return 0
    try:
        with open(REF_PATH) as f:
            ref = json.load(f)
    except (OSError, ValueError):
        print("\nNema reference.  Pokreni `pse-rstdp selftest --update` na masini")
        print("koju uzimas kao referentnu, pa isti fajl ponesi na drugu.")
        return 0
    if ref.get("hash") == h:
        print("\nOK -- bit-identicno s referencom (numpy %s)." % ref.get("numpy"))
        return 0
    print("\nRAZLIKA od reference!")
    print("  referenca: %s  (numpy %s)" % (ref.get("hash"), ref.get("numpy")))
    print("  ovdje    : %s  (numpy %s)" % (h, np.__version__))
    print("\nRezultati s ove i referentne masine se NE smiju spajati u isti")
    print("dizajn -- masina bi postala konfund (v3 §12.1).")
    return 1


def cmd_bench(a) -> int:
    """v3 §8.5: ne procjenjuj vrijeme, izmjeri ga."""
    prof = get_profile(a.profile)
    p = Params()
    specs = [(1 + i // len(DR_LADDER), DR_LADDER[i % len(DR_LADDER)])
             for i in range(prof.batch)]
    net = Network(p, specs)
    net.g_syn = 100.0
    n_tr = max(2, int(a.trials))
    t0 = time.time()
    net.run_trials(n_tr, "bench", learn=True)
    dt = time.time() - t0
    steps = n_tr * p.trial_len
    us = dt / steps * 1e6
    per_run = us * p.T / 1e6
    print("profil %s: batch=%d, jobs=%d" % (prof.name, prof.batch, prof.jobs))
    print("izmjereno : %.0f us/korak  (%d koraka za %.1f s)" % (us, steps, dt))
    print("jedan batch od %d runova x %d koraka: %.1f min"
          % (prof.batch, p.T, per_run / 60.0))
    for name, n in (("R3 (3 vrha)", 3), ("R4 faza A (10 tacaka)", 10),
                    ("R4 grid (27)", 27), ("R5 (12)", 12)):
        n_runs = n * len(DR_LADDER) * len(_seeds(a))
        n_batch = int(np.ceil(n_runs / prof.batch))
        print("  %-22s %3d runova = %2d batcheva = %5.1f h"
              % (name, n_runs, n_batch, n_batch * per_run / 3600.0))
    print("\nAko je ukupno vise od jedne noci, skrati Dr ljestvicu na 3 tacke")
    print("(--short-ladder) PRIJE nego smanjujes broj sjemena -- sjemena trebaju")
    print("za pure error u lack-of-fit testu (v3 §8.3).")
    return 0


def cmd_run(a) -> int:
    prof = get_profile(a.profile)
    p = _params_from_args(a)
    jr = Journal(a.out, prof)
    jr.write_meta(p, extra=dict(seeds=_seeds(a), scale=a.scale,
                                ladder=list(_ladder(a))))
    phases = PHASE_ORDER if a.phase == "all" else [a.phase]
    if a.reset:
        for ph in phases:
            jr.reset_phase(ph)
    seeds = _seeds(a)
    rc = 0
    for ph in phases:
        print("\n" + "=" * 66)
        print("  FAZA %s" % ph.upper())
        print("=" * 66)
        t0 = time.time()
        try:
            res = PHASES[ph](p, jr, seeds, scale=a.scale)
        except SystemExit as exc:
            print("  ZAUSTAVLJENO: %s" % exc)
            return 2
        for k, v in res.items():
            if k in ("phase",):
                continue
            print("  %-22s %s" % (k, _fmt(v)))
        print("  %-22s %.1f s" % ("trajanje", time.time() - t0))
        if not res.get("gate", True):
            if a.no_gate:
                print("\n  GATE NIJE PROSAO -- nastavljam jer je --no-gate.")
                print("  Rezultati NISU naucno upotrebljivi; ovo je samo provjera koda.")
                rc = 3
            else:
                print("\n  GATE NIJE PROSAO.  Sljedeca faza se ne pokrece.")
                print("  Ovo nije kvar programa -- to je falsifikator koji je opalio.")
                print("  Vidi `pse-rstdp report --out %s`." % a.out)
                rc = 3
                break
    print("\nrezultati: %s" % os.path.abspath(a.out))
    return rc


def _fmt(v):
    if isinstance(v, float):
        return "%.4g" % v
    if isinstance(v, dict):
        return json.dumps(v, default=str)[:300]
    if isinstance(v, list) and len(v) > 8:
        return "[%d stavki]" % len(v)
    return str(v)[:300]


def cmd_report(a) -> int:
    return report(a.out, plots=a.plots)


def cmd_info(a) -> int:
    p = Params()
    print("PSE-R-STDP %s   (spec: nacrt v3)" % __version__)
    print("\nIzvedene velicine:")
    for k in ("gamma", "tau_e", "lam", "n_features"):
        print("  %-12s %s" % (k, getattr(p, k)))
    print("\nMreza: N=%d, sinapsi=%d, stanje modela < 1 MB" % (110, 1680))
    print("\nProfili:")
    for nm, pr in PROFILES.items():
        print("  %-8s jobs=%d batch=%d checkpoint=%d"
              % (nm, pr.jobs, pr.batch, pr.checkpoint_every))
    print("\nMJERENE velicine (proizvode ih faze, ne unose se):")
    print("  g_syn, out_drive, W_target  <- R1")
    print("  A_minus                     <- R2 kalibracijski run")
    print("\nRedoslijed faza: " + " -> ".join(x.upper() for x in PHASE_ORDER))
    return 0


# ------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="pse-rstdp",
        description="R-STDP koincidencijski detektor s Care Invariantom "
                    "kao modulatorom (nacrt v3).")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", default=DEFAULT_OUT, help="izlazni direktorij")
        sp.add_argument("--profile", default="surface", choices=sorted(PROFILES))
        sp.add_argument("--seeds", default="1,2", help="npr. 1,2,3")

    s = sub.add_parser("run", help="pokreni fazu (ili sve)")
    common(s)
    s.add_argument("--phase", default="all",
                   choices=list(PHASE_ORDER) + ["all"])
    s.add_argument("--scale", type=float, default=1.0,
                   help="skalira duzinu runova; 0.05 = brza provjera cijelog lanca")
    s.add_argument("--steps", type=int, default=None, help="pregazi T")
    s.add_argument("--dt0", choices=("LTP", "ZERO"), default=None)
    s.add_argument("--short-ladder", action="store_true")
    s.add_argument("--reset", action="store_true", help="obrisi te faze i kreni ispocetka")
    s.add_argument("--no-gate", action="store_true",
                   help="nastavi i kad gate padne -- SAMO za provjeru koda uz --scale")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("bench", help="izmjeri brzinu i procijeni budzet")
    common(s)
    s.add_argument("--trials", type=int, default=4)
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("selftest", help="provjeri determinizam izmedju masina")
    s.add_argument("--update", action="store_true", help="upisi novu referencu")
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("report", help="izvjestaj iz rezultata")
    s.add_argument("--out", default=DEFAULT_OUT)
    s.add_argument("--plots", action="store_true", help="i grafike (treba matplotlib)")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("info", help="parametri i porijeklo")
    s.set_defaults(func=cmd_info)

    a = ap.parse_args(argv)
    return int(a.func(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
