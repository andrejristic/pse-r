# -*- coding: utf-8 -*-
"""
Izvjestaj iz rezultata.   v3 §7, §8, §11

Poredak izlaza prati logiku nacrta: prvo da li su gate-ovi prosli, pa mjerenja,
pa falsifikatori.  Falsifikatori se ispisuju IZRICITO -- ako je neki opalio,
to je rezultat, ne kvar.

matplotlib je opcion (v3 §12.4): bez njega se grafike preskacu, brojevi ostaju.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .design import escalate
from .runner import Journal


def _hr(t=""):
    print("\n" + "-" * 68)
    if t:
        print(t)
        print("-" * 68)


def report(outdir: str, plots: bool = False) -> int:
    jr = Journal(outdir)
    meta_path = os.path.join(jr.dir, "meta.json")
    if not os.path.exists(meta_path):
        print("nema rezultata u %s" % jr.dir)
        return 1
    with open(meta_path) as f:
        meta = json.load(f)

    print("=" * 68)
    print("PSE-R-STDP  --  izvjestaj")
    print("=" * 68)
    print("direktorij : %s" % jr.dir)
    print("napravljeno: %s" % meta.get("created"))
    print("masina     : %s / numpy %s" % (meta.get("platform"), meta.get("numpy")))
    print("profil     : %s (batch %s)" % (meta.get("profile"), meta.get("profile_batch")))
    print("sjemena    : %s   skala: %s" % (meta.get("seeds"), meta.get("scale")))

    _hr("MJERENE VELICINE  (proizvele ih faze, nisu unesene)")
    if jr.measured:
        for k, v in sorted(jr.measured.items()):
            print("  %-12s %s" % (k, v))
    else:
        print("  (jos nijedna)")

    falsified = []

    # ---------------- R0 ----------------
    r0 = jr.read("r0")
    if r0:
        _hr("R0  --  validnost zadatka")
        print("  detektor stope        %.3f   (mora biti ~0.50)"
              % np.mean([r["acc_rate_detector"] for r in r0]))
        print("  marginalni detektor   %.3f   (mora biti ~0.50)"
              % np.mean([r["acc_marginal_detector"] for r in r0]))
        print("  koincidencijski orakl %.3f   (mora biti > 0.80)"
              % np.mean([r["acc_coincidence_oracle"] for r in r0]))
        w = max(r0, key=lambda r: r["worst_cohen_d"])
        print("  najveca razlika marginala: %s  Cohen d = %.4f"
              % (w["worst_marginal"], w["worst_cohen_d"]))
        print("  GATE: %s" % ("PROSAO" if all(r["gate"] for r in r0) else "PAO"))

    # ---------------- R1 ----------------
    r1 = jr.read("r1")
    if r1:
        r = r1[-1]
        _hr("R1  --  supstrat, bez ucenja")
        print("  detektora po mrezi    %s" % r["n_detectors"])
        print("  g_syn                 %.1f" % r["g_syn"])
        print("  W_target              %.3f" % r["w_target"])
        print("  d' (ROC)              %.2f   (gate: > 1.5)" % r["dprime_roc"])
        if r.get("optimum_on_grid_edge"):
            print("  UPOZORENJE: optimum (g_syn, out_drive) je na IVICI rjesetke")
            print("    -> rjesetka nije obuhvatila maksimum; prosiri raspon")
        print("  pik naucenog kasnjenja %.1f ms  (ocekivano ~20)" % r["peak_lag_ms"])
        print("  pik sa zamijenjenim kanalima %.1f ms" % r["peak_lag_swapped_ms"])
        if r.get("fwhm_ms") is not None:
            print("  sirina podesenosti     %.0f ms  modulacija %.2f"
                  % (r["fwhm_ms"], r.get("modulation", 0.0)))
            if r.get("modulation", 0.0) < 0.15:
                falsified.append(
                    "Kriva odgovora nad razmakom A-B je RAVNA (modulacija %.2f) -- "
                    "mreza nije podesena na kasnjenje, pa d' dolazi odnekud "
                    "drugdje.  Provjeri da opseg skeniranja nije uzi od sirine "
                    "podesenosti prije nego se ovo prihvati kao nalaz."
                    % r.get("modulation", 0.0))
        w = r.get("stdp_window", {})
        print("  empirijski STDP prozor: Dt=0 -> %s, pik na Dt=%+d"
              % (w.get("sign_at_zero"), w.get("peak_delta_t", 0)))
        exp = "LTP" if w.get("dt0_mode") == "LTP" else "ZERO"
        if w and w.get("sign_at_zero") != exp:
            falsified.append(
                "Empirijski STDP prozor NE odgovara namjeravanom (Dt=0 daje %s, "
                "ocekivano %s).  Greska je u knjigovodstvu tragova -- svi "
                "rezultati do te tacke su nevazeci (v3 §11)."
                % (w.get("sign_at_zero"), exp))
        print("  GATE: %s" % ("PROSAO" if r["gate"] else "PAO"))
        if not r["gate"]:
            falsified.append(
                "Rucno ozicena mreza NE razdvaja SIGNAL od NOISE (d' = %.2f).  "
                "Problem je u SUPSTRATU, ne u pravilu ucenja -- nema smisla "
                "debugovati R-STDP (v3 §10 R1)." % r["dprime_roc"])

    # ---------------- R2 ----------------
    r2 = jr.read("r2")
    if r2:
        _hr("R2  --  plasticnost")
        cal = [r for r in r2 if r.get("step") == "calibrate_a_minus"]
        if cal:
            c = cal[-1]
            print("  A_minus (izmjeren)    %.4f" % c["a_minus"])
            print("    drift na kraju      %+.3e" % c["trace"][-1]["drift"])
            if c.get("on_range_edge"):
                print("    UPOZORENJE: bisekcija je istrcala na ivicu opsega %s"
                      % c.get("search_range"))
                falsified.append(
                    "Kalibracija A_minus nije nulovala drift nego je istrcala na "
                    "ivicu opsega (A_minus = %.3f, drift = %+.2e).  Vrijednost "
                    "NIJE izmjerena nego granicna -- prosiri opseg prije nego se "
                    "koristi." % (c["a_minus"], c["trace"][-1]["drift"]))
        conv = [r for r in r2 if r.get("step") == "convention"]
        if conv:
            print("  konvencija na Dt = 0  (seed-upareno):")
            for c in conv:
                print("    %-8s dt0=%-5s A_minus=%.3f  d'=%+.2f  frac@w_max=%.3f"
                      % (c["name"], c["dt0_mode"], c["a_minus"], c["dprime_roc"],
                         c["weights"]["frac_at_max"]))
        pr = {r["theta_vector"]: r for r in r2 if r.get("step") == "pr"}
        if True in pr and False in pr:
            print("  participation ratio (v3 §6.10 -- tvrdnja o vektorskom theta):")
            print("    vektorski prag  PR=%.2f  d'=%+.2f"
                  % (pr[True]["pr_out"], pr[True]["dprime_roc"]))
            print("    skalarni prag   PR=%.2f  d'=%+.2f"
                  % (pr[False]["pr_out"], pr[False]["dprime_roc"]))
            if pr[True]["pr_out"] > 8.0 and pr[False]["pr_out"] > 8.0:
                print("    (oba blizu maksimuma 10 -- ako je run kratak, mreza se")
                print("     jos nije diferencirala i test nema snagu)")
            if pr[False]["pr_out"] < 1.5 and \
                    pr[False]["dprime_roc"] >= pr[True]["dprime_roc"] - 0.15:
                falsified.append(
                    "PR ~ 1 kod skalarnog praga uz JEDNAK d' -- tvrdnja da OUT "
                    "mora biti populacija PADA.  To je ispravno: postavljena je "
                    "kao provjerljiva (v3 §11).")
        main = [r for r in r2 if r.get("step") == "main"]
        if main:
            m = main[-1]
            print("  glavni run: d'=%.2f  PR=%.2f  pik=%.1f ms  frac@w_max=%.3f"
                  % (m["dprime_roc"], m["pr_out"], m["peak_lag_ms"],
                     m["weights_after"]["frac_at_max"]))
            if m["weights_after"]["frac_at_max"] > 0.5:
                falsified.append(
                    "Tezine nagurane uz w_max (%.0f%%) -- ili je A_minus premali, "
                    "ili je asimetrija obrnula predznak pod M < 0 (v3 §6.9).  "
                    "Uz to pada izolacija kontrasta (v3 §6.11)."
                    % (100 * m["weights_after"]["frac_at_max"]))
            print("  GATE: %s" % ("PROSAO" if m["gate"] else "PAO"))

    # ---------------- R3 ----------------
    r3 = [r for r in jr.read("r3") if r.get("point")]
    if r3:
        _hr("R3  --  mikser: vrhovi simpleksa kao gate")
        for r in r3:
            print("  %-16s c=%s  odziv=%.4f +- %.4f"
                  % (r["point"], [round(x, 2) for x in r["c"]],
                     r["response"], r["response_sd"]))
        vs = [r for r in r3 if r["point"].startswith("vertex_") and "norbf" not in r["point"]]
        if len(vs) >= 2:
            spread = max(v["response"] for v in vs) - min(v["response"] for v in vs)
            noise = float(np.mean([v["response_sd"] for v in vs]))
            print("  raspon vrhova %.4f   sum po sjemenu %.4f" % (spread, noise))
            if spread <= noise:
                falsified.append(
                    "Sva tri vrha simpleksa daju isti odziv (raspon %.4f <= sum "
                    "%.4f).  Mikser je SUVISAN -- ostaje obican R-STDP s tri "
                    "sabirka (v3 §11, prvi falsifikator)." % (spread, noise))
        abl = [r for r in r3 if "norbf" in r["point"]]
        vf = [r for r in r3 if r["point"] == "vertex_f"]
        if abl and vf:
            drop = vf[-1]["response"] - abl[-1]["response"]
            print("  ablacija bez faznih RBF: odziv %.4f (pad %.4f)"
                  % (abl[-1]["response"], drop))
            if abs(drop) > 0.5 * abs(vf[-1]["response"]):
                falsified.append(
                    "Efekat r_f nestaje bez faznih RBF obiljezja -- kriticar je "
                    "naucio SAT TRIAL-a, a ne stanje zadatka (v3 §11).")

    # ---------------- R4 ----------------
    r4 = [r for r in jr.read("r4") if r.get("point") and r.get("grid_wf") is None]
    if r4:
        _hr("R4  --  povrsina nad simpleksom (Scheffe)")
        seen = {}
        for r in r4:
            seen[r["point"]] = r
        design = [(tuple(v["c"]), [o["auc_log"] for o in v["per_seed"]])
                  for v in seen.values()]
        fit = escalate(design)
        if fit.get("ok"):
            print("  model: %s" % ("specijalni kubni" if fit["cubic"] else "kvadratni"))
            for k, v in fit["coefficients"].items():
                print("    %-6s %+.4f" % (k, v))
            print("  R2 = %.3f   R2_adj = %.3f  (ne prodavati ispod 0.7)"
                  % (fit["r2"], fit["r2_adj"]))
            lof = fit["lack_of_fit"]
            if lof.get("available"):
                print("  lack-of-fit: F=%.3f  df=(%d,%d)  p=%.4f  -> %s"
                      % (lof["F"], lof["df_lof"], lof["df_pure"], lof["p_value"],
                         "PUCA" if lof["significant"] else "prolazi"))
            else:
                print("  lack-of-fit: %s" % lof.get("reason"))
            print("  optimum c = %s   (na ivici: %s)"
                  % ([round(x, 3) for x in fit["optimum_c"]], fit["optimum_on_edge"]))
            print("  verdikt: %s" % fit.get("verdict"))
            if fit.get("lack_of_fit_significant"):
                falsified.append(
                    "Lack-of-fit puca i poslije kubnog clana -- odziv nad "
                    "simpleksom NIJE glatka funkcija i mjesavinski model se ne "
                    "smije koristiti kao opis; vrijede samo sirove tacke (v3 §11).")
            if fit["r2"] < 0.7:
                print("  UPOZORENJE: R2 < 0.7 -- povrsina se ne prodaje.")
            print("  reziduali po koordinati (sistematski isti znak = dijagnoza):")
            for k, v in fit["residual_by_coordinate"].items():
                print("    %-4s visoke: %+.4f   niske: %+.4f"
                      % (k, v["mean_resid_high"], v["mean_resid_low"]))

    grid = [r for r in jr.read("r4") if r.get("grid_wf") is not None]
    if grid:
        _hr("R4  --  grid w_f x tau_e  (koja osa nosi efekat)")
        gw = np.array([r["grid_wf"] for r in grid])
        gt = np.array([np.log(r["grid_tau_e"]) for r in grid])
        gy = np.array([r["response"] for r in grid])
        X = np.column_stack([np.ones_like(gw), gw, gt, gw * gt])
        beta, *_ = np.linalg.lstsq(X, gy, rcond=None)
        sd = float(np.std(gy - X @ beta)) or 1e-9
        print("  b_w_f          %+.4f" % beta[1])
        print("  b_log_tau_e    %+.4f" % beta[2])
        print("  b_interakcija  %+.4f" % beta[3])
        sig_wf = abs(beta[1]) > 2 * sd
        sig_te = abs(beta[2]) > 2 * sd
        sig_ix = abs(beta[3]) > 2 * sd
        if sig_ix:
            v = ("INTERAKCIJA -- jedini ishod koji opravdava vezanje tau_e za "
                 "w_f; nagib grebena MJERI kappa (v3 §8.4)")
        elif sig_te and not sig_wf:
            v = "prati samo tau_e"
            falsified.append(
                "Dr* prati samo tau_e a ne w_f -- veza izmedju invarianta i "
                "horizonta NE stoji.  Ostaje standardan nalaz 'duzi trag = "
                "dalji kredit'; vezu ukloniti prije nego udje u tekst (v3 §11).")
        elif sig_wf and not sig_te:
            v = "prati samo w_f"
            falsified.append(
                "Dr* prati samo w_f a ne tau_e -- spojnica je suvisna, w_f vec "
                "radi posao sam.  Ukloniti je, iz suprotnog razloga (v3 §11).")
        else:
            v = "ravno po obje ose"
            falsified.append(
                "Dr* ne prati ni w_f ni tau_e -- sistem ne uci doseg kredita i "
                "ovaj zadatak ga ne mjeri.  Nazad na R1/R2 (v3 §11).")
        print("  verdikt: %s" % v)

    # ---------------- R5 ----------------
    r5 = [r for r in jr.read("r5") if r.get("q") is not None]
    if r5:
        _hr("R5  --  raste li optimalno w_o s pouzdanoscu peer-a q")
        qs = sorted(set(r["q"] for r in r5))
        best = {}
        for q in qs:
            sub = [r for r in r5 if abs(r["q"] - q) < 1e-9]
            b = max(sub, key=lambda r: r["response"])
            best[q] = b
            print("  q=%.1f  optimalno w_o=%.2f  odziv=%.4f"
                  % (q, b["w_o"], b["response"]))
        mono = all(best[qs[i]]["w_o"] <= best[qs[i + 1]]["w_o"]
                   for i in range(len(qs) - 1))
        print("  monotono: %s" % mono)
        if not mono:
            falsified.append(
                "Optimalno w_o NE raste s pouzdanoscu peer-a q -- druga tvrdnja "
                "pada, nezavisno od tau_e (v3 §10 R5).")

    # ---------------- falsifikatori ----------------
    _hr("FALSIFIKATORI")
    if falsified:
        print("  Opalilo ih je %d.  To su REZULTATI, ne kvarovi." % len(falsified))
        for i, f in enumerate(falsified, 1):
            print("\n  %d) %s" % (i, f))
    else:
        print("  Nijedan nije opalio na dosad odvrcenim fazama.")

    if plots:
        _plots(jr)
    print()
    return 0


def _plots(jr: Journal) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib nije instaliran -- grafike preskocene; "
              "brojevi gore su potpuni)")
        return
    d = os.path.join(jr.dir, "plots")
    os.makedirs(d, exist_ok=True)
    made = []

    r1 = jr.read("r1")
    if r1 and r1[-1].get("stdp_window"):
        w = r1[-1]["stdp_window"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.plot(w["delta_t"], w["e"], "o-", ms=3)
        ax.set_xlabel("Dt = t_post - t_dolaska  [ms]")
        ax.set_ylabel("promjena eligibility-ja")
        ax.set_title("Empirijski STDP prozor (Dt=0 -> %s)" % w["sign_at_zero"])
        fig.tight_layout()
        fig.savefig(os.path.join(d, "stdp_window.png"), dpi=120)
        plt.close(fig)
        made.append("stdp_window.png")

        sc = r1[-1].get("delay_scan")
        if sc:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(sc["lags"], sc["response"], "o-", ms=3, label="A->B")
            ax.plot(sc["lags"], sc["response_swapped"], "s--", ms=3, label="zamijenjeno")
            ax.axvline(20, color="r", ls=":", lw=1, label="pravi lag")
            ax.set_xlabel("razmak A-B [ms]")
            ax.set_ylabel("OUT spike-ova / trial")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(d, "delay_scan.png"), dpi=120)
            plt.close(fig)
            made.append("delay_scan.png")

    pts = [r for r in jr.read("r4") if r.get("point") and r.get("grid_wf") is None]
    if pts:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for r in pts:
            s, o, f = r["c"]
            x = 0.5 * (2 * o + f) / max(s + o + f, 1e-9)
            y = (np.sqrt(3) / 2) * f / max(s + o + f, 1e-9)
            ax.scatter([x], [y], s=180, c=[r["response"]], cmap="viridis",
                       vmin=min(p["response"] for p in pts),
                       vmax=max(p["response"] for p in pts))
            ax.annotate("%.3f" % r["response"], (x, y), fontsize=6,
                        ha="center", va="center", color="w")
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], "k-", lw=0.8)
        for lbl, xy in (("w_s", (0, -0.05)), ("w_o", (1, -0.05)),
                        ("w_f", (0.5, np.sqrt(3) / 2 + 0.03))):
            ax.annotate(lbl, xy, ha="center", fontsize=9)
        ax.set_axis_off()
        ax.set_title("odziv nad simpleksom")
        fig.tight_layout()
        fig.savefig(os.path.join(d, "simplex.png"), dpi=120)
        plt.close(fig)
        made.append("simplex.png")

    print("\n  grafike -> %s   (%s)" % (d, ", ".join(made) if made else "nema"))
