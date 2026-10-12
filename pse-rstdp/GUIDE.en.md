# PSE-R-STDP — guide

Ordered from most important to least.

---

## 1. What this measures

The Care Invariant `c = (w_s, w_o, w_f)`, with `w_s + w_o + w_f = 1`, acts as a
**neuromodulator** in three-factor R-STDP learning. A single number `M` is
broadcast to the whole network and decides whether — and in which direction —
anything consolidates.

The claim under test, in one sentence:

> The weight the invariant gives to the future (`w_f`) should shift the
> **reach of credit assignment backwards** — the largest reward delay `Δr` at
> which the task is still learned. That quantity is **`Δr*`**, and it is the
> main dependent variable.

The task is temporal coincidence detection: a burst on channel A, then a burst
on channel B 20 ms later (SIGNAL), versus two independent bursts (NOISE).
Marginal statistics are **identical by construction** — only the dependence
differs. No rate detector can do anything, and this is **proven** in phase R0
rather than assumed.

No gradient, no backprop, online. Pure NumPy, 110 neurons, 1680 synapses,
model state under 1 MB.

---

## 2. Installation

```bash
sudo dpkg -i pse-rstdp_3.0.0_all.deb
sudo apt-get -f install          # if python3-numpy is missing
```

The only hard dependency is `python3-numpy`. `python3-matplotlib` is
recommended — without it everything still works, plots are simply skipped.

Check:
```bash
pse-rstdp info
```

---

## 3. First steps — in exactly this order

### 3.1 Measure speed before starting anything long

```bash
pse-rstdp bench --profile surface
```

This runs 10 000 steps and **extrapolates the budget for every phase**. Don't
estimate the time — measure it. If it comes out longer than one night, shorten
the `Δr` ladder (`--short-ladder`) **before** reducing the number of seeds;
seeds are needed for pure error in the lack-of-fit test.

### 3.2 Validate the whole chain at small scale (10–15 min)

```bash
pse-rstdp run --phase all --scale 0.03 --no-gate --out ~/probe
```

`--scale` shortens every run; `--no-gate` continues even when a gate fails.
Both are needed: at 3% length R2 has no chance of learning, so the chain would
legitimately stop there, and the point is to get through **all** phases.

The results are **not scientifically usable** — the point is only to confirm all
six phases pass without breaking. Takes ~30–45 min on the Surface.

### 3.3 The real run

```bash
pse-rstdp run --phase all --profile surface --seeds 1,2 --out ~/pse-rstdp-results
```

Phases run in order and **each has a gate**. If a gate fails, the chain stops —
that is not a program failure but a falsifier that fired.

### 3.4 Read the result

```bash
pse-rstdp report --out ~/pse-rstdp-results --plots
```

---

## 4. How to read the result

The report ends with a **FALSIFIERS** section. Read that first.

If one fired, that is a **result**, not a bug. The design is built so that it
can be knocked down; if it is, you learned something. Nine falsifiers are built
in as executable checks, and they run **before** you look at the numbers —
otherwise they get read as confirmation.

The important ones:

| Finding | Means |
|---|---|
| All three simplex vertices give the same response | The mixer is redundant — plain R-STDP with three addends |
| `Δr*` tracks only `tau_e`, not `w_f` | The invariant↔horizon link **does not hold**; remove it before it enters any text |
| `Δr*` tracks only `w_f`, not `tau_e` | The link is redundant for the opposite reason |
| `Δr*` flat on both axes | The system isn't learning credit reach; back to R1/R2 |
| Interaction (high only when both are high) | **The only outcome that justifies the link** — and the ridge slope *measures* `kappa` instead of you guessing it |
| `PR ≈ 1` with equal `d′` | The claim that OUT must be a population falls — correctly, since it was posed as testable |
| Empirical STDP window ≠ intended | Bookkeeping error in the traces — **every result up to that point is invalid** |
| Weights piled at `w_max` | `A_minus` too small, or the asymmetry inverted under `M < 0` |
| Lack-of-fit fails even after the cubic term | The response is not a smooth function; the mixture model must not be used as a description |

---

## 5. Two machines — this is not optional

If you run half the design on one machine and half on another and the results
differ, **the machine becomes a confound** in a design that is entirely about
contrasts. That would undo everything done around channel normalisation and
`eta_eff`.

So:

```bash
# on the machine you take as reference:
pse-rstdp selftest --update

# on the other machine, with the same package file:
pse-rstdp selftest
```

It must print `OK — bit-identical with reference`. If it doesn't, **do not
merge** results from the two sources into one design.

Determinism rests on three things: one RNG per `(seed, phase name)` instead of
a global one; `zlib.crc32` instead of the built-in `hash()` which is randomised
per process; and **no BLAS matmul in the hot loop** — multithreaded BLAS
reorders summations and breaks bit-identity.

**There is no separate Dell version.** The difference between machines is three
numbers in a profile:

| | `surface` | `dell` |
|---|---|---|
| parallel processes | 2 | 8 |
| batch B | 8 | 32 |
| checkpoint every | 50k steps | 200k steps |

```bash
pse-rstdp run --profile dell --out ~/pse-rstdp-results   # same directory
```

Results merge into the same JSONL with no confound.

**Why there is no GPU path:** arrays are 1680–13 000 elements, with ~20
operations per step. A kernel launch costs 5–10 µs; a GPU only pays off from
~10⁵–10⁶ elements per operation. At these sizes a GPU version would be
**slower** than NumPy. It would only make sense at ~5000 neurons, and that is
conditional on the main result passing first.

---

## 6. Interruption and resume

A laptop on a trip gets closed, overheats, and runs out of battery. The package
assumes this:

- **Incremental JSONL** — every run is written the moment it finishes, with
  `fsync`. An interruption at run 200 leaves 199 results, not zero.
- **Resume** — the same command skips already-finished units by key.

So: `Ctrl-C` whenever, then the same command again. Nothing is lost.

To redo a phase from scratch: `--reset`.

---

## 7. Phases

Each produces something the next one uses, and each has a gate.

| Phase | What it does | Gate | Produces |
|---|---|---|---|
| **R0** | Proves the marginals are identical, that a rate detector gives ~50%, and that a coincidence oracle gives >80% | all three | — |
| **R1** | Hand-wires a detector **without learning** | `d′ > 1.5` | `g_syn`, `out_drive`, `W_target` |
| **R2** | Pure R-STDP `c=(1,0,0)`; `A_minus` calibration; convention comparison; PR check | `d′ > 1.0`, weights bimodal | `A_minus` |
| **R3** | Three simplex vertices + critic ablation | at least two vertices differ above noise | — |
| **R4** | 10 mixture-design points, Scheffé fit, lack-of-fit, `w_f × tau_e` grid | `R² ≥ 0.7`, lack-of-fit passes | — |
| **R5** | Does optimal `w_o` grow with peer reliability `q` | monotone | — |

**R1 is the most valuable phase in the plan.** If a hand-wired network cannot
separate SIGNAL from NOISE, no learning will — and then you know the problem is
in the **substrate**, not the rule. Without it you spend days debugging R-STDP
while the bug is in the LIF.

Individually:
```bash
pse-rstdp run --phase r1 --out ~/pse-rstdp-results
```

---

## 8. Parameters and their provenance

No number in `config.py` is a guess. Each falls into one of three categories:

- **DERIVED** from the task or from algebra
- **DECIDED** as a convention or normalisation
- **MEASURED** — produced by a named phase

Measured (never entered by hand):

| Quantity | From | Why it must not be a guess |
|---|---|---|
| `g_syn`, `out_drive` | R1 | The two layers have different operating points; the coincidence margin is only 2× and the window is narrow |
| `W_target` | R1 | Depends on how many H neurons drive OUT — a property of the wired network, not of the threshold |
| `A_minus` | R2 calibration run | A closed-form correction assumes a **flat** `Δt` distribution, whereas the real one is shaped by rates, refractoriness, delays, and the fact that the post neuron fires *because of* the arrivals |

A few derived ones, to show how:

```
gamma     = exp(-dt/tau_r) = 0.99750        tau_r = 400 ms is the TASK horizon
tau_e_max = tau_r                           from gamma*lambda <= gamma
tau_e     = 50 * (tau_r/50)^w_f             geometric; w_f=1 -> lambda=1 (Monte Carlo)
k_crit    = 2                               FA floor 3.7% at lambda = 0.3
rho_max   = 0.25 Hz                         FA <= 9% at k_crit = 2
```

`kappa` from earlier versions **no longer exists** — `tau_e_max` is derived,
not a free number.

---

## 9. Output layout

```
~/pse-rstdp-results/
├── meta.json        versions, machine, profile, all parameters
├── measured.json    g_syn, out_drive, W_target, A_minus
├── r0.jsonl ... r5.jsonl
└── plots/           if run with --plots
```

One JSONL line = one work unit, with a `key` used to recognise it on resume.
Each line also carries the diagnostics from §7 of the spec: `d′` per `Δr`, full
ROC, participation ratio, weight histogram, `eta_eff`, `E[|M|]`, `theta`
distribution.

---

## 10. When something doesn't work

**"GATE NIJE PROSAO" (gate failed)** — usually the correct outcome, not a bug.
Read the report; the falsifier tells you what fell.

**R1 gives `d′ ≈ 0`** — the substrate doesn't carry the task. Check
`n_detectors` in the report: if it's 0, the delays don't cover the bin around
20 ms. If it's large but `d′` is low, check that OUT fires at all
(`rate_out_hz`).

**Network is silent (`rate_h_hz ≈ 0`)** — `g_syn` is too small. R1 searches a
2-D grid; if the optimum sits on the grid edge, widen the range.

**Too slow on the Surface** — `--short-ladder` first, fewer seeds only after.
Never the other way round.

**Results don't match between machines** — `pse-rstdp selftest` on both. The
most common cause is a different NumPy version; that is why it is written into
`meta.json`.

---

## 11. What this package deliberately does **not** have

- **A GUI.** The spec says explicitly that no GUI arrives until the peak
  shifts. When it does, a PySide6 layer is added on top of the JSONL without
  touching the kernel.
- **GPU.** See §5 — it would be slower.
- **PyTorch.** The whole point is that this fits in 4 GB and runs online.
- **scipy.** The inverse normal, the incomplete beta, and Nelder-Mead are
  hand-written so the `.deb` has one hard dependency and installs reliably on
  Mint.

---

## 12. Relation to the spec

The code implements `PSE_R-STDP_nacrt_v3.docx` (shipped in
`/usr/share/doc/pse-rstdp/`). References like "v3 §6.4" in the comments point
to sections of that document.

Three things were found **during implementation** and went into the code:

1. **R0 caught a bug in the task definition.** As NOISE was specified
   (`t_B ~ U(0,400)`, independent), the marginals were **not** identical — the
   mean time of the B burst would differ by ~20 ms between classes, and a plain
   mean-time detector would solve the task with no timing at all. Fix: in NOISE,
   `t_B` is built from an independent draw of `t_A'` **and** an independent lag,
   so both marginals are the same convolution and only the dependence differs.

2. **The empirical STDP window caught double counting at `Δt = 0`.** The delayed
   pre-trace `x_hist` already includes the spike sent at `t−d`, so option B is
   simply `xd`, while option A requires *subtracting* the current arrival.

3. **Homeostasis needs two bands, not one.** The limits derived from the FA
   budget (`0.02–0.25 Hz`) apply to OUT; applying them to H would suffocate the
   hidden layer, which must be active to compute anything. H gets `2–20 Hz`.

All three are of a kind that is invisible on paper — hence R0 and R1 before
everything else.
