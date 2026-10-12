# PSE-R-STDP

R-STDP koincidencijski detektor s Care Invariantom kao modulatorom.
Implementacija nacrta **v3** (`PSE_R-STDP_nacrt_v3.docx`).

Vodiči: **[GUIDE.sr.md](GUIDE.sr.md)** · **[GUIDE.en.md](GUIDE.en.md)**

---

## Ukratko

Care Invariant `c = (w_s, w_o, w_f)` sa `w_s + w_o + w_f = 1` je neuromodulator
u trofaktorskom R-STDP učenju. Mjeri se da li težina koju invariant daje
budućnosti pomjera domet dodjele kredita unazad — `Δr*`, najveće kašnjenje
nagrade pri kojem se zadatak još nauči.

Čist NumPy · 110 neurona · 1680 sinapsi · stanje modela < 1 MB · bez PyTorch-a,
bez GPU-a · deterministično (isti seed → bit-identičan rezultat na svakoj
mašini).

## Brzi start

```bash
sudo dpkg -i pse-rstdp_3.0.0_all.deb

pse-rstdp bench                                   # izmjeri budžet
pse-rstdp run --phase all --scale 0.03 --out ~/probe   # provjeri lanac, ~15 min
pse-rstdp run --phase all --out ~/rezultati       # pravi run
pse-rstdp report --out ~/rezultati --plots
```

## Gradnja

```bash
./build_deb.sh            # -> pse-rstdp_3.0.0_all.deb   (treba samo ar/tar/gzip)
```

Ili bez paketa:

```bash
PYTHONPATH=. python3 -m pse_rstdp.cli info
```

## Moduli

| Fajl | Šta je |
|---|---|
| `config.py` | Parametri, s porijeklom (izveden / odlučen / mjeren) i profili |
| `task.py` | Generator trialova + faza R0 (validnost zadatka) |
| `net.py` | LIF + kašnjenja + COO sinapse + R-STDP + kritičar + homeostaza |
| `evaluate.py` | Ručno ožičavanje (R1), `d′`/ROC, skeniranje kašnjenja, empirijski STDP prozor |
| `stats.py` | `d′`, ROC (Mann-Whitney), AUC nad `log Δr`, `Δr*`, participation ratio |
| `design.py` | Scheffé mixture model, lack-of-fit (F test ručno), najbrži uspon |
| `phases.py` | R0–R5 s gate-ovima |
| `runner.py` | JSONL dnevnik, nastavljanje, `measured.json` |
| `analysis.py` | Izvještaj + falsifikatori |
| `cli.py` | `pse-rstdp` |

## Zavisnosti

Tvrda: `python3-numpy`. Opcionalna: `python3-matplotlib` (samo za grafike).
`scipy` **nije** potreban — inverzna normalna, nepotpuna beta i Nelder-Mead su
napisani ručno.
