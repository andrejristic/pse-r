# PSE-R-STDP — vodič

Redom od najvažnijeg ka manje važnom.

---

## 1. Šta ovo mjeri

Care Invariant `c = (w_s, w_o, w_f)`, `w_s + w_o + w_f = 1`, koristi se kao
**neuromodulator** u trofaktorskom R-STDP učenju. Jedan broj `M` se emituje
cijeloj mreži i određuje da li se, i u kom smjeru, bilo šta konsoliduje.

Tvrdnja pod testom, u jednoj rečenici:

> Težina koju invariant daje budućnosti (`w_f`) treba da pomjeri **domet
> dodjele kredita unazad** — najveće kašnjenje nagrade `Δr` pri kojem se
> zadatak još nauči. Ta veličina se zove **`Δr*`** i ona je glavna zavisna
> varijabla.

Zadatak je detekcija temporalne koincidencije: burst na kanalu A, pa burst na
kanalu B 20 ms kasnije (SIGNAL), naspram dva nezavisna bursta (NOISE).
Marginalne statistike su **identične po konstrukciji** — razlikuje se samo
zavisnost. Nijedan detektor stope ne može ništa, i to se **dokazuje** u fazi R0,
ne pretpostavlja.

Bez gradijenta, bez backpropa, online. Čist NumPy, 110 neurona, 1680 sinapsi,
stanje modela ispod 1 MB.

---

## 2. Instalacija

```bash
sudo dpkg -i pse-rstdp_3.0.0_all.deb
sudo apt-get -f install          # ako fali python3-numpy
```

Tvrda zavisnost je samo `python3-numpy`. `python3-matplotlib` je
preporučen — bez njega sve radi, samo se grafike preskaču.

Provjera:
```bash
pse-rstdp info
```

---

## 3. Prvi koraci — tačno ovim redom

### 3.1 Izmjeri brzinu prije nego pokreneš išta dugo

```bash
pse-rstdp bench --profile surface
```

Ovo odvrti 10 000 koraka i **ekstrapolira budžet za sve faze**. Ne procjenjuj
vrijeme — izmjeri ga. Ako ispadne više od jedne noći, skrati `Δr` ljestvicu
(`--short-ladder`) **prije** nego smanjuješ broj sjemena; sjemena trebaju za
pure error u lack-of-fit testu.

### 3.2 Provjeri cijeli lanac na maloj skali (10-15 min)

```bash
pse-rstdp run --phase all --scale 0.03 --no-gate --out ~/probe
```

`--scale` skraćuje svaki run; `--no-gate` nastavlja i kad gate padne. Oboje je
nužno: na 3% dužine R2 nema šanse da nauči, pa bi lanac legitimno stao već tamo,
a cilj je proći **sve** faze kroz kod.

Rezultati **nisu naučno upotrebljivi** — svrha je samo da se vidi da svih šest
faza prolazi bez pucanja. Traje ~30–45 min na Surfaceu.

### 3.3 Pravi run

```bash
pse-rstdp run --phase all --profile surface --seeds 1,2 --out ~/pse-rstdp-results
```

Faze idu redom i **svaka ima gate**. Ako gate ne prođe, lanac staje — to nije
kvar programa nego falsifikator koji je opalio.

### 3.4 Pročitaj rezultat

```bash
pse-rstdp report --out ~/pse-rstdp-results --plots
```

---

## 4. Kako se čita rezultat

Izvještaj završava sekcijom **FALSIFIKATORI**. Pročitaj nju prvu.

Ako je neki opalio — to je **rezultat**, ne greška. Nacrt je postavljen tako da
se može oboriti; ako se obori, saznao si nešto. Devet falsifikatora je ugrađeno
kao izvršne provjere, i pokreću se **prije** nego pogledaš brojeve, jer se
inače čitaju kao potvrda.

Najvažniji:

| Nalaz | Znači |
|---|---|
| Sva tri vrha simpleksa daju isti odziv | Mikser je suvišan — ostaje običan R-STDP s tri sabirka |
| `Δr*` prati samo `tau_e`, ne `w_f` | Veza invariant↔horizont **ne stoji**; ukloniti je prije nego uđe u tekst |
| `Δr*` prati samo `w_f`, ne `tau_e` | Spojnica je suvišna iz suprotnog razloga |
| `Δr*` ravno po obje ose | Sistem ne uči doseg kredita; nazad na R1/R2 |
| Interakcija (visoko samo kad su oba visoka) | **Jedini ishod koji opravdava vezu** — i nagib grebena *mjeri* `kappa` umjesto da ga pogađaš |
| `PR ≈ 1` uz jednak `d′` | Tvrdnja da OUT mora biti populacija pada — ispravno, postavljena je kao provjerljiva |
| Empirijski STDP prozor ≠ namjeravani | Greška u knjigovodstvu tragova — **svi rezultati do te tačke su nevažeći** |
| Težine nagurane uz `w_max` | `A_minus` premali, ili je asimetrija obrnula predznak pod `M < 0` |
| Lack-of-fit puca i poslije kubnog člana | Odziv nije glatka funkcija; mješavinski model se ne smije koristiti kao opis |

---

## 5. Dvije mašine — ovo nije opciono

Ako pola dizajna odvrtiš na jednoj mašini a pola na drugoj i rezultati se
razlikuju, **mašina postaje konfund** u dizajnu koji je cijeli o kontrastima.
Time bi se poništilo sve što je urađeno oko normalizacije kanala i `eta_eff`.

Zato:

```bash
# na mašini koju uzimaš kao referentnu:
pse-rstdp selftest --update

# na drugoj mašini, s istim fajlom paketa:
pse-rstdp selftest
```

Mora izaći `OK — bit-identično s referencom`. Ako ne izađe, **ne spajaj**
rezultate iz dva izvora u isti dizajn.

Determinizam je osiguran s tri stvari: jedan RNG po `(seed, ime faze)` umjesto
globalnog; `zlib.crc32` umjesto ugrađenog `hash()` koji je randomizovan po
procesu; i **nikakav BLAS matmul u vrućoj petlji** — višenitni BLAS
prerasporedi sabiranja i razbije bit-identičnost.

**Nema druge verzije za Dell.** Razlika između mašina su tri broja u profilu:

| | `surface` | `dell` |
|---|---|---|
| paralelnih procesa | 2 | 8 |
| batch B | 8 | 32 |
| checkpoint svakih | 50k koraka | 200k koraka |

```bash
pse-rstdp run --profile dell --out ~/pse-rstdp-results   # isti direktorij
```

Rezultati se spajaju u isti JSONL bez ijednog konfunda.

**Zašto nema GPU putanje:** nizovi su 1680–13 000 elemenata, a ima ih ~20
operacija po koraku. Kernel launch košta 5–10 µs; GPU se isplati tek od
~10⁵–10⁶ elemenata po operaciji. Na ovim veličinama bi GPU bio **sporiji** od
NumPy-ja. Ima smisla tek na ~5000 neurona, a to je uslovljeno time da glavni
rezultat prvo prođe.

---

## 6. Prekid i nastavak

Laptop na putovanju se zatvara, pregrijava i ostaje bez baterije. Paket to
pretpostavlja:

- **Inkrementalni JSONL** — svaki run se upisuje čim završi, uz `fsync`.
  Prekid na 200. runu ostavlja 199 rezultata, ne nula.
- **Nastavljanje** — ista komanda preskače već završene jedinice po ključu.

Dakle: `Ctrl-C` kad hoćeš, pa ista komanda opet. Ništa se ne gubi.

Da se faza ponovi od nule: `--reset`.

---

## 7. Faze

Svaka proizvodi nešto što sljedeća koristi, i svaka ima gate.

| Faza | Šta radi | Gate | Proizvodi |
|---|---|---|---|
| **R0** | Dokazuje da su marginale identične i da detektor stope daje ~50%, a koincidencijski orakl >80% | sve troje | — |
| **R1** | Ručno ožiči detektor **bez učenja** | `d′ > 1.5` | `g_syn`, `out_drive`, `W_target` |
| **R2** | Čist R-STDP `c=(1,0,0)`; kalibracija `A_minus`; poređenje konvencija; PR provjera | `d′ > 1.0`, težine bimodalne | `A_minus` |
| **R3** | Tri vrha simpleksa + ablacija kritičara | bar dva vrha se razlikuju iznad šuma | — |
| **R4** | 10 tačaka mixture dizajna, Scheffé fit, lack-of-fit, grid `w_f × tau_e` | `R² ≥ 0.7`, lack-of-fit ne puca | — |
| **R5** | Raste li optimalno `w_o` s pouzdanošću peer-a `q` | monotono | — |

**R1 je najvrednija faza plana.** Ako ručno ožičena mreža ne razdvaja SIGNAL od
NOISE, nikakvo učenje neće — i tada znaš da je problem u **supstratu**, ne u
pravilu. Bez nje se danima debuguje R-STDP dok je greška u LIF-u.

Pojedinačno:
```bash
pse-rstdp run --phase r1 --out ~/pse-rstdp-results
```

---

## 8. Parametri i njihovo porijeklo

Nijedan broj u `config.py` nije pogodak. Svaki je u jednoj od tri kategorije:

- **IZVEDEN** iz zadatka ili iz algebre
- **ODLUČEN** kao konvencija ili normalizacija
- **MJEREN** — proizvodi ga imenovana faza

Mjereni (nikad ih ne unosiš ručno):

| Veličina | Odakle | Zašto ne smije biti pogodak |
|---|---|---|
| `g_syn`, `out_drive` | R1 | Dva sloja imaju različite radne tačke; marža koincidencije je samo 2× i prozor je uzak |
| `W_target` | R1 | Zavisi od toga koliko H neurona gađa OUT — svojstvo naučene mreže, ne praga |
| `A_minus` | R2 kalibracijski run | Zatvorena formula pretpostavlja **ravnu** raspodjelu `Δt`, a stvarna je oblikovana stopama, refrakcijom, kašnjenjima i time što post neuron ispaljuje *zbog* dolazaka |

Nekoliko izvedenih, da se vidi kako:

```
gamma     = exp(-dt/tau_r) = 0.99750        tau_r = 400 ms je horizont ZADATKA
tau_e_max = tau_r                           iz gamma*lambda <= gamma
tau_e     = 50 * (tau_r/50)^w_f             geometrijski; w_f=1 -> lambda=1 (Monte-Carlo)
k_crit    = 2                               FA pod 3.7% pri lambda = 0.3
rho_max   = 0.25 Hz                         FA <= 9% pri k_crit = 2
```

`kappa` iz ranijih verzija **više ne postoji** — `tau_e_max` je izveden, ne
slobodan broj.

---

## 9. Struktura izlaza

```
~/pse-rstdp-results/
├── meta.json        verzije, mašina, profil, svi parametri
├── measured.json    g_syn, out_drive, W_target, A_minus
├── r0.jsonl ... r5.jsonl
└── plots/           ako je pokrenuto s --plots
```

Jedan red JSONL = jedna jedinica posla, sa `key` po kojem se prepoznaje pri
nastavljanju. U svakom redu su i dijagnostike iz §7 nacrta: `d′` po `Δr`, puna
ROC, participation ratio, histogram težina, `eta_eff`, `E[|M|]`, raspodjela
`theta`.

---

## 10. Kad nešto ne radi

**„GATE NIJE PROSAO"** — to je najčešće ispravan ishod, ne kvar. Pogledaj
izvještaj; falsifikator ti kaže šta je palo.

**R1 daje `d′ ≈ 0`** — supstrat ne nosi zadatak. Provjeri `n_detectors` u
izvještaju: ako je 0, kašnjenja ne pokrivaju bin oko 20 ms. Ako je velik a `d′`
nizak, provjeri da OUT uopšte ispaljuje (`rate_out_hz`).

**Mreža je nijema (`rate_h_hz ≈ 0`)** — `g_syn` je premali. R1 ga traži na 2-D
rešetki; ako je optimum na ivici rešetke, proširi raspon.

**Prespro na Surfaceu** — `--short-ladder`, pa tek onda manje sjemena.
Nikad obrnuto.

**Rezultati se ne poklapaju između mašina** — `pse-rstdp selftest` na obje.
Najčešći uzrok je različita verzija NumPy-ja; ona se upisuje u `meta.json` baš
zato.

---

## 11. Šta ovaj paket namjerno **nema**

- **GUI.** Nacrt izričito kaže da GUI ne dolazi dok se pik ne pomjeri. Kad se
  pomjeri, PySide6 sloj se dodaje preko JSONL-a bez diranja kernela.
- **GPU.** Vidi §5 — bio bi sporiji.
- **PyTorch.** Cijela poenta je da ovo staje na 4 GB i radi online.
- **scipy.** Inverzna normalna, nepotpuna beta i Nelder-Mead su napisani ručno
  da bi `.deb` imao jednu tvrdu zavisnost i sigurno se instalirao na Mint.

---

## 12. Odnos prema nacrtu

Kod implementira `PSE_R-STDP_nacrt_v3.docx` (isporučen u
`/usr/share/doc/pse-rstdp/`). Reference tipa „v3 §6.4" u komentarima pokazuju
na sekcije tog dokumenta.

Tri stvari su nađene **tokom implementacije** i ušle su u kod:

1. **R0 je otkrio grešku u definiciji zadatka.** Kako je NOISE bio opisan
   (`t_B ~ U(0,400)` nezavisno), marginale **nisu** bile identične — srednje
   vrijeme B bursta razlikovalo bi se za ~20 ms i običan detektor srednjeg
   vremena riješio bi zadatak bez ikakvog tajminga. Ispravka: u NOISE se `t_B`
   gradi iz nezavisnog izvlačenja `t_A'` **i** nezavisnog lag-a, pa su obje
   marginale ista konvolucija a razlikuje se samo zavisnost.

2. **Empirijski STDP prozor je uhvatio dvostruko brojanje na `Δt = 0`.**
   Odgođeni pre-trag `x_hist` već uključuje spike poslan u `t−d`, pa je opcija B
   naprosto `xd`, a opcija A traži *oduzimanje* tekućeg dolaska.

3. **Homeostaza traži dva pojasa, ne jedan.** Granice izvedene iz FA budžeta
   (`0.02–0.25 Hz`) važe za OUT; primijeniti ih na H ugušilo bi skriveni sloj,
   koji mora biti aktivan da bi išta računao. H ide na `2–20 Hz`.

Sve tri su tipa koji se ne vidi na papiru — otud R0 i R1 prije svega ostalog.
