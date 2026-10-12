# -*- coding: utf-8 -*-
"""
Parametri i profili.  Svaki broj ovdje ima porijeklo iz nacrta v3 §5.5:
IZVEDEN iz zadatka/algebre, ODLUCEN kao konvencija, ili MJEREN (proizveden fazom).

MJERENI parametri (A_minus, W_target) imaju samo pocetnu vrijednost ovdje;
konacnu upisuje faza koja ih proizvodi (R2 odn. R1) u state/measured.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict, replace

# ------------------------------------------------------------------
# Populacije i topologija  (v3 §4)
# ------------------------------------------------------------------
N_A = 20
N_B = 20
N_H = 60
N_OUT = 10
N = N_A + N_B + N_H + N_OUT           # 110

SL_A = slice(0, N_A)
SL_B = slice(N_A, N_A + N_B)
SL_H = slice(N_A + N_B, N_A + N_B + N_H)
SL_OUT = slice(N_A + N_B + N_H, N)

IDX_A = tuple(range(SL_A.start, SL_A.stop))
IDX_B = tuple(range(SL_B.start, SL_B.stop))
IDX_H = tuple(range(SL_H.start, SL_H.stop))
IDX_OUT = tuple(range(SL_OUT.start, SL_OUT.stop))

# Fiksni broj sinapsi po projekciji.  Fiksan (a ne Bernoulli p) da bi n_syn bio
# identican kroz batch os -- inace se COO nizovi ne mogu slagati u (B, n_syn).
# Brojevi su tacno p * (n_pre * n_post) iz v3 §4.
K_A_H = 360        # 20*60*0.30
K_B_H = 360        # 20*60*0.30
K_H_H = 360        # 60*60*0.10
K_H_OUT = 600      # 60*10*1.00  (gusto)
N_SYN = K_A_H + K_B_H + K_H_H + K_H_OUT      # 1680

FRAC_INH = 0.20    # udio inhibitornih u H


@dataclass(frozen=True)
class Params:
    # ---------- vrijeme ----------  §5.5: odlucen / izveden
    dt: float = 1.0                 # ms   odlucen -- rezolucija kasnjenja
    T: int = 1_000_000              # koraka ~= 2000 trialova   izveden
    trial_len: int = 500            # ms   zadatak
    # Prozor odgovora = CIJELI trial.  v3 §2.3 ga je uveo da "iskljuci A burst",
    # ali detektor po konstrukciji ne moze ispaliti na jedan kanal (marza 2x),
    # pa taj razlog ne postoji.  Zato je prozor od 200 ms bio cista steta:
    # t_A ~ U(0,400), a odgovor stize na t_A + ~60 ms, pa je prozor [300,500)
    # odsijecao vise od polovine koincidencija -- hit je bio zaglavljen na 0.53
    # bez obzira na pojacanje.  Nadjeno u R1.
    # Prelijevanja iz prethodnog trial-a nema: zadnji burst spike je na
    # t_A + 20 <= 420, plus najvece kasnjenje 30 -> 450 < 500.
    resp_win: int = 500
    d_max: int = 30                 # ms   zadatak -- pokriva lag 20 ms s rezervom

    # ---------- neuron ----------
    tau_m_exc: float = 20.0         # ms   odlucen
    tau_m_inh: float = 10.0         # ms   izveden -- inhibicija mora prednjaciti
    v_rest: float = -65.0
    v_reset: float = -70.0
    theta_0: float = -52.0
    t_refrac: int = 3               # koraka

    # ---------- tragovi i STDP ----------
    tau_x: float = 20.0
    tau_y: float = 20.0
    a_plus: float = 1.0             # odlucen -- normalizacija skale
    a_minus: float = 1.1            # MJEREN  -- pocetna; konacnu daje R2 kalibracija
    eta: float = 1e-3               # izveden -- ~0.5% w_max po trialu
    w_max: float = 1.0
    w_init_hi: float = 0.3
    dt0_mode: str = "LTP"           # odlucen -- opcija B (v3 §6.4).  'LTP' | 'ZERO'

    # ---------- kredit i vrijednost ----------
    tau_r: float = 400.0            # ms   zadatak -- trial je 500 ms
    tau_e_min: float = 50.0         # ms   izveden -- ~2*tau_m
    alpha_critic: float = 0.05      # odlucen -- NLMS je skalno invarijantan
    tau_phi: float = 50.0           # ms   odlucen
    n_rbf: int = 5                  # izveden -- 100 ms rezolucija nad 500 ms
    use_rbf: bool = True            # ablacija u R3 gasi ovo

    # ---------- Care Invariant ----------
    w_s: float = 1.0
    w_o: float = 0.0
    w_f: float = 0.0
    tau_e_override: float | None = None   # grid w_f x tau_e (v3 §8.4) puni ovo

    # ---------- peer ----------
    q: float = 0.5                  # izveden -- neinformativan peer = cista kontrola

    # ---------- odluka i nagrada ----------
    k_crit: int = 2                 # IZVEDEN  -- FA pod 3.7% pri lambda=0.3 (v3 §5.5)
    delta_r: int = 20               # ms   kasnjenje nagrade; ljestvica u DR_LADDER

    # ---------- normalizacija ----------
    tau_rms: float = 5000.0         # ms   izveden -- ~10 trialova
    eps: float = 1e-8
    eta_eff_scale: float = 1.0      # puni ga pre-run izjednacavanja (v3 §6.7)

    # ---------- sinapticko pojacanje ----------
    # Pretvara bezdimenzionu tezinu w u struju.  w ostaje u [0,1] (normalizovano),
    # a g_syn nosi fizicku skalu.  MJEREN: kalibrise se pri konstrukciji mreze
    # tako da H sjedne u sredinu svog pojasa -- inace je mreza nijema ili u
    # rafalima, i sve dalje mjeri kvar a ne ucenje.
    g_syn: float = 5.0
    calibrate_gain: bool = True
    h_rate_target_hz: float = 6.0   # ciljna stopa H pri kalibraciji

    # ---------- homeostaza ----------
    tau_rho: float = 2000.0         # ms   izveden -- >= 10x konstanta ucenja
    # Homeostaza i z-skoriranje obiljezja se azuriraju svakih `slow_every`
    # koraka s proporcionalno vecim korakom.  Njihove konstante su 2000 ms
    # odn. sekunde, pa je 20 ms grublji Euler bez posljedica -- a stedi ~25
    # NumPy poziva po koraku, sto je na sporoj masini polovina cijene petlje.
    # Vrijednost je dio specifikacije (ulazi u JSONL), ne skrivena optimizacija.
    slow_every: int = 20
    # DVA POJASA, ne jedan.  Granice za OUT su izvedene iz FA budzeta (§5.5) i
    # nemaju smisla za skriveni sloj: H mora biti AKTIVAN da bi ista racunao.
    # Primijeniti OUT pojas na H bi ugusilo mrezu -- kvar koji izgleda kao
    # neuspjeh ucenja.
    rho_min_hz: float = 0.02        # Hz   OUT: izveden -- anti-smrt pod
    # IZVEDEN iz FA budzeta.  Prozor je sada 500 ms (v. resp_win), pa je
    #   lambda = 10 neurona * rho * 0.5 s = 5*rho
    #   P(>=2 | lambda=0.5) = 9.0%   ->  rho_max = 0.10 Hz
    # (Uz stari prozor od 200 ms izlazilo je 0.25 Hz; broj se mijenja s prozorom
    #  jer je izveden, a ne izabran.)
    rho_max_hz: float = 0.10
    rho_min_h_hz: float = 2.0       # Hz   H: donja ivica radnog rezima
    rho_max_h_hz: float = 20.0      # Hz   H: gornja ivica radnog rezima
    beta_up: float = 1e-3           # odlucen -- bijeg opasniji od tisine
    beta_dn: float = 1e-4
    theta_lo: float = -58.0
    theta_hi: float = -45.0
    w_target: float = 3.0           # MJEREN -- izlaz iz R1; ovo je samo fallback
    rho_scale: float = 1e-3
    scale_every: int = 100          # koraka

    # ---------- zadatak ----------
    lag_true: float = 20.0          # ms
    jitter: float = 3.0             # ms  (sd)
    burst_n: int = 5                # spike-ova po burstu
    burst_isi: float = 5.0          # ms  (200 Hz)
    burst_jitter: float = 1.0       # ms
    # Udio neurona kanala koji dobiju burst.  MORA biti 1.0: detektor iz R1 ima
    # tacno jednu A-sinapsu i jednu B-sinapsu (marza koincidencije je samo 2x,
    # v. evaluate.handwire).  Pri 0.5 su oba bas njegova neurona u burstu tek u
    # 25% trialova, pa detektor radi na cetvrtini duty cycle-a i OUT nikad ne
    # dosegne prag.  Nadjeno u R1.  Klase i dalje imaju identicne marginale.
    burst_frac: float = 1.0
    t_max_burst: float = 400.0      # ms  gornja granica pocetka burst-a

    # ---------- ucenje ukljuceno? ----------
    plastic: bool = True
    plastic_inh: bool = False       # inhibitorne fiksirane u fazi 1 (v3 §6.9)
    homeostasis: bool = True
    theta_vector: bool = True       # False = skalarni prag po populaciji (PR test, R2)
    shuffle_input: bool = False     # kalibracijski run (v3 §6.9)

    # ---------- izvedene velicine ----------
    @property
    def gamma(self) -> float:
        """IZVEDEN: gamma = exp(-dt/tau_r).  v1 je imao 0.95 -> horizont 20 ms."""
        return math.exp(-self.dt / self.tau_r)

    @property
    def tau_e_max(self) -> float:
        """IZVEDEN iz algebre TD(lambda): gamma*lambda <= gamma  =>  tau_e <= tau_r."""
        return self.tau_r

    @property
    def tau_e(self) -> float:
        """Geometrijski u w_f.  w_f=0 -> 50 ms, w_f=1 -> tau_r (lambda = 1)."""
        if self.tau_e_override is not None:
            return float(self.tau_e_override)
        lo, hi = self.tau_e_min, self.tau_e_max
        return lo * (hi / lo) ** self.w_f

    @property
    def lam(self) -> float:
        """lambda = exp(dt*(1/tau_r - 1/tau_e));  <= 1 po konstrukciji jer tau_e <= tau_r."""
        return math.exp(self.dt * (1.0 / self.tau_r - 1.0 / self.tau_e))

    @property
    def n_features(self) -> int:
        # [lowpass(sumA), lowpass(sumB), lowpass(H)*60, lowpass(OUT)*10, RBF*n, bias]
        return 2 + N_H + N_OUT + (self.n_rbf if self.use_rbf else 0) + 1

    @property
    def rho_min(self) -> float:
        """Hz -> po koraku (OUT)."""
        return self.rho_min_hz * self.dt / 1000.0

    @property
    def rho_max(self) -> float:
        return self.rho_max_hz * self.dt / 1000.0

    @property
    def rho_min_h(self) -> float:
        """Hz -> po koraku (H)."""
        return self.rho_min_h_hz * self.dt / 1000.0

    @property
    def rho_max_h(self) -> float:
        return self.rho_max_h_hz * self.dt / 1000.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(gamma=self.gamma, tau_e=self.tau_e, lam=self.lam,
                 n_features=self.n_features)
        return d


# ------------------------------------------------------------------
# Ljestvica kasnjenja nagrade  (v3 §2.3) -- log, faktor 4
# ------------------------------------------------------------------
DR_LADDER = (20, 80, 320, 1280)
DR_LADDER_SHORT = (20, 160, 1280)      # fallback iz v3 §8.5 ako je Surface prespor


# ------------------------------------------------------------------
# Profili  (v3 §12.4) -- razlika su tri broja, ne druga verzija
# ------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str
    jobs: int
    batch: int
    checkpoint_every: int


PROFILES = {
    "surface": Profile("surface", jobs=2, batch=8, checkpoint_every=50_000),
    "dell": Profile("dell", jobs=8, batch=32, checkpoint_every=200_000),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise SystemExit("nepoznat profil %r; dostupni: %s"
                         % (name, ", ".join(sorted(PROFILES))))
    return PROFILES[name]


# ------------------------------------------------------------------
# Tacke mixture dizajna  (v3 §8.2) -- augmentirani simplex-centroid
# ------------------------------------------------------------------
_T = 1.0 / 3.0
_S = 1.0 / 6.0
_D = 2.0 / 3.0

DESIGN_POINTS = (
    # vrhovi -- ujedno gate iz v3 §10 R3
    ("vertex_s", (1.0, 0.0, 0.0)),
    ("vertex_o", (0.0, 1.0, 0.0)),
    ("vertex_f", (0.0, 0.0, 1.0)),
    # sredine ivica -- daju interakcije beta_so, beta_sf, beta_of
    ("edge_so", (0.5, 0.5, 0.0)),
    ("edge_sf", (0.5, 0.0, 0.5)),
    ("edge_of", (0.0, 0.5, 0.5)),
    # centroid -- lack-of-fit
    ("centroid", (_T, _T, _T)),
    # aksijalne -- jos 2 df + specijalni kubni clan
    ("axial_s", (_D, _S, _S)),
    ("axial_o", (_S, _D, _S)),
    ("axial_f", (_S, _S, _D)),
)

VERTEX_NAMES = ("vertex_s", "vertex_o", "vertex_f")

# Grid w_f x tau_e  (v3 §8.4)
GRID_WF = (0.1, 0.3, 0.5)
GRID_TAU_E = (50.0, 141.0, 400.0)

# Skeniranje pouzdanosti peer-a  (v3 §10 R5)
Q_LADDER = (0.5, 0.7, 0.9)

SCHEMA_VERSION = 3
