# -*- coding: utf-8 -*-
"""
Mreza: LIF + kasnjenja + R-STDP + Care-modulisan neuromodulator.   v3 §5

Reprezentacija je COO (ravni nizovi po sinapsi), ne guste N x N matrice.
Razlog nije samo memorija (1680 umjesto 12100 elemenata) nego to sto greska
"ucenje gradi tezinu tamo gdje sinapse nema" postaje strukturno nemoguca --
nema sta da se maskira.  Uz to se kasnjenja prirodno uklapaju: d je samo jos
jedna kolona u istoj listi.  (v3 §3)

Batch os: sve je oblika (B, ...).  Na ovim velicinama je NumPy dominiran
overhead-om poziva, ne aritmetikom, pa vrtjeti 8 kombinacija odjednom kosta
skoro isto koliko jednu.  (v3 §8.5)

DVA NOSIVA REDOSLIJEDA  -- v3 §12.2, ne dirati:
  1. s_hist/x_hist se CITAJU u koraku (a) PRIJE nego se PISU u koraku (d).
     Pri d = D_max je row == ptr, i daje ispravnu vrijednost s(t - D_max)
     samo zbog tog reda.
  2. Tragovi x i y se azuriraju POSLIJE racunanja e.  Ako se pomjere iznad,
     konvencija na dt=0 neprimjetno predje iz opcije B u opciju C -- greska iz v1.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .config import Params
from .rng import make_rng
from .task import SIGNAL, build_raster, make_trials, shuffle_raster


# ==================================================================
# Topologija
# ==================================================================
def _sample_pairs(rng, src, dst, k, allow_self=False):
    """Tacno k razlicitih (pre, post) parova.  Fiksan broj -- ne Bernoulli --
    da bi n_syn bio identican kroz batch os."""
    src = np.asarray(src)
    dst = np.asarray(dst)
    n_pairs = src.size * dst.size
    pool = np.arange(n_pairs)
    if not allow_self and src.size == dst.size and np.array_equal(src, dst):
        keep = (pool // dst.size) != (pool % dst.size)
        pool = pool[keep]
    sel = rng.choice(pool.size, size=k, replace=False)
    idx = pool[np.sort(sel)]
    return src[idx // dst.size], dst[idx % dst.size]


def _stratified_delays(rng, k, d_max):
    """
    Ciklicno po binovima pa promijesaj.  Uniformno izvlacenje ostavlja neki
    bin prazan uz osjetnu vjerovatnocu -- i to lako bas onaj oko 20 ms, koji
    je jedini koji zadatak trazi.  Stratifikacija garantuje pokrivenost.  (v3 §4)
    """
    d = (np.arange(k) % d_max) + 1
    return d[rng.permutation(k)].astype(np.int32)


def build_topology(rng, p: Params):
    """COO lista sinapsi + Daleov predznak po neuronu."""
    pre_l, post_l = [], []
    for src, dst, k in ((C.IDX_A, C.IDX_H, C.K_A_H),
                        (C.IDX_B, C.IDX_H, C.K_B_H),
                        (C.IDX_H, C.IDX_H, C.K_H_H),
                        (C.IDX_H, C.IDX_OUT, C.K_H_OUT)):
        a, b = _sample_pairs(rng, src, dst, k)
        pre_l.append(a)
        post_l.append(b)
    pre = np.concatenate(pre_l).astype(np.int32)
    post = np.concatenate(post_l).astype(np.int32)
    d = _stratified_delays(rng, pre.size, p.d_max)

    # Dale: predznak po PRE-neuronu, ulazi pri injekciji struje, ne u tezinu.
    # Inace bi neuron mijenjao znak tokom ucenja.  (v3 §6.9)
    sgn = np.ones(C.N)
    n_inh = int(round(C.FRAC_INH * C.N_H))
    inh = rng.choice(np.asarray(C.IDX_H), size=n_inh, replace=False)
    sgn[inh] = -1.0
    return pre, post, d, sgn


# ==================================================================
# Mreza
# ==================================================================
class Network:
    """
    specs: lista (seed, delta_r).  Batch elementi s istim seed-om dijele
    topologiju I niz trialova -- time su poredjenja preko delta_r uparena,
    sto je uslov da test kasnjenja nagrade uopste ima snagu.
    """

    def __init__(self, p: Params, specs, run_name: str = "run"):
        self.p = p
        self.specs = list(specs)
        self.B = len(self.specs)
        self.run_name = run_name
        self.seeds = np.array([s for s, _ in self.specs], dtype=np.int64)
        self.dr = np.array([d for _, d in self.specs], dtype=np.int64)
        self._build()

    # ---------------- konstrukcija ----------------
    def _build(self):
        p, B = self.p, self.B
        n_syn = C.N_SYN
        self.n_syn = n_syn

        pre = np.zeros((B, n_syn), dtype=np.int32)
        post = np.zeros((B, n_syn), dtype=np.int32)
        d = np.zeros((B, n_syn), dtype=np.int32)
        sgn = np.zeros((B, C.N))
        topo_cache = {}
        for b, (seed, _) in enumerate(self.specs):
            if seed not in topo_cache:
                topo_cache[seed] = build_topology(make_rng(seed, "topology"), p)
            pre[b], post[b], d[b], sgn[b] = topo_cache[seed]
        self.pre, self.post, self.d, self.sgn = pre, post, d, sgn

        # ravni indeksi: bincount je 1-D, pa se batch presavija u indeks
        self.post_flat = (post + (np.arange(B, dtype=np.int32)[:, None] * C.N)).ravel()
        self.sgn_syn = np.take_along_axis(sgn, pre.astype(np.int64), axis=1)
        self.plastic = self.sgn_syn > 0 if not p.plastic_inh else np.ones_like(self.sgn_syn, bool)
        self.is_out_syn = post >= C.SL_OUT.start

        # tabela indeksa za citanje bafera: idx_table[ptr] -> (B, n_syn)
        boff = (np.arange(B, dtype=np.int32)[:, None] * (p.d_max * C.N))
        self.idx_table = np.stack(
            [boff + ((ptr - d) % p.d_max) * C.N + pre for ptr in range(p.d_max)]
        ).astype(np.int32)

        self.lif_mask = np.zeros(C.N, dtype=bool)
        self.lif_mask[C.SL_H] = True
        self.lif_mask[C.SL_OUT] = True
        self.n_in = C.N_A + C.N_B

        self.tau_m = np.where(sgn > 0, p.tau_m_exc, p.tau_m_inh)
        self.dec_x = np.exp(-p.dt / p.tau_x)
        self.dec_y = np.exp(-p.dt / p.tau_y)
        self.dec_e = np.exp(-p.dt / p.tau_e)
        self.dec_phi = p.dt / p.tau_phi
        self.dec_rho = p.dt / p.tau_rho
        self.dec_rms = p.dt / p.tau_rms

        self.g_syn = float(p.g_syn)
        self.ring_len = int(self.dr.max()) + p.trial_len + 2
        self.reset()

    # ---------------- kalibracija pojacanja ----------------
    def calibrate_gain(self, n_trials: int = 6, lo: float = 0.05,
                       hi: float = 5000.0, iters: int = 18) -> float:
        """
        Bisekcija g_syn tako da H sjedne blizu ciljne stope.

        Bez ovoga je pocetna mreza ili nijema (tezine u [0, 0.3] daju EPSP reda
        0.02 mV, a prag je 13 mV iznad mirovanja) ili u rafalima.  U oba slucaja
        sve dalje mjeri kvar, ne ucenje -- i to je tacno onaj tihi kvar iz
        v3 §6.10 zbog kojeg homeostaza uopste postoji.  Deterministicno.
        """
        p = self.p
        target = p.h_rate_target_hz
        best = self.g_syn
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))
            self.reset()
            self.g_syn = mid
            r = self.run_trials(n_trials, "gaincal", learn=False, chunk=n_trials)
            hz = float(r["rate_h_hz"])
            best = mid
            if hz < target:
                lo = mid
            else:
                hi = mid
            if abs(hz - target) < 0.05 * target:
                break
        self.reset()
        self.g_syn = best
        return best

    def reset(self):
        p, B = self.p, self.B
        rng = make_rng(int(self.seeds[0]), "winit")
        self.w = np.zeros((B, self.n_syn))
        for b, (seed, _) in enumerate(self.specs):
            self.w[b] = make_rng(seed, "winit").uniform(0.0, p.w_init_hi, self.n_syn)
        self.e = np.zeros((B, self.n_syn))

        self.V = np.full((B, C.N), p.v_rest)
        self.s = np.zeros((B, C.N), dtype=bool)
        self.x = np.zeros((B, C.N))
        self.y = np.zeros((B, C.N))
        self.refrac = np.zeros((B, C.N), dtype=np.int32)
        self.theta = np.full((B, C.N), p.theta_0)
        self.rho = np.zeros((B, C.N))

        self.s_hist = np.zeros((B, p.d_max, C.N), dtype=bool)
        self.x_hist = np.zeros((B, p.d_max, C.N))

        F = p.n_features
        self.w_v = np.zeros((B, F))
        self.z = np.zeros((B, F))
        self.phi_prev = np.zeros((B, F))
        self.V_prev = np.zeros(B)
        self.lp_a = np.zeros(B)
        self.lp_b = np.zeros(B)
        self.phi_mu = np.zeros((B, F))
        self.phi_ms = np.ones((B, F))

        self.ring_r = np.zeros((B, self.ring_len))
        self.ring_o = np.zeros((B, self.ring_len))
        self.ms_s = np.zeros(B) + 1e-4
        self.ms_o = np.zeros(B) + 1e-4
        self.ms_f = np.zeros(B) + 1e-4

        self.n_out = np.zeros(B, dtype=np.int64)
        self.last_n_out = np.zeros(B, dtype=np.int64)
        self._peer_rng = [make_rng(int(sd), "peer") for sd in self.seeds]
        self.t = 0

        # --- predracunato za brzinu (v3 §12.4: masina je overhead-dominirana) ---
        c = np.linspace(0, p.trial_len, p.n_rbf)
        wid = max(p.trial_len / max(p.n_rbf - 1, 1), 1.0)
        tt = np.arange(p.trial_len)[:, None]
        self._rbf_tab = np.exp(-0.5 * ((tt - c[None, :]) / wid) ** 2)  # (trial_len, n_rbf)

        F = p.n_features
        self._phi = np.zeros((B, F))
        self._phi[:, -1] = 1.0                       # bias, nikad se ne mijenja
        self._sl_ab = slice(0, 2)
        self._sl_ho = slice(2, 2 + C.N_H + C.N_OUT)
        self._sl_rbf = slice(2 + C.N_H + C.N_OUT, F - 1)
        self.lp_ho = np.zeros((B, C.N_H + C.N_OUT))
        self._sl_lif = slice(C.SL_H.start, C.SL_OUT.stop)
        self._phi_sd = np.ones((B, F))

        # per-neuron granice pojasa: dva razlicita pojasa (H i OUT) bez
        # Python petlje po populacijama
        self._rho_lo = np.zeros(C.N)
        self._rho_hi = np.full(C.N, np.inf)
        self._rho_lo[C.SL_H], self._rho_hi[C.SL_H] = p.rho_min_h, p.rho_max_h
        self._rho_lo[C.SL_OUT], self._rho_hi[C.SL_OUT] = p.rho_min, p.rho_max
        self._plastic_f = self.plastic.astype(np.float64)
        self._out_syn_f = self.is_out_syn.astype(np.float64)
        self._spk_acc = np.zeros((B, C.N))

        # koji kanali su uopste ukljuceni -- iskljuceni se ne racunaju
        self.use_s = p.w_s > 0.0
        self.use_o = p.w_o > 0.0
        self.need_critic = p.w_f > 0.0
        self._rms_s = np.sqrt(np.maximum(self.ms_s, p.eps))
        self._rms_o = np.sqrt(np.maximum(self.ms_o, p.eps))
        self._rms_f = np.sqrt(np.maximum(self.ms_f, p.eps))
        self.eta_eff = p.eta * p.eta_eff_scale
        self._mod_const = np.zeros(B)

    # ---------------- obiljezja kriticara ----------------
    def _features(self, tphase: int) -> np.ndarray:
        """Puni predalociran bafer umjesto concatenate/tile po koraku."""
        phi = self._phi
        phi[:, 0] = self.lp_a
        phi[:, 1] = self.lp_b
        phi[:, self._sl_ho] = self.lp_ho
        if self.p.use_rbf:
            phi[:, self._sl_rbf] = self._rbf_tab[tphase]
        return phi

    # ---------------- jedan korak ----------------
    def step(self, ext: np.ndarray, tphase: int, learn: bool,
             kind: np.ndarray | None, end_of_trial: bool,
             mod_override: float | None = None,
             force: np.ndarray | None = None):
        p = self.p
        ptr = self.t % p.d_max

        # ---- (a) dolasci i ulazna struja ---------------------------------
        #      CITA bafere.  Pisanje je u koraku (d).  Red je nosiv -- §12.2
        idx = self.idx_table[ptr]
        arr = self.s_hist.ravel()[idx.ravel()].reshape(self.B, self.n_syn)
        xd = self.x_hist.ravel()[idx.ravel()].reshape(self.B, self.n_syn)

        contrib = self.sgn_syn * self.w * arr
        I = np.bincount(self.post_flat, weights=contrib.ravel(),
                        minlength=self.B * C.N).reshape(self.B, C.N) * self.g_syn

        # ---- (b) LIF integracija ----------------------------------------
        self.V += (p.dt / self.tau_m) * (-(self.V - p.v_rest) + I)
        self.V[self.refrac > 0] = p.v_reset

        # ---- (c) prag i reset -------------------------------------------
        s_lif = (self.V > self.theta) & (self.refrac == 0) & self.lif_mask[None, :]
        self.V[s_lif] = p.v_reset
        self.refrac[s_lif] = p.t_refrac
        self.refrac = np.maximum(self.refrac - 1, 0)      # POD NA NULI -- §6.3
        s = s_lif
        s[:, :self.n_in] = ext                            # A i B su forsirani izvori
        if force is not None:
            # Test-hook: forsira spike na bilo kom neuronu.  Koristi ga mjerenje
            # empirijskog STDP prozora (v3 §7.3) -- jedina dijagnostika koja
            # hvata gresku u knjigovodstvu tragova, i mora ici kroz PRAVI kod.
            s |= force
        self.s = s

        if tphase >= p.trial_len - p.resp_win:
            self.n_out += s[:, C.SL_OUT].sum(axis=1)

        # ---- (d) eligibility --------------------------------------------
        s_post = s.ravel()[self.post_flat].reshape(self.B, self.n_syn)
        y_post = self.y.ravel()[self.post_flat].reshape(self.B, self.n_syn)
        # Konvencija na Dt = 0  (v3 §6.4).
        #   xd = x_i(t - d) i VEC ukljucuje spike poslan u t-d, jer se x_hist
        #   pise poslije x += s.  Dakle na Dt = 0 je xd = 1 i opcija B je
        #   naprosto `xd`, a opcija A trazi ODUZIMANJE tekuceg dolaska.
        #   (Prva verzija je ovdje sabirala arr i dobijala +2 na Dt = 0 --
        #   uhvacena empirijskim STDP prozorom, v3 §7.3.)
        xd_ltp = xd if p.dt0_mode == "LTP" else xd - arr
        self.e *= self.dec_e
        self.e += p.a_plus * xd_ltp * s_post
        self.e -= p.a_minus * arr * y_post

        self.x = self.x * self.dec_x + s                  # POSLIJE e -- §12.2
        self.y = self.y * self.dec_y + s
        self.s_hist[:, ptr, :] = s
        self.x_hist[:, ptr, :] = self.x

        # ---- (e) nagrada: sve cetiri celije, odgodjena za dr -------------
        if end_of_trial and kind is not None:
            fired = self.n_out >= p.k_crit
            correct = fired == (kind == SIGNAL)
            val = np.where(correct, 1.0, -1.0)
            slot = (self.t + self.dr) % self.ring_len
            bi = np.arange(self.B)
            self.ring_r[bi, slot] = val
            truth = np.where(kind == SIGNAL, 1.0, -1.0)
            # Jedan RNG PO BATCH ELEMENTU, izveden iz NJEGOVOG seed-a.
            # Prva verzija je crpila iz seeds[0] za sve, pa su razlicita sjemena
            # dijelila isti peer sum -- a sjemena moraju biti nezavisni replikati,
            # inace pure error u lack-of-fit testu (v3 §8.3) nije pure.
            # Isti seed i dalje daje isti niz, pa poredjenje preko Dr ostaje upareno.
            rr = np.array([g.random() for g in self._peer_rng])
            self.ring_o[bi, slot] = np.where(rr < p.q, truth, -truth)
            # sacuvaj prije nulovanja: evaluacija cita OVO, ne self.n_out --
            # inace bi joj nedostajao zadnji korak trial-a (nagrada ga vidi,
            # jer se n_out inkrementira u bloku (c) prije ovog bloka)
            self.last_n_out = self.n_out.copy()
            self.n_out[:] = 0

        cur = self.t % self.ring_len
        r_s_raw = self.ring_r[:, cur].copy()
        r_o_raw = self.ring_o[:, cur].copy()
        self.ring_r[:, cur] = 0.0
        self.ring_o[:, cur] = 0.0

        # ---- (f) kriticar sa stanjem -------------------------------------
        # Kad je w_f = 0 (npr. cist R-STDP u R2), r_f se nigdje ne koristi, pa
        # se cijeli blok preskace.  ~30 NumPy poziva manje po koraku.
        r_f_raw = 0.0
        if self.need_critic:
            self.lp_a += (s[:, C.SL_A].sum(axis=1) - self.lp_a) * self.dec_phi
            self.lp_b += (s[:, C.SL_B].sum(axis=1) - self.lp_b) * self.dec_phi
            self.lp_ho += (s[:, self._sl_lif] - self.lp_ho) * self.dec_phi

            phi = self._features(tphase)
            if self.t % p.slow_every == 0:          # normalizator se mijenja sporo
                k = self.dec_phi * p.slow_every
                self.phi_mu += (phi - self.phi_mu) * k
                self.phi_ms += (phi * phi - self.phi_ms) * k
                np.sqrt(np.maximum(self.phi_ms - self.phi_mu ** 2, 1e-8),
                        out=self._phi_sd)
            phi_z = (phi - self.phi_mu) / self._phi_sd

            # eksplicitna suma, NE BLAS -- visenitni BLAS razbija bit-identicnost (§12.1)
            V_now = np.sum(self.w_v * phi_z, axis=1)
            delta = r_s_raw / self._rms_s + p.gamma * V_now - self.V_prev
            self.z *= p.gamma * p.lam
            self.z += self.phi_prev
            znorm = np.sum(self.z * self.z, axis=1) + p.eps
            self.w_v += ((p.alpha_critic / znorm) * delta)[:, None] * self.z
            self.V_prev = V_now
            self.phi_prev[...] = phi_z
            r_f_raw = delta

        # ---- (g) normalizacija kanala i modulator ------------------------
        # RMS ide preko SVIH koraka, ukljucujuci nule.  Time se automatski
        # kompenzuje duty cycle: r_s je nenulti 1/500 koraka pa mu je RMS ~0.045,
        # a dijeljenje njime vraca kanale na uporedive doprinose.  (v3 §6.7)
        # ms_s se azurira i kad je w_s = 0, jer KRITICAR uci V iz r_s bez
        # obzira na w_s (v3 §6.5: r_f nije nezavisan kanal nego izveden iz r_s).
        # Bez ovoga na vrhu (0,0,1) ostaje pocetnih 1e-4, pa r_s/rms_s eksplodira
        # na ~100x i TD greska postaje besmislena.
        if self.use_s or self.need_critic:
            self.ms_s += (r_s_raw * r_s_raw - self.ms_s) * self.dec_rms
        if self.use_o:
            self.ms_o += (r_o_raw * r_o_raw - self.ms_o) * self.dec_rms
        if self.need_critic:
            self.ms_f += (r_f_raw * r_f_raw - self.ms_f) * self.dec_rms
        if self.t % p.slow_every == 0:              # sqrt tek svakih slow_every
            self._rms_s = np.sqrt(np.maximum(self.ms_s, p.eps))
            self._rms_o = np.sqrt(np.maximum(self.ms_o, p.eps))
            self._rms_f = np.sqrt(np.maximum(self.ms_f, p.eps))

        M = 0.0
        if self.use_s:
            M = M + p.w_s * (r_s_raw / self._rms_s)
        if self.use_o:
            M = M + p.w_o * (r_o_raw / self._rms_o)
        if self.need_critic:
            M = M + p.w_f * (r_f_raw / self._rms_f)
        if mod_override is not None:
            if self._mod_const[0] != mod_override:
                self._mod_const[:] = float(mod_override)
            M = self._mod_const
        elif not np.ndim(M):
            M = np.zeros(self.B)

        # ---- (h) plasticnost ---------------------------------------------
        # Kad je M svuda nula, w += eta*M*e ne mijenja nista.  Bez kriticara je
        # M nenulto samo u trenutku isporuke nagrade -- jednom u 500 koraka --
        # pa se najskuplja operacija nad (B, n_syn) preskace 499 puta od 500.
        if learn and (self.need_critic or mod_override is not None or M.any()):
            self.w += (self.eta_eff * M)[:, None] * self.e * self._plastic_f
            np.clip(self.w, 0.0, p.w_max, out=self.w)

        # ---- (i) homeostaza: pojas, ne setpoint --------------------------
        #      Konstanta je 2000 ms, pa korak od slow_every*dt = 20 ms ne mijenja
        #      ponasanje a stedi ~12 poziva po koraku.
        self._spk_acc += s
        if p.homeostasis and self.t % p.slow_every == 0:
            kk = self.dec_rho * p.slow_every
            self.rho += (self._spk_acc / p.slow_every - self.rho) * kk
            if p.theta_vector:
                lo = self.rho < self._rho_lo
                hi = self.rho > self._rho_hi
                self.theta += (p.beta_up * p.slow_every) * hi
                self.theta -= (p.beta_dn * p.slow_every) * lo
            else:
                # skalarni prag po populaciji -- kontrola za PR test (v3 §10 R2)
                for sl, lo_v, hi_v in ((C.SL_H, p.rho_min_h, p.rho_max_h),
                                       (C.SL_OUT, p.rho_min, p.rho_max)):
                    m = self.rho[:, sl].mean(axis=1)
                    adj = np.where(m < lo_v, -p.beta_dn, np.where(m > hi_v, p.beta_up, 0.0))
                    self.theta[:, sl] += adj[:, None] * p.slow_every
            np.clip(self.theta, p.theta_lo, p.theta_hi, out=self.theta)
            self._spk_acc[...] = 0.0

        if learn and p.homeostasis and self.t % p.scale_every == 0:
            tot = np.bincount(self.post_flat, weights=self.w.ravel(),
                              minlength=self.B * C.N).reshape(self.B, C.N)
            tp = tot.ravel()[self.post_flat].reshape(self.B, self.n_syn)
            fac = (p.w_target / np.maximum(tp, p.eps)) ** p.rho_scale
            self.w *= np.where(self.is_out_syn, fac, 1.0)
            np.clip(self.w, 0.0, p.w_max, out=self.w)

        self.t += 1
        return M

    # ---------------- vrtjeti trialove ----------------
    def run_trials(self, n_trials: int, rng_name: str, learn: bool = True,
                   collect: bool = False, shuffle: bool = False,
                   mod_override: float | None = None, chunk: int = 32) -> dict:
        p = self.p
        rec_counts, rec_kind, mod_abs, mod_n = [], [], 0.0, 0
        spk = np.zeros((self.B, C.N))
        n_steps = 0

        # jedan niz trialova po seed-u; batch elementi s istim seed-om ga dijele
        uniq = sorted(set(int(s) for s in self.seeds))
        smap = {s: i for i, s in enumerate(uniq)}
        bidx = np.array([smap[int(s)] for s in self.seeds])
        rngs = {s: make_rng(s, rng_name) for s in uniq}

        done = 0
        while done < n_trials:
            m = min(chunk, n_trials - done)
            rasters, kinds = [], []
            for s in uniq:
                tr = make_trials(rngs[s], p, m)
                r = build_raster(rngs[s], p, tr["t_a"], tr["t_b"])
                if shuffle:
                    r = shuffle_raster(rngs[s], r)
                rasters.append(r)
                kinds.append(tr["kind"])
            R = np.stack(rasters)[bidx]                   # (B, m, trial_len, n_in)
            K = np.stack(kinds)[bidx]                     # (B, m)

            for j in range(m):
                self.n_out[:] = 0
                for tp in range(p.trial_len):
                    end = tp == p.trial_len - 1
                    M = self.step(R[:, j, tp, :], tp, learn, K[:, j], end,
                                  mod_override=mod_override)
                    if end and collect:
                        rec_counts.append(self.last_n_out.copy())
                        rec_kind.append(K[:, j].copy())
                    spk += self.s
                    n_steps += 1
                    if M is not None and np.ndim(M):
                        mod_abs += float(np.abs(M).mean())
                        mod_n += 1
            done += m

        secs = max(n_steps, 1) * p.dt / 1000.0
        out = dict(mean_abs_M=mod_abs / max(mod_n, 1),
                   rate_h_hz=float(spk[:, C.SL_H].mean() / secs),
                   rate_out_hz=float(spk[:, C.SL_OUT].mean() / secs),
                   rate_out_per_neuron_hz=(spk[:, C.SL_OUT].mean(axis=0) / secs))
        if collect:
            out["counts"] = np.array(rec_counts).T        # (B, n_trials)
            out["kind"] = np.array(rec_kind).T
        return out

    # ---------------- pomocno ----------------
    def out_rates_hz(self) -> np.ndarray:
        return self.rho[:, C.SL_OUT] * 1000.0 / self.p.dt

    def snapshot(self) -> dict:
        return dict(w=self.w.copy(), theta=self.theta.copy(),
                    rho=self.rho.copy(), w_v=self.w_v.copy())
