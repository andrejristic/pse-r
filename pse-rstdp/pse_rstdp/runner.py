# -*- coding: utf-8 -*-
"""
Dnevnik, checkpoint i nastavljanje.   v3 §12.3

Razvoj i prvi sweep-ovi idu na laptopu na putovanju: zatvaranje poklopca,
pregrijavanje, prazna baterija.  Otud dva zahtjeva, oba nezavisna od masine:

  * inkrementalni JSONL -- svaki run se upisuje CIM zavrsi.  Prekid na 200. runu
    ostavlja 199 rezultata, ne nula.
  * nastavljanje -- vec zavrsene jedinice se preskacu po kljucu.

Uz to `measured.json` nosi velicine koje jedna faza PROIZVODI a sljedece
koriste (W_target iz R1, g_syn iz R1, A_minus iz R2) -- v3 §5.5.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np

from . import __version__
from .config import SCHEMA_VERSION, Params, Profile


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(type(o)))


class Journal:
    def __init__(self, outdir: str, profile: Profile | None = None):
        self.dir = os.path.abspath(outdir)
        os.makedirs(self.dir, exist_ok=True)
        self.profile = profile
        self._keys: dict[str, set] = {}
        self.measured = self._load_measured()

    # ---------------- meta ----------------
    def write_meta(self, params: Params, extra: dict | None = None) -> None:
        meta = dict(
            schema=SCHEMA_VERSION, package_version=__version__,
            numpy=np.__version__, python=sys.version.split()[0],
            platform=platform.platform(), machine=platform.machine(),
            profile=(self.profile.name if self.profile else None),
            profile_jobs=(self.profile.jobs if self.profile else None),
            profile_batch=(self.profile.batch if self.profile else None),
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            params=params.to_dict(),
        )
        if extra:
            meta.update(extra)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=_default)

    # ---------------- mjerene velicine ----------------
    def _mpath(self) -> str:
        return os.path.join(self.dir, "measured.json")

    def _load_measured(self) -> dict:
        try:
            with open(self._mpath()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def set_measured(self, **kw) -> None:
        self.measured.update(kw)
        with open(self._mpath(), "w") as f:
            json.dump(self.measured, f, indent=2, default=_default)

    def need(self, name: str, phase: str):
        """Velicina koju je trebalo da proizvede ranija faza."""
        if name not in self.measured:
            raise SystemExit(
                "nedostaje '%s' -- proizvodi je faza %s.  Pokreni je prvo "
                "(pse-rstdp run --phase %s)." % (name, phase, phase.lower()))
        return self.measured[name]

    # ---------------- zapisi ----------------
    def path(self, phase: str) -> str:
        return os.path.join(self.dir, "%s.jsonl" % phase.lower())

    def keys(self, phase: str) -> set:
        if phase not in self._keys:
            ks = set()
            try:
                with open(self.path(phase)) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ks.add(json.loads(line).get("key"))
                        except ValueError:
                            pass
            except OSError:
                pass
            self._keys[phase] = ks
        return self._keys[phase]

    def done(self, phase: str, key: str) -> bool:
        return key in self.keys(phase)

    def write(self, phase: str, record: dict) -> None:
        record.setdefault("t_wall", time.time())
        with open(self.path(phase), "a") as f:
            f.write(json.dumps(record, default=_default) + "\n")
            f.flush()
            os.fsync(f.fileno())          # prezivi gasenje baterije
        if "key" in record:
            self.keys(phase).add(record["key"])

    def read(self, phase: str) -> list:
        out = []
        try:
            with open(self.path(phase)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
        except OSError:
            pass
        return out

    def reset_phase(self, phase: str) -> None:
        try:
            os.remove(self.path(phase))
        except OSError:
            pass
        self._keys.pop(phase, None)


def chunk_specs(specs, batch: int):
    """
    Grupise jedinice posla u batcheve, drzeci isti seed zajedno.

    Batch elementi s istim seed-om dijele topologiju I niz trialova, cime su
    poredjenja preko delta_r uparena -- uslov da test kasnjenja nagrade ima
    snagu (v3 §8.5).  Zato se grupise po seed-u, ne redom.
    """
    by_seed: dict[int, list] = {}
    for s in specs:
        by_seed.setdefault(s[0], []).append(s)
    out, cur = [], []
    for seed in sorted(by_seed):
        for item in by_seed[seed]:
            cur.append(item)
            if len(cur) >= batch:
                out.append(cur)
                cur = []
    if cur:
        out.append(cur)
    return out


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def s(self) -> float:
        return time.time() - self.t0
