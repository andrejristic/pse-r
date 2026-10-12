# -*- coding: utf-8 -*-
"""
Deterministicki RNG i hesiranje stanja.   v3 §12.1

Zahtjev iz nacrta: isti seed -> bit-identican rezultat na obje masine
(Surface 4 GB i Dell 16 GB).  Ako se pola dizajna odvrti na jednoj a pola na
drugoj s razlicitim rezultatima, masina postaje konfund u dizajnu koji je
cijeli o kontrastima.

Zato:
  * jedan Generator po (seed, ime faze) -- nikad globalni np.random
  * zlib.crc32 umjesto ugradjenog hash() (koji je randomizovan po procesu)
  * hash stanja se poredi izmedju masina preko `pse-rstdp selftest`
"""
from __future__ import annotations

import hashlib
import zlib

import numpy as np


def stream_id(name: str) -> int:
    """Stabilan cijeli broj iz imena toka.  CRC32 je isti u svakom procesu."""
    return int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


def make_rng(seed: int, name: str) -> np.random.Generator:
    """
    Nezavisan tok za (seed, name).  Dvije faze s istim seed-om ne dijele
    stanje, pa dodavanje faze ne pomjera brojeve u postojecim fazama.
    """
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(stream_id(name),))
    return np.random.Generator(np.random.PCG64(ss))


def _feed(h: "hashlib._Hash", a) -> None:
    arr = np.ascontiguousarray(a)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    if arr.dtype.kind == "f":
        # Kvantizacija na 1e-12 relativno: stiti od zadnjeg bita razlike u
        # transcendentnim funkcijama izmedju verzija libm, a i dalje hvata
        # svaku stvarnu numericku divergenciju.
        h.update(np.round(arr.astype(np.float64), 12).tobytes())
    else:
        h.update(arr.tobytes())


def state_hash(**arrays) -> str:
    """Stabilan hash skupa nizova.  Redoslijed kljuceva se sortira."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode("utf-8"))
        _feed(h, arrays[k])
    return h.hexdigest()[:32]
