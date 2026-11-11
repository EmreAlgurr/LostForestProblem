"""
Evrimsel arama: kendinden uyarlamalı Diferansiyel Evrim (jDE + current-to-pbest).

Her "ajan" bir rotadır (genom = n x (uzunluk, dönüş açısı)).
Uygunluk = en kötü durum kaçış mesafesi (küçük = iyi), bkz. path.fitness.

Neden DE?
  * Sürekli, türevsiz, çok-modlu bir min-max problemi.
  * Fark vektörleri popülasyonun geometrisine kendiliğinden uyum sağlar
    (CMA-ES'e benzer davranış, çok daha basit kod).
  * F ve CR parametreleri her ajan için ayrıca evrimleşir (jDE).
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np

from .shapes import make_forest, Forest
from .path import fitness, genome_to_vertices, escape_length, path_length, resample_genome

# ------------------------------------------------------------ paralel değerlendirme
_FOREST: Forest | None = None
_MM_KW: dict = {}


def _init_worker(shape, shape_kw, mm_kw):
    global _FOREST, _MM_KW
    _FOREST = make_forest(shape, **shape_kw)
    _MM_KW = mm_kw


def _eval(genome):
    return fitness(_FOREST, genome, **_MM_KW)


def min_width(forest: Forest) -> float:
    if hasattr(forest, "N"):
        return float(min(bi - (forest.V @ n).min() for n, bi in zip(forest.N, forest.b)))
    return forest.diameter


@dataclass
class Config:
    shape: str = "square"
    shape_kw: dict = field(default_factory=dict)
    segments: int = 6
    pop: int = 60
    gens: int = 300
    p_best: float = 0.15          # current-to-pbest için en iyi oran
    seed: int | None = None
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    patience: int = 120           # bu kadar nesil iyileşme yoksa yeniden başlat (en iyiyi koru)
    n_grid: int = 120             # evrim sırasında açı ızgarası
    rounds: int = 4               # kaçış mesafesi arama turu (hassasiyet ≈ uzunluk/17^rounds)
    init_genome: list | None = None   # sıcak başlatma: önceki bir çözüm (segment sayısına uyarlanır)
    init_frac: float = 0.5        # popülasyonun ne kadarı sıcak başlangıcın etrafında
    polish: int = 0               # sonda (1,λ)-ES yerel cilalama iterasyonu
    out_dir: str = "results"
    verbose: bool = True


@dataclass
class Result:
    shape: str
    genome: list
    vertices: list
    escape_length: float
    path_length: float
    diameter: float
    known_best: float | None
    known_note: str
    history: list
    seconds: float
    config: dict


class Evolver:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.forest = make_forest(cfg.shape, **cfg.shape_kw)
        self.max_len = min(self.forest.diameter, 2.5 * min_width(self.forest))
        n = cfg.segments
        self.lo = np.tile([0.0, -math.pi], n)
        self.hi = np.tile([self.max_len, math.pi], n)
        self.mm_kw = dict(n_grid=cfg.n_grid, rounds=cfg.rounds)
        self.pool = None

    # -------------------------------------------------------------- helpers
    def _random_genomes(self, k):
        g = self.rng.uniform(self.lo, self.hi, size=(k, len(self.lo)))
        # başlangıçta toplam uzunluğu makul tut: ~ 0.5..2 x çap
        L = g[:, 0::2]
        target = self.rng.uniform(0.5, 2.0, size=(k, 1)) * self.forest.diameter
        g[:, 0::2] = L / L.sum(1, keepdims=True) * target
        return g

    def _repair(self, g):
        g[..., 0::2] = np.clip(g[..., 0::2], 0.0, self.max_len)
        g[..., 1::2] = (g[..., 1::2] + math.pi) % (2 * math.pi) - math.pi
        return g

    def _evaluate(self, G):
        if self.pool is not None:
            return np.array(list(self.pool.map(_eval, list(G), chunksize=max(1, len(G) // (4 * self.cfg.workers)))))
        return np.array([fitness(self.forest, g, **self.mm_kw) for g in G])

    # ------------------------------------------------------------------ run
    def run(self, callback=None) -> Result:
        cfg = self.cfg
        t0 = time.time()
        if cfg.workers > 1:
            self.pool = ProcessPoolExecutor(cfg.workers, initializer=_init_worker,
                                            initargs=(cfg.shape, cfg.shape_kw, self.mm_kw))
        try:
            return self._run(t0, callback)
        finally:
            if self.pool is not None:
                self.pool.shutdown()

    def _run(self, t0, callback):
        cfg, rng = self.cfg, self.rng
        NP, D = cfg.pop, len(self.lo)
        X = self._random_genomes(NP)
        if cfg.init_genome is not None:
            g0 = np.asarray(cfg.init_genome, float)
            V0 = genome_to_vertices(g0)
            E0, ok0, _ = escape_length(self.forest, V0, **self.mm_kw)
            g0 = resample_genome(g0, cfg.segments, trim_to=E0 if ok0 else None)
            k = max(1, int(cfg.init_frac * NP))
            X[0] = g0
            sig = rng.uniform(0.002, 0.03, (k - 1, 1))
            X[1:k] = g0 + rng.normal(0, 1, (k - 1, D)) * sig * (self.hi - self.lo)
            X = self._repair(X)
            if cfg.verbose:
                print(f"  sıcak başlatma: E0 = {E0:.5f}, {cfg.segments} segmente bölündü", flush=True)
        fit = self._evaluate(X)
        F = np.full(NP, 0.5)
        CR = np.full(NP, 0.9)
        best_i = fit.argmin()
        best_g, best_f = X[best_i].copy(), fit[best_i]
        history, stall = [], 0

        for gen in range(cfg.gens):
            # jDE: parametre öz-uyarlaması
            F_new = np.where(rng.random(NP) < 0.1, 0.1 + 0.9 * rng.random(NP), F)
            CR_new = np.where(rng.random(NP) < 0.1, rng.random(NP), CR)

            order = np.argsort(fit)
            n_pbest = max(2, int(cfg.p_best * NP))
            trials = np.empty_like(X)
            for i in range(NP):
                pb = X[order[rng.integers(n_pbest)]]
                r1, r2 = rng.choice([j for j in range(NP) if j != i], 2, replace=False)
                v = X[i] + F_new[i] * (pb - X[i]) + F_new[i] * (X[r1] - X[r2])
                mask = rng.random(D) < CR_new[i]
                mask[rng.integers(D)] = True
                trials[i] = np.where(mask, v, X[i])
            # küçük olasılıkla gauss mutasyonu (keşif)
            gm = rng.random((NP, D)) < 0.02
            trials += gm * rng.normal(0, 0.05, (NP, D)) * (self.hi - self.lo)
            trials = self._repair(trials)

            tf = self._evaluate(trials)
            win = tf <= fit
            X[win], fit[win] = trials[win], tf[win]
            F[win], CR[win] = F_new[win], CR_new[win]

            i = fit.argmin()
            if fit[i] < best_f - 1e-9:
                best_f, best_g, stall = fit[i], X[i].copy(), 0
            else:
                stall += 1
            history.append(float(best_f))

            if cfg.verbose and (gen % 10 == 0 or gen == cfg.gens - 1):
                kb = self.forest.known_best
                gap = f"  (bilinen {kb:.4f}, fark {best_f - kb:+.4f})" if kb else ""
                print(f"[{self.forest.name}] nesil {gen + 1:4d}/{cfg.gens}  "
                      f"en iyi E = {best_f:.5f}  medyan = {np.median(fit):.4f}{gap}", flush=True)
            if callback:
                callback(gen, best_g, best_f, X, fit)

            # durgunluk -> kısmi yeniden başlatma (en iyi %20 korunur)
            if stall >= cfg.patience:
                keep = np.argsort(fit)[: max(2, NP // 5)]
                newX = self._random_genomes(NP)
                newX[: len(keep)] = X[keep]
                # bir kısmını en iyinin etrafında pertürbe et
                k = NP // 3
                newX[len(keep):len(keep) + k] = self._repair(
                    best_g + rng.normal(0, 0.03, (k, D)) * (self.hi - self.lo))
                X = newX
                fit = self._evaluate(X)
                F[:], CR[:] = 0.5, 0.9
                stall = 0
                if cfg.verbose:
                    print("  -> durgunluk: popülasyon kısmen yeniden başlatıldı", flush=True)

        if cfg.polish > 0:
            best_g, best_f = self._polish(best_g, best_f, history)
        return self._finalize(best_g, history, t0)

    def _polish(self, g, f, history):
        """(1,λ)+elitizm evrim stratejisi, 1/5 kuralına benzer adım uyarlaması.

        DE yakınsadıktan sonra son ondalıkları sıkmak için kullanılır.
        """
        cfg, rng = self.cfg, self.rng
        lam = max(8, 2 * cfg.workers)
        sigma = 0.01
        D = len(g)
        for it in range(cfg.polish):
            # her seferinde rastgele bir koordinat alt kümesini oynat
            mask = rng.random((lam, D)) < rng.uniform(0.1, 1.0)
            C = self._repair(g + mask * rng.normal(0, sigma, (lam, D)) * (self.hi - self.lo))
            fc = self._evaluate(C)
            i = fc.argmin()
            if fc[i] < f - 1e-10:
                g, f = C[i].copy(), fc[i]
                sigma = min(sigma * 1.5, 0.05)
            else:
                sigma = max(sigma * 0.85, 1e-5)
            history.append(float(f))
            if cfg.verbose and it % 20 == 0:
                print(f"[{self.forest.name}] cilalama {it + 1:4d}/{cfg.polish}  E = {f:.5f}  σ = {sigma:.1e}",
                      flush=True)
        return g, f

    def _finalize(self, g, history, t0):
        V = genome_to_vertices(g)
        # hassas doğrulama: ince açı ızgarası ve daha çok arama turu
        E, ok, _ = escape_length(self.forest, V, rounds=5, n_grid=720, n_top=6, refine_levels=3)
        cfg = dict(asdict(self.cfg))
        res = Result(shape=self.cfg.shape, genome=g.tolist(), vertices=V.tolist(),
                     escape_length=float(E) if ok else float("inf"),
                     path_length=path_length(V), diameter=self.forest.diameter,
                     known_best=self.forest.known_best, known_note=self.forest.known_note,
                     history=history, seconds=time.time() - t0, config=cfg)
        return res


def save_result(res: Result, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{res.shape}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, ensure_ascii=False)
    return p


def load_result(path: str) -> Result:
    with open(path, encoding="utf-8") as f:
        return Result(**json.load(f))

