"""
Rota (genom) temsili ve uygunluk (fitness) hesabı.

Genom: n adet (uzunluk, dönüş açısı) çifti -> başlangıç (0,0), ilk yön 0.
Dönme/öteleme değişmezliği sayesinde başlangıç noktası ve yön sabitlenebilir.
Rota poligonal (kırık çizgi) olduğu için, herhangi bir ön-parçanın (prefix)
konveks zarfı köşe noktalarıyla belirlenir; bu da sığma testini kesin yapar.
"""
from __future__ import annotations

import numpy as np

from .shapes import Forest


def genome_to_vertices(genome: np.ndarray) -> np.ndarray:
    g = np.asarray(genome, float).reshape(-1, 2)
    lengths = np.abs(g[:, 0])
    headings = np.concatenate([[0.0], np.cumsum(g[:-1, 1])])
    steps = np.stack([lengths * np.cos(headings), lengths * np.sin(headings)], axis=1)
    return np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])


def vertices_to_genome(V: np.ndarray) -> np.ndarray:
    """genome_to_vertices'in tersi (sıfır uzunluklu segmentler atılır)."""
    V = np.asarray(V, float)
    d = np.diff(V, axis=0)
    L = np.linalg.norm(d, axis=1)
    d, L = d[L > 1e-12], L[L > 1e-12]
    h = np.arctan2(d[:, 1], d[:, 0])
    turns = np.append(np.diff(h), 0.0)
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    return np.stack([L, turns], axis=1).ravel()


def resample_genome(genome, n_segments: int, trim_to: float | None = None) -> np.ndarray:
    """Rotayı (isteğe bağlı olarak trim_to uzunluğunda keserek) n_segments segmente çıkar.

    En uzun segment tekrar tekrar ikiye bölünür; şekil aynen korunur.
    Sıcak başlatmada (warm start) kaba çözümü daha ince bir genoma taşımak için.
    """
    V = genome_to_vertices(genome)
    if trim_to is not None:
        V = prefix(V, trim_to)
    pts = [p for p in V]
    while len(pts) - 1 < n_segments:
        seg = [np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])]
        i = int(np.argmax(seg))
        pts.insert(i + 1, 0.5 * (pts[i] + pts[i + 1]))
    g = vertices_to_genome(np.array(pts))
    while len(g) // 2 < n_segments:                    # sıfır segmentler atıldıysa doldur
        g = np.append(g, [0.0, 0.0])
    if len(g) // 2 > n_segments:                      # fazla segment: kuyruğu kes
        g = g[: 2 * n_segments]
    return g


def path_length(V: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(V, axis=0), axis=1).sum())


def prefix(V: np.ndarray, L: float) -> np.ndarray:
    """Rotanın ilk L uzunluğundaki kısmının köşe noktaları."""
    seg = np.linalg.norm(np.diff(V, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if L >= cum[-1]:
        return V
    k = int(np.searchsorted(cum, L, side="right"))      # V[k-1]..V[k] aralığında
    a, b = V[k - 1], V[k]
    t = (L - cum[k - 1]) / seg[k - 1] if seg[k - 1] > 0 else 0.0
    return np.vstack([V[:k], a + t * (b - a)])


def _prefix_batch(V: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    """Birden çok prefix'i aynı köşe sayısına (son noktayı tekrarlayarak) doldur."""
    k = len(V)
    out = np.empty((len(Ls), k, 2))
    for i, L in enumerate(Ls):
        p = prefix(V, L)
        out[i, :len(p)] = p
        out[i, len(p):] = p[-1]
    return out


def escape_length(forest: Forest, V: np.ndarray, n_probe: int = 16, rounds: int = 3, **mm_kw):
    """En kötü durum kaçış mesafesi E(γ).

    "prefix(L) sığıyor mu?" L'de monotondur, bu yüzden [lo,hi] aralığında
    n_probe noktayı TOPLU test edip aralığı daraltırız (paralel ikili arama).
    Sığma testi kesin (branch-and-bound) olduğundan dönen değer gerçek
    en kötü durum mesafesinin kanıtlanmış bir ÜST SINIRIDIR.
    Hassasiyet ≈ toplam / (n_probe+1)^rounds.

    Returns (E, escapes, full_margin)
      escapes=False ise rota tamamen ormanın içine sığıyor demektir
      (bazı başlangıçlarda yürüyüşçü hiç çıkamaz).
    """
    total = path_length(V)
    full_margin = forest.max_margin(V, **mm_kw)[0]
    if full_margin > 0 or forest.fits_batch(V[None])[0]:
        return total, False, max(full_margin, 0.0)
    lo, hi = 0.0, total               # prefix(lo) sığar, prefix(hi) sığmaz
    for _ in range(rounds):
        Ls = np.linspace(lo, hi, n_probe + 2)[1:-1]
        fits = forest.fits_batch(_prefix_batch(V, Ls))
        bad = np.nonzero(~fits)[0]
        if len(bad):
            j = bad[0]
            hi = Ls[j]
            lo = Ls[j - 1] if j > 0 else lo
        else:
            lo = Ls[-1]
    return hi, True, full_margin


def fitness(forest: Forest, genome: np.ndarray, **mm_kw) -> float:
    """Küçük olan daha iyi.

    Rota kaçış garantiliyorsa -> en kötü durum kaçış mesafesi.
    Aksi halde -> ceza: toplam uzunluk + çap + 10·margin
    (margin küçüldükçe ceza azalır, böylece evrim kaçışa doğru yönlenir).
    """
    V = genome_to_vertices(genome)
    E, ok, m = escape_length(forest, V, **mm_kw)
    if ok:
        return E
    return path_length(V) + forest.diameter + 10.0 * m


def simulate_walk(forest: Forest, V: np.ndarray, start: np.ndarray, theta: float,
                  extend: float = 10.0):
    """Rotayı (start, θ) yerleşiminde yürüt; çıkış noktasına kadar olan izi döndür."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    W = V @ R.T + start
    # rota bittiğinde düz devam et
    d = W[-1] - W[-2] if len(W) > 1 else np.array([c, s])
    nd = np.linalg.norm(d)
    if nd > 0:
        W = np.vstack([W, W[-1] + d / nd * extend * forest.diameter])
    trace, dist = [W[0]], 0.0
    for a, b in zip(W[:-1], W[1:]):
        t = forest.exit_param(a, b)
        if t is not None:
            p = a + t * (b - a)
            trace.append(p)
            return np.array(trace), dist + np.linalg.norm(p - a)
        trace.append(b)
        dist += np.linalg.norm(b - a)
    return np.array(trace), dist
