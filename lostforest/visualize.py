"""Görselleştirme: en iyi rota, kritik (en kötü) yerleşim, örnek yürüyüşler, evrim eğrisi."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .shapes import Forest
from .path import prefix, simulate_walk


def _place(V, theta, t):
    c, s = np.cos(theta), np.sin(theta)
    return V @ np.array([[c, -s], [s, c]]).T + t


def plot_result(forest: Forest, V: np.ndarray, E: float, history=None,
                n_random=40, seed=0, save_path=None, show=True):
    V = np.asarray(V, float)
    P = prefix(V, E)
    ncols = 4 if history is not None else 3
    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    kb = forest.known_best
    title = f"{forest.name}:  E = {E:.4f}   çap = {forest.diameter:.4f}"
    if kb:
        title += f"   bilinen en iyi = {kb:.4f}"
    fig.suptitle(title, fontsize=13)

    # 1) rota kendi çerçevesinde
    ax = axs[0]
    ax.plot(P[:, 0], P[:, 1], "-o", color="tab:green", ms=4)
    ax.plot(*P[0], "ks", ms=8, label="başlangıç")
    ax.set_title(f"Evrimleşmiş rota (uzunluk {E:.4f})")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend()

    # 2) kritik yerleşim: E'den hemen kısa prefix'in sığdığı (sınıra değen) konum
    ax = axs[1]
    B = forest.boundary()
    m, th, t = forest.max_margin(prefix(V, E * 0.999))
    W = _place(prefix(V, E), th, t)
    ax.fill(B[:, 0], B[:, 1], color="forestgreen", alpha=.15)
    ax.plot(B[:, 0], B[:, 1], color="darkgreen", lw=2)
    ax.plot(W[:, 0], W[:, 1], "-o", color="crimson", ms=4)
    ax.plot(*W[0], "ks", ms=8)
    ax.set_title("En kötü yerleşim (rota sınıra ancak sonunda değer)")
    ax.set_aspect("equal"); ax.axis("off")

    # 3) rastgele başlangıçlar
    ax = axs[2]
    rng = np.random.default_rng(seed)
    ax.fill(B[:, 0], B[:, 1], color="forestgreen", alpha=.15)
    ax.plot(B[:, 0], B[:, 1], color="darkgreen", lw=2)
    lo, hi = B.min(0), B.max(0)
    dists, cnt = [], 0
    while cnt < n_random:
        p = rng.uniform(lo, hi)
        if not forest.contains(p):
            continue
        tr, d = simulate_walk(forest, V, p, rng.uniform(0, 2 * np.pi))
        dists.append(d); cnt += 1
        ax.plot(tr[:, 0], tr[:, 1], lw=.8, alpha=.7)
        ax.plot(*tr[0], "k.", ms=3)
    ax.set_title(f"{n_random} rastgele başlangıç  (maks {max(dists):.3f} ≤ E)")
    ax.set_aspect("equal"); ax.axis("off")

    # 4) evrim eğrisi
    if history is not None:
        ax = axs[3]
        h = np.array(history)
        ax.plot(np.where(np.isfinite(h), h, np.nan), color="tab:blue")
        ax.axhline(forest.diameter, ls="--", color="gray", label="çap (düz çizgi)")
        if kb:
            ax.axhline(kb, ls=":", color="red", label="bilinen en iyi")
        ax.set_xlabel("nesil"); ax.set_ylabel("en iyi kaçış mesafesi E")
        lo_y = min(np.nanmin(h), kb or np.inf) * 0.97
        ax.set_ylim(lo_y, max(forest.diameter, np.nanmin(h)) * 1.25)
        ax.set_title("Evrim süreci"); ax.grid(alpha=.3); ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)

