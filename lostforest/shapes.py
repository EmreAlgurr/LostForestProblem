"""
Orman (forest) şekilleri.

Temel matematiksel gözlem
-------------------------
Yürüyüşçü başlangıç noktasını ve baktığı yönü bilmez, ama elindeki rotayı
(path) kendi yerel koordinat sisteminde uygular. Dolayısıyla rota, ormana
herhangi bir *katı hareketle* (dönme + öteleme) yerleştirilmiş olabilir.

    Rota γ'nın ilk L uzunluğundaki parçası (prefix) ormanın İÇİNE hiçbir
    dönme+öteleme ile sığmıyorsa, yürüyüşçü en geç L mesafede ormandan çıkar.

Bu yüzden en kötü durum kaçış mesafesi:

    E(γ) = min { L : γ[0, L] ormana sığmıyor }

Bir şeklin "sığma marjı" (margin), rotanın ormanın içine ne kadar boşlukla
yerleştirilebildiğinin ölçüsüdür:

    margin(θ) = max_t  min_i ( b_i - n_i · (R_θ v + t) )      (tüm v köşeleri)

margin > 0  -> rota içeride (sınıra değmeden) yerleştirilebilir -> kaçış YOK
margin <= 0 -> her yerleşimde sınıra ulaşılır                   -> kaçış VAR

Konveks orman için bu, (tx, ty, s) üzerinde küçük bir lineer programdır
(Chebyshev merkezi problemi). Normaller θ'dan bağımsız olduğu için LP'nin
köşe adaylarını (3'lü kısıt kombinasyonlarını) önceden ters çevirip tüm θ
değerleri için tek seferde vektörel çözüyoruz.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


class Forest:
    name: str = "forest"
    diameter: float = 1.0
    known_best: float | None = None   # literatürdeki en iyi bilinen değer
    known_note: str = ""

    def max_margin(self, pts: np.ndarray, **kw) -> tuple[float, float, np.ndarray]:
        """Tüm dönmeler üzerinden en büyük margin.

        Returns (margin, theta, translation)
        """
        raise NotImplementedError

    def max_margin_batch(self, P: np.ndarray, **kw):
        """P: (B,k,2) -> (margins (B,), thetas (B,), translations (B,2))"""
        raise NotImplementedError

    def boundary(self) -> np.ndarray:
        raise NotImplementedError

    def contains(self, p: np.ndarray, eps: float = 1e-12) -> bool:
        raise NotImplementedError

    def exit_param(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """a (içeride) -> b doğru parçası sınırı hangi t∈[0,1]'de keser."""
        raise NotImplementedError


def _rotations(thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.cos(thetas), np.sin(thetas)


class ConvexPolygonForest(Forest):
    def __init__(self, vertices, name="polygon", known_best=None, known_note=""):
        V = np.asarray(vertices, dtype=float)
        # saat yönünün tersine sırala
        area2 = np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
        if area2 < 0:
            V = V[::-1]
        self.V = V
        self.name = name
        self.known_best = known_best
        self.known_note = known_note
        E = np.roll(V, -1, axis=0) - V
        N = np.stack([E[:, 1], -E[:, 0]], axis=1)          # dış normaller
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        self.N = N
        self.b = np.einsum("ij,ij->i", N, V)                # n_i · x <= b_i
        m = len(V)
        self.diameter = max(np.linalg.norm(V[i] - V[j]) for i in range(m) for j in range(m))

        # LP: max s  s.t.  n_i·t + s <= c_i.  Köşe adayları = 3'lü kısıtlar.
        triples, invs = [], []
        for tri in itertools.combinations(range(m), 3):
            A = np.hstack([N[list(tri)], np.ones((3, 1))])
            if abs(np.linalg.det(A)) > 1e-9:
                triples.append(tri)
                invs.append(np.linalg.inv(A))
        self.triples = np.array(triples)                    # (K,3)
        self.invs = np.array(invs)                          # (K,3,3)

    # ------------------------------------------------------------------ LP
    def _margins(self, P: np.ndarray, thetas: np.ndarray):
        """Toplu (batch) margin hesabı.

        P: (B, k, 2) nokta kümeleri, thetas: (B, T) açılar
        Returns margins (B, T), translations (B, T, 2)
        """
        c, s = np.cos(thetas), np.sin(thetas)                       # (B,T)
        px, py = P[:, None, :, 0], P[:, None, :, 1]                 # (B,1,k)
        x = px * c[..., None] - py * s[..., None]                   # (B,T,k)
        y = px * s[..., None] + py * c[..., None]
        # destek fonksiyonu h_i(θ) = max_v n_i·R v  -> (B,T,m)
        proj = x[..., None] * self.N[:, 0] + y[..., None] * self.N[:, 1]   # (B,T,k,m)
        C = self.b - proj.max(axis=2)                               # (B,T,m)
        # LP köşe adayları: (B,T,K,3)
        sol = np.einsum("kij,btkj->btki", self.invs, C[..., self.triples])
        t_xy, s_val = sol[..., :2], sol[..., 2]
        lhs = t_xy @ self.N.T + s_val[..., None]                    # (B,T,K,m)
        feas = np.all(lhs <= C[:, :, None, :] + 1e-9, axis=-1)
        s_val = np.where(feas, s_val, -np.inf)
        k = np.argmax(s_val, axis=-1)                               # (B,T)
        m = np.take_along_axis(s_val, k[..., None], -1)[..., 0]
        t = np.take_along_axis(t_xy, k[..., None, None], 2)[:, :, 0]
        return m, t

    def max_margin_batch(self, P, n_grid=120, n_top=3, refine_levels=2):
        """P: (B,k,2). Her küme için tüm dönmeler üzerinden en büyük margin.

        Kaba ızgara + en iyi n_top adayın etrafında iki kademe inceltme.
        Returns margins (B,), thetas (B,), translations (B,2)
        """
        P = np.asarray(P, dtype=float)
        B = P.shape[0]
        grid = np.broadcast_to(np.linspace(0, 2 * np.pi, n_grid, endpoint=False), (B, n_grid))
        m, T = self._margins(P, grid)
        ar = np.arange(B)
        i = m.argmax(1)
        best_m, best_th, best_t = m[ar, i], grid[ar, i], T[ar, i]
        cand = np.take_along_axis(grid, np.argsort(m, 1)[:, -n_top:], 1)   # (B,n_top)
        step = 2 * np.pi / n_grid
        for _ in range(refine_levels):
            offs = np.linspace(-step, step, 15)
            th = (cand[..., None] + offs).reshape(B, -1)
            mm, TT = self._margins(P, th)
            i = mm.argmax(1)
            better = mm[ar, i] > best_m
            best_m = np.where(better, mm[ar, i], best_m)
            best_th = np.where(better, th[ar, i], best_th)
            best_t = np.where(better[:, None], TT[ar, i], best_t)
            mm = mm.reshape(B, n_top, -1)
            cand = np.take_along_axis(th.reshape(B, n_top, -1), mm.argmax(2)[..., None], 2)[..., 0]
            step /= 7
        return best_m, best_th, best_t

    def max_margin(self, pts, **kw):
        m, th, t = self.max_margin_batch(np.asarray(pts, float)[None], **kw)
        return float(m[0]), float(th[0]), t[0]

    def fits_batch(self, P, n_grid=72, min_h=1e-7, **_):
        """KESİN sığma testi (dal-sınır / branch-and-bound).

        margin(θ), ρ-Lipschitz'tir; ρ = noktaların bir merkezden en büyük uzaklığı
        (LP'nin dual değişkenlerinin toplamı 1 olduğundan |Δs| <= ρ|Δθ|).
        Böylece [θc-h, θc+h] aralığında margin <= margin(θc) + ρh garanti olur.
        Üst sınırı > 0 olan aralıklar 3'e bölünerek incelenir.

        Returns bool dizisi (B,):
          True  -> en az bir yerleşimde rota tamamen içeride (kaçış garanti DEĞİL)
          False -> her yerleşimde rota sınıra ulaşır (kaçış KANITLANDI)
        Belirsiz kalan (h < min_h) durumlar ihtiyatlı olarak True sayılır,
        bu nedenle hesaplanan kaçış mesafesi her zaman gerçek değerin ÜST SINIRIDIR.
        """
        P = np.asarray(P, float)
        B = len(P)
        cen = 0.5 * (P.min(1) + P.max(1))
        P = P - cen[:, None, :]                          # dönme merkezi = kutunun merkezi
        rho = np.linalg.norm(P, axis=2).max(1)           # (B,)
        h = np.pi / n_grid
        bi = np.repeat(np.arange(B), n_grid)
        th = np.tile(np.linspace(h, 2 * np.pi - h, n_grid), B)
        result = np.zeros(B, bool)
        undecided = np.ones(B, bool)
        while len(bi):
            m = np.empty(len(bi))
            for s in range(0, len(bi), 4096):           # bellek için parçala
                sl = slice(s, s + 4096)
                m[sl] = self._margins(P[bi[sl]], th[sl, None])[0][:, 0]
            pos = np.unique(bi[m > 0])
            result[pos] = True
            undecided[pos] = False
            keep = (m + rho[bi] * h > 0) & undecided[bi]
            bi, th = bi[keep], th[keep]
            if h < min_h:
                result[np.unique(bi)] = True            # ihtiyatlı karar
                break
            h /= 3
            bi = np.repeat(bi, 3)
            th = (th[:, None] + np.array([-2 * h, 0, 2 * h])).ravel()
        return result

    # ------------------------------------------------------------ geometry
    def boundary(self):
        return np.vstack([self.V, self.V[:1]])

    def contains(self, p, eps=1e-12):
        return bool(np.all(self.N @ p <= self.b + eps))

    def exit_param(self, a, b):
        d = b - a
        best = None
        for n, bi in zip(self.N, self.b):
            nd = n @ d
            if nd > 1e-15:
                t = (bi - n @ a) / nd
                if 0 <= t <= 1 and (best is None or t < best):
                    best = t
        return best


class CompositePolygonForest(Forest):
    """Konkav / Non-konveks çokgenler için konveks ayrışım (decomposition) destekli orman.
    
    Çokgen F, konveks parçaların birleşimidir: F = C_1 ∪ C_2 ∪ ... ∪ C_K
    Bir rota parçasının sığması:
      1. Herhangi bir konveks parça C_k içine sığıyorsa (kesin LP testi) -> içeride
      2. Parçalar arasındaki birleşimde kalıyorsa (vektörize ızgara + sınır kesişim testi) -> içeride
    """
    def __init__(self, boundary_vertices, pieces, name="composite", known_best=None, known_note=""):
        V = np.asarray(boundary_vertices, dtype=float)
        # Saat yönünün tersine sıralama kontrolü
        area2 = np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1])
        if area2 < 0:
            V = V[::-1]
        self.V = V
        self.pieces = pieces
        self.name = name
        self.known_best = known_best
        self.known_note = known_note
        m = len(self.V)
        self.diameter = max(np.linalg.norm(self.V[i] - self.V[j]) for i in range(m) for j in range(m))

        self.E_start = self.V
        self.E_end = np.roll(self.V, -1, axis=0)

        # Birleşim içinde düzgün örnekleme noktaları (ızgara)
        min_xy = self.V.min(axis=0)
        max_xy = self.V.max(axis=0)
        gx = np.linspace(min_xy[0] + 0.05, max_xy[0] - 0.05, 9)
        gy = np.linspace(min_xy[1] + 0.05, max_xy[1] - 0.05, 9)
        xx, yy = np.meshgrid(gx, gy)
        pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        in_poly = np.array([self.contains(p) for p in pts])
        self.grid_pts = pts[in_poly]
        self.thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        c, s = np.cos(self.thetas), np.sin(self.thetas)
        self.R = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)

    def boundary(self):
        return np.vstack([self.V, self.V[:1]])

    def contains(self, p, eps=1e-12):
        return any(c.contains(p, eps=eps) for c in self.pieces)

    def exit_param(self, a, b):
        d = b - a
        M = len(self.V)
        best_t = None
        for i in range(M):
            p1 = self.V[i]
            p2 = self.V[(i + 1) % M]
            dp = p2 - p1
            cross = d[0] * dp[1] - d[1] * dp[0]
            if abs(cross) < 1e-12:
                continue
            q = p1 - a
            t = (q[0] * dp[1] - q[1] * dp[0]) / cross
            u = (q[0] * d[1] - q[1] * d[0]) / cross
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                if best_t is None or t < best_t:
                    best_t = t
        return best_t

    def max_margin(self, pts, **kw):
        res = [c.max_margin(pts, **kw) for c in self.pieces]
        return max(res, key=lambda x: x[0])

    def max_margin_batch(self, P, **kw):
        res = [c.max_margin_batch(P, **kw) for c in self.pieces]
        margins = np.maximum.reduce([r[0] for r in res])
        thetas = res[0][1]
        trans = res[0][2]
        return margins, thetas, trans

    def fits_batch(self, P, **kw):
        # 1. Konveks parçaların LP testi (hızlı)
        piece_fits = np.stack([c.fits_batch(P, **kw) for c in self.pieces], axis=0)
        fits = np.any(piece_fits, axis=0)
        unresolved = np.where(~fits)[0]
        if len(unresolved) == 0:
            return fits

        # 2. Parçalar arası geçiş yapan rotaların birleşim testi
        G = len(self.grid_pts)
        T = len(self.thetas)
        for idx in unresolved:
            curve = P[idx]
            diff = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            if diff.sum() > self.diameter + 1e-7:
                continue
            keep = np.concatenate([[True], diff > 1e-7])
            c_pts = curve[keep]
            if len(c_pts) < 2:
                fits[idx] = True
                continue

            rot_c = np.einsum('tij,kj->tki', self.R, c_pts) # (T, K, 2)
            placed = (rot_c[None, :, :, :] + self.grid_pts[:, None, None, :]).reshape(-1, len(c_pts), 2)
            
            # Tüm noktaları ormanın içinde olan yerleşimler
            in_F = np.any([np.all(placed @ c.N.T <= c.b + 1e-7, axis=-1) for c in self.pieces], axis=0)
            all_in_F = np.all(in_F, axis=-1)
            valid_idx = np.where(all_in_F)[0]
            if len(valid_idx) == 0:
                continue

            valid_placed = placed[valid_idx]
            p1 = valid_placed[:, :-1, None, :]
            p2 = valid_placed[:, 1:, None, :]
            p3 = self.E_start[None, None, :, :]
            p4 = self.E_end[None, None, :, :]

            d1 = p2 - p1
            d2 = p4 - p3
            cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
            par = np.abs(cross) < 1e-12
            cross_safe = np.where(par, 1.0, cross)
            dp = p3 - p1
            t = (dp[..., 0] * d2[..., 1] - dp[..., 1] * d2[..., 0]) / cross_safe
            u = (dp[..., 0] * d1[..., 1] - dp[..., 1] * d1[..., 0]) / cross_safe
            hit = (~par) & (t >= 1e-5) & (t <= 1.0 - 1e-5) & (u >= 1e-5) & (u <= 1.0 - 1e-5)
            no_hit = ~np.any(hit, axis=(1, 2))
            if np.any(no_hit):
                fits[idx] = True

        return fits


class CircleForest(Forest):
    """Daire: dönme önemsiz. margin = r - (rotanın minimum çevreleyen çemberi)."""

    def __init__(self, radius=0.5, name="circle"):
        self.r = radius
        self.name = name
        self.diameter = 2 * radius
        self.known_best = 2 * radius
        self.known_note = "kanıtlanmış optimum: çap"

    @staticmethod
    def _mec(P):
        P = np.unique(np.round(P, 12), axis=0)
        n = len(P)
        if n == 1:
            return P[0], 0.0
        best = (None, math.inf)

        def ok(cen, r):
            return np.all(np.linalg.norm(P - cen, axis=1) <= r + 1e-9)

        for i in range(n):
            for j in range(i + 1, n):
                cen = (P[i] + P[j]) / 2
                r = np.linalg.norm(P[i] - cen)
                if r < best[1] and ok(cen, r):
                    best = (cen, r)
        for i, j, k in itertools.combinations(range(n), 3):
            a, b, c = P[i], P[j], P[k]
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(d) < 1e-14:
                continue
            ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
            uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
            cen = np.array([ux, uy])
            r = np.linalg.norm(a - cen)
            if r < best[1] and ok(cen, r):
                best = (cen, r)
        return best

    def max_margin(self, pts, **_):
        cen, r = self._mec(np.asarray(pts, float))
        return float(self.r - r), 0.0, -cen

    def max_margin_batch(self, P, **_):
        res = [self.max_margin(p) for p in P]
        return (np.array([r[0] for r in res]), np.array([r[1] for r in res]),
                np.array([r[2] for r in res]))

    def fits_batch(self, P, **_):
        return self.max_margin_batch(P)[0] > 1e-12

    def boundary(self):
        t = np.linspace(0, 2 * np.pi, 200)
        return np.stack([self.r * np.cos(t), self.r * np.sin(t)], axis=1)

    def contains(self, p, eps=1e-12):
        return bool(np.linalg.norm(p) <= self.r + eps)

    def exit_param(self, a, b):
        d = b - a
        A, B, C = d @ d, 2 * a @ d, a @ a - self.r ** 2
        disc = B * B - 4 * A * C
        if A < 1e-18 or disc < 0:
            return None
        t = (-B + math.sqrt(disc)) / (2 * A)
        return t if 0 <= t <= 1 else None


def create_l_shape(notch):
    return [
        (0.0, 0.0), (1.0, 0.0), (1.0, notch), 
        (notch, notch), (notch, 1.0), (0.0, 1.0)
    ]


def regular_polygon(n, side=1.0):
    R = side / (2 * math.sin(math.pi / n))
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    return np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)


def make_forest(name: str, **kw) -> Forest:
    name = name.lower()
    if name == "square":
        return ConvexPolygonForest(regular_polygon(4), "square",
                                   math.sqrt(2), "kanıtlanmış optimum: çap √2")
    if name in ("triangle", "equilateral"):
        return ConvexPolygonForest(regular_polygon(3), "equilateral triangle",
                                   math.sqrt(27 / 28), "Besicovitch zig-zag √(27/28) ≈ 0.981981 (en iyi bilinen)")
    if name == "hexagon":
        return ConvexPolygonForest(regular_polygon(6), "regular hexagon",
                                   2.0, "kanıtlanmış optimum: çap")
    if name == "pentagon":
        pentagon_coords = []
        for i in range(5):
            angle = math.radians(i * 72 - 90)
            r = 0.5257311121191336
            pentagon_coords.append((0.5 + r * math.cos(angle), 0.5 + r * math.sin(angle)))
        P = np.array(pentagon_coords)
        return ConvexPolygonForest(P, "regular pentagon (diam 1.0)", 1.0, "kanıtlanmış optimum: çap")
    if name == "rectangle":
        w, h = kw.get("w", 2.0), kw.get("h", 1.0)
        R = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        d = math.hypot(w, h)
        kb = min(d, 2.278292 * min(w, h))
        return ConvexPolygonForest(R, f"rectangle {w}x{h}", kb,
                                   "kanıtlanmış optimum: min(çap, 2.2783·kısa kenar)")
    if name == "strip":
        # sonsuz şerit ~ çok uzun dikdörtgen (genişlik 1)
        L = kw.get("length", 12.0)
        R = np.array([[0, 0], [L, 0], [L, 1], [0, 1]], float)
        f = ConvexPolygonForest(R, "strip (width 1)", 2.278292,
                                "Zalgaller broadworm ≈2.2783 (yaylar içerir; poligonal rota biraz üstünde kalır)")
        return f
    if name in ("right_triangle", "isosceles_right"):
        return ConvexPolygonForest([[0, 0], [1, 0], [0, 1]], "isosceles right triangle",
                                   None, "optimum bilinmiyor")
    if name in ("triangle_30_60_90", "t306090"):
        # çap (hipotenüs) = 1'e normalize: kısa kenar 1/2, uzun kenar √3/2
        return ConvexPolygonForest([[0, 0], [0.5, 0], [0, math.sqrt(3) / 2]],
                                   "30-60-90 triangle", None, "optimum bilinmiyor")
    if name in ("obtuse_120", "t120"):
        # 120-30-30 ikizkenar geniş açılı üçgen, taban (çap) = 1, yükseklik = 1/(2√3)
        h = 1.0 / (2.0 * math.sqrt(3))
        return ConvexPolygonForest([[0, 0], [1.0, 0], [0.5, h]],
                                   "obtuse 120 triangle", None, "optimum bilinmiyor")
    if name in ("golden_gnomon", "gnomon"):
        # Altın Gnomon (108-36-36 Üçgeni): Taban = 1.0, Taban açısı = 36°, Yükseklik = 0.5 * tan(36°)
        h_gnomon = 0.5 * math.tan(math.radians(36))
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.5, h_gnomon]],
                                   "golden gnomon (108-36-36)", None, "optimum bilinmiyor")
    if name.startswith("iso_"):
        # Dinamik ikizkenar üçgen: taban = 1.0, tepe açısı örn: iso_95, iso_100, ...
        apex_angle = float(name.split("_")[1])
        base_angle = (180.0 - apex_angle) / 2.0
        h = 0.5 * math.tan(math.radians(base_angle))
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.5, h]],
                                   f"isosceles apex {apex_angle} deg", None, "optimum bilinmiyor")
    if name in ("triangle_3_4_5", "tri_345"):
        # 3-4-5 dik üçgeni: kenarlar 0.6, 0.8, hipotenüs (çap) = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.64, 0.48]],
                                   "3-4-5 right triangle", None, "optimum bilinmiyor")
    if name in ("obtuse_asymmetric", "obtuse_asym"):
        # Asimetrik basık geniş açılı üçgen: taban = 1.0, köşe = (0.1, 0.2)
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.1, 0.2]],
                                   "obtuse asymmetric triangle", None, "optimum bilinmiyor")
    if name in ("trapezoid_iso", "trapezoid"):
        # İkizkenar yamuk: taban = 1.0, tavan = 0.4, yükseklik = 0.4
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.7, 0.4], [0.3, 0.4]],
                                   "isosceles trapezoid", None, "optimum bilinmiyor")
    if name in ("l_shape", "l_forest", "l_notch_05"):
        # L-Şekli: [0,1]x[0,0.5] ve [0,0.5]x[0.5,1.0] konveks dikdörtgen birleşimi
        b_L = create_l_shape(0.5)
        p1 = ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]], "box1")
        p2 = ConvexPolygonForest([[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]], "box2")
        return CompositePolygonForest(b_L, [p1, p2], "L-shape forest (notch 0.5)", None, "konkav geometrili orman")
    if name == "l_notch_03":
        b_L = create_l_shape(0.3)
        p1 = ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [1.0, 0.3], [0.0, 0.3]], "box1")
        p2 = ConvexPolygonForest([[0.0, 0.3], [0.3, 0.3], [0.3, 1.0], [0.0, 1.0]], "box2")
        return CompositePolygonForest(b_L, [p1, p2], "L-shape forest (notch 0.3)", None, "konkav geometrili orman")
    if name == "l_notch_07":
        b_L = create_l_shape(0.7)
        p1 = ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [1.0, 0.7], [0.0, 0.7]], "box1")
        p2 = ConvexPolygonForest([[0.0, 0.7], [0.7, 0.7], [0.7, 1.0], [0.0, 1.0]], "box2")
        return CompositePolygonForest(b_L, [p1, p2], "L-shape forest (notch 0.7)", None, "konkav geometrili orman")
    if name.startswith("l_notch_"):
        n_val = float(name.replace("l_notch_", "").replace("_", "."))
        b_L = create_l_shape(n_val)
        p1 = ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [1.0, n_val], [0.0, n_val]], "box1")
        p2 = ConvexPolygonForest([[0.0, n_val], [n_val, n_val], [n_val, 1.0], [0.0, 1.0]], "box2")
        return CompositePolygonForest(b_L, [p1, p2], f"L-shape forest (notch {n_val})", None, "konkav geometrili orman")
    if name in ("plus_shape", "cross_forest"):
        # Artı / Haç Şekli: yatay dikdörtgen ve dikey dikdörtgen birleşimi
        b_plus = [
            (0.33, 0.0), (0.66, 0.0), (0.66, 0.33), (1.0, 0.33),
            (1.0, 0.66), (0.66, 0.66), (0.66, 1.0), (0.33, 1.0),
            (0.33, 0.66), (0.0, 0.66), (0.0, 0.33), (0.33, 0.33)
        ]
        p_horiz = ConvexPolygonForest([[0.0, 0.33], [1.0, 0.33], [1.0, 0.66], [0.0, 0.66]], "horiz_bar")
        p_vert = ConvexPolygonForest([[0.33, 0.0], [0.66, 0.0], [0.66, 1.0], [0.33, 1.0]], "vert_bar")
        return CompositePolygonForest(b_plus, [p_horiz, p_vert], "plus-shape forest", None, "4 kollu simetrik haç")
    if name in ("t_shape", "tee_forest"):
        # T-Şekli: üst yatay çubuk ve dikey gövde birleşimi
        b_t = [
            (0.35, 0.0), (0.65, 0.0), (0.65, 0.7), (1.0, 0.7),
            (1.0, 1.0), (0.0, 1.0), (0.0, 0.7), (0.35, 0.7)
        ]
        p_bar = ConvexPolygonForest([[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]], "t_top_bar")
        p_stem = ConvexPolygonForest([[0.35, 0.0], [0.65, 0.0], [0.65, 0.7], [0.35, 0.7]], "t_stem")
        return CompositePolygonForest(b_t, [p_bar, p_stem], "T-shape forest", None, "T-biçimli dar koridorlu orman")
    if name in ("asym_150", "asym_10_20_150", "triangle_150"):
        # 10-20-150 aşırı basık ve asimetrik üçgen, çap = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.6736, 0.1187]],
                                   "10-20-150 asymmetric triangle", None, "optimum bilinmiyor")
    if name in ("kepler", "kepler_tri", "kepler_triangle"):
        # Kepler Dik Üçgeni (1 : √phi : phi), çap = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.381966, 0.485868]],
                                   "Kepler right triangle", None, "optimum bilinmiyor")
    if name in ("golden", "golden_tri", "golden_triangle"):
        # Altın Üçgen (72-72-36), çap = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [0.618034, 0.0], [0.309017, 0.951056]],
                                   "Golden triangle (72-72-36)", None, "optimum bilinmiyor")
    if name in ("needle_5", "needle5", "needle"):
        # Ultra-sivri iğne üçgeni (tepe 5°), çap = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [0.087155, 0.0], [0.043577, 0.999048]],
                                   "Needle triangle (5 deg apex)", None, "optimum bilinmiyor")
    if name in ("flat_wedge", "iso_170", "wedge_170"):
        # 5-5-170 ultra-basık ikizkenar üçgen, çap = 1.0, h = 0.5 * tan(5°) = 0.04373
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.5, 0.04373]],
                                   "Flat wedge (5-5-170 isosceles)", None, "optimum bilinmiyor")
    if name in ("heron", "heron_13_14_15", "tri_heron"):
        # Klasik Heron üçgeni (13-14-15), çap (15) = 1.0'a normalize
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.56, 0.74666]],
                                   "Heron triangle (13-14-15)", None, "optimum bilinmiyor")
    if name in ("asym_100", "triangle_20_60_100", "asym_20_60_100"):
        # 20-60-100 asimetrik geniş açılı üçgen, çap = 1.0
        return ConvexPolygonForest([[0.0, 0.0], [1.0, 0.0], [0.82635, 0.30076]],
                                   "Asymmetric 20-60-100 triangle", None, "optimum bilinmiyor")
    if name == "circle":
        return CircleForest(0.5)
    raise ValueError(f"bilinmeyen şekil: {name}")


SHAPES = ["square", "triangle", "right_triangle", "triangle_30_60_90", "tri_345", "obtuse_120",
          "obtuse_asym", "trapezoid", "l_shape", "l_notch_03", "l_notch_05", "l_notch_07",
          "plus_shape", "t_shape", "golden_gnomon", "asym_150",
          "kepler", "golden", "needle_5", "flat_wedge", "heron", "asym_100",
          "iso_95", "iso_96", "iso_97",
          "iso_97.1", "iso_97.2", "iso_97.3", "iso_97.4", "iso_97.5",
          "iso_97.6", "iso_97.7", "iso_97.8", "iso_97.9",
          "iso_98", "iso_99",
          "iso_100", "iso_105", "iso_110", "iso_115",
          "pentagon", "hexagon", "rectangle", "strip", "circle"]
