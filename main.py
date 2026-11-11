"""
Bellman'ın "ormanda kaybolma" problemi için evrimsel arama.

Örnekler:
    py main.py --shape square
    py main.py --shape triangle --segments 8 --gens 600
    py main.py --shape strip --ladder 6,10,16,24 --gens 250 --polish 150
    py main.py --shape triangle --init results/triangle.json --segments 12 --polish 200
    py main.py --shape all --gens 300          # tüm şekiller, sonuç tablosu
    py main.py --show results/triangle.json    # kayıtlı sonucu çiz

--ladder (çok çözünürlüklü arama):
    Önce az segmentle evrimleşir, sonra en iyi rota segmentlere bölünüp daha fazla
    segmentle yeniden evrimleşir. Eğri (kavisli) optimumlara böyle yaklaşılır;
    E(n) ≈ E* + c/n² varsayımıyla Richardson ekstrapolasyonu da raporlanır.
"""
import argparse
import os

import matplotlib
import numpy as np

from lostforest import Config, Evolver, save_result, load_result, make_forest, SHAPES


def run_one(args, shape, segments=None, init=None, tag=None):
    cfg = Config(shape=shape, segments=segments or args.segments, pop=args.pop, gens=args.gens,
                 seed=args.seed, workers=args.workers, patience=args.patience,
                 init_genome=init, polish=args.polish, out_dir=args.out)
    res = Evolver(cfg).run()
    name = tag or shape
    os.makedirs(args.out, exist_ok=True)
    path = save_result(res, args.out) if tag is None else _save_as(res, args.out, name)
    print(f"\n=== {name} ({cfg.segments} segment) ===")
    print(f"  en kötü durum kaçış mesafesi E = {res.escape_length:.6f}")
    print(f"  çap (düz çizgi)               = {res.diameter:.6f}")
    if res.known_best:
        print(f"  bilinen en iyi                = {res.known_best:.6f}   [{res.known_note}]")
    print(f"  süre {res.seconds:.1f}s, kaydedildi: {path}")
    from lostforest.visualize import plot_result
    plot_result(make_forest(shape), res.vertices, res.escape_length, res.history,
                save_path=os.path.join(args.out, f"{name}.png"), show=not args.no_show)
    return res


def _save_as(res, out, name):
    import json
    from dataclasses import asdict
    p = os.path.join(out, f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2, ensure_ascii=False)
    return p


def run_ladder(args, shape, ladder):
    init = load_result(args.init).genome if args.init else None
    rows = []
    for n in ladder:
        r = run_one(args, shape, segments=n, init=init, tag=f"{shape}_n{n}")
        rows.append((n, r.escape_length))
        init = r.genome
    print("\n  n    E(n)")
    for n, E in rows:
        print(f"  {n:<4} {E:.6f}")
    if len(rows) >= 2:
        (n1, E1), (n2, E2) = rows[-2], rows[-1]
        Ex = (n2 ** 2 * E2 - n1 ** 2 * E1) / (n2 ** 2 - n1 ** 2)
        print(f"  Richardson tahmini (n→∞, eğri limit): E* ≈ {Ex:.6f}  "
              f"(sadece tahmin; kanıtlanmış üst sınır = {min(E for _, E in rows):.6f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="square", help=f"{', '.join(SHAPES)} veya 'all'")
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--init", metavar="JSON", help="önceki sonuçtan sıcak başlatma")
    ap.add_argument("--polish", type=int, default=0, help="sonda yerel cilalama iterasyonu")
    ap.add_argument("--ladder", help="segment sayıları, örn. 6,10,16,24 (çok çözünürlüklü arama)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-show", action="store_true", help="grafik penceresi açma (sadece PNG kaydet)")
    ap.add_argument("--show", metavar="JSON", help="kayıtlı sonucu görselleştir")
    args = ap.parse_args()

    if args.no_show or args.ladder:
        matplotlib.use("Agg")
        args.no_show = True

    if args.show:
        from lostforest.visualize import plot_result
        r = load_result(args.show)
        plot_result(make_forest(r.shape, **r.config.get("shape_kw", {})), r.vertices,
                    r.escape_length, r.history)
        return

    if args.ladder:
        run_ladder(args, args.shape, [int(x) for x in args.ladder.split(",")])
        return

    init = load_result(args.init).genome if args.init else None
    shapes = SHAPES if args.shape == "all" else [args.shape]
    results = [run_one(args, s, init=init) for s in shapes]

    if len(results) > 1:
        print("\n{:<16} {:>10} {:>10} {:>12}".format("şekil", "E (evrim)", "çap", "bilinen"))
        for r in results:
            kb = f"{r.known_best:.4f}" if r.known_best else "-"
            print(f"{r.shape:<16} {r.escape_length:>10.4f} {r.diameter:>10.4f} {kb:>12}")


if __name__ == "__main__":
    main()
