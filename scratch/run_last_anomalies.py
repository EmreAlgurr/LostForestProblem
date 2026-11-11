"""
Son Anomaliler Taraması:
1. flat_wedge: ladder 10, 20
2. heron: ladder 6, 10, 14
3. asym_100: ladder 6, 10, 14
"""
import sys, os, subprocess, json
import numpy as np

tasks = [
    ("flat_wedge", [10, 20], 250, 60),
    ("heron", [6, 10, 14], 250, 60),
    ("asym_100", [6, 10, 14], 250, 60)
]

results = []

for shape, ladder, gens, pop in tasks:
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    ladder_str = ",".join(str(x) for x in ladder)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", ladder_str, "--gens", str(gens), "--pop", str(pop)]
    subprocess.run(cmd, check=True)

    ladder_res = {}
    for n in ladder:
        p = f"results/{shape}_n{n}.json"
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            ladder_res[n] = d["escape_length"]

    json_path = f"results/{shape}_n{ladder[-1]}.json"
    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        E = data["escape_length"]
        diam = data["diameter"]
        imp = (diam - E) / diam * 100
        V = np.array(data["vertices"])

        diffs = np.diff(V, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        m = lens > 1e-4
        diffs, lens = diffs[m], lens[m]
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        turns = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
        deg_turns = np.degrees(turns)

        sig_turns = deg_turns[np.abs(deg_turns) > 5.0]
        pos_count = np.sum(sig_turns > 0)
        neg_count = np.sum(sig_turns < 0)

        if len(sig_turns) == 0:
            shape_type = "Düz Çizgi"
        elif pos_count > 0 and neg_count > 0:
            if len(sig_turns) >= 4:
                shape_type = f"Çoklu Zikzak / Dalga ({len(sig_turns)} kırılma)"
            else:
                shape_type = "Z (Zikzak)"
        elif pos_count >= 2 or neg_count >= 2:
            shape_type = "C (U-Kavisi / Kanca)"
        else:
            shape_type = "V / Kırık Çizgi"

        print(f"[{shape}] E = {E:.6f} (Çap: {diam:.4f}, Kazanç: %{imp:.2f}) | Tip: {shape_type} | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        results.append({
            "shape": shape,
            "ladder": ladder_res,
            "E": E,
            "diameter": diam,
            "improvement_pct": imp,
            "type": shape_type,
            "turns": deg_turns.tolist(),
            "sig_turns_count": int(len(sig_turns))
        })

with open("results/last_anomalies_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n\nBÜTÜN SON ANOMALİ TESTLERİ TAMAMLANDI!")

