"""
T-Şekli ve L-Notch Taraması:
1. t_shape: ladder 6, 10, 14
2. l_notch_03: ladder 6, 10, 14
3. l_notch_07: ladder 6, 10, 14
(l_notch_05 zaten l_shape olarak çalıştırıldı: E = 1.118034)
"""
import sys, os, subprocess, json
import numpy as np

tasks = [
    ("t_shape", [6, 10, 14], 200, 50),
    ("l_notch_03", [6, 10, 14], 200, 50),
    ("l_notch_07", [6, 10, 14], 200, 50)
]

results = []

# Önceden tamamlanmış l_shape (notch 0.5) verisini de ekleyelim
if os.path.exists("results/l_shape_n14.json"):
    d_05 = json.load(open("results/l_shape_n14.json", encoding="utf-8"))
    results.append({
        "shape": "l_notch_05",
        "E": d_05["escape_length"],
        "diameter": d_05["diameter"],
        "improvement_pct": (d_05["diameter"] - d_05["escape_length"]) / d_05["diameter"] * 100,
        "type": "Düz Çizgi (Kırılmasız)",
        "turns": [0.0] * 13,
        "ladder": {6: 1.118034, 10: 1.118034, 14: 1.118034}
    })

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
            shape_type = "Düz Çizgi (Kırılmasız)"
        elif pos_count > 0 and neg_count > 0:
            if len(sig_turns) >= 3:
                shape_type = "S-Kavisi / Çoklu Zikzak"
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
            "turns": deg_turns.tolist()
        })

print("\n\n==================== T-ŞEKLİ VE L-NOTCH ÖZET TABLOSU ====================")
for r in results:
    print(f"Şekil: {r['shape']:<15} | E = {r['E']:.6f} (Çap: {r['diameter']:.4f}, %{r['improvement_pct']:.2f} kazanç) | Morfoloji: {r['type']}")

with open("results/t_and_l_notches_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

