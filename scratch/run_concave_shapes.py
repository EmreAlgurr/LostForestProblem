"""
Konkav / Non-Konveks Geometriler:
1. l_shape (L-Şekli)
2. plus_shape (Artı / Haç Şekli)

Ladder 6, 10, 14 ile çalıştırır ve morfolojiyi detaylı analiz eder.
"""
import sys, os, subprocess, json
import numpy as np

shapes = ["l_shape", "plus_shape"]
results = []

for shape in shapes:
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", "6,10,14", "--gens", "250", "--pop", "60"]
    subprocess.run(cmd, check=True)
    
    ladder_res = {}
    for n in [6, 10, 14]:
        p = f"results/{shape}_n{n}.json"
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            ladder_res[n] = d["escape_length"]
            
    json_path = f"results/{shape}_n14.json"
    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        E = data["escape_length"]
        diam = data["diameter"]
        imp = (diam - E) / diam * 100
        V = np.array(data["vertices"])
        
        diffs = np.diff(V, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        m = lens > 1e-5
        diffs, lens = diffs[m], lens[m]
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        turns = (np.diff(headings) + np.pi) % (2*np.pi) - np.pi
        deg_turns = np.degrees(turns)
        
        sig_turns = deg_turns[np.abs(deg_turns) > 10]
        # Morfoloji analizi:
        # Eğer dönüş işaretleri değişiyorsa (örn: +, -, +) -> S-kavisi veya zikzak
        # Eğer hepsi aynıysa -> C-kavisi
        pos_count = np.sum(sig_turns > 0)
        neg_count = np.sum(sig_turns < 0)
        if pos_count > 0 and neg_count > 0:
            if len(sig_turns) >= 3:
                shape_type = "S-Kavisi / Çoklu Zikzak"
            else:
                shape_type = "Z (Zikzak)"
        elif pos_count >= 2 or neg_count >= 2:
            shape_type = "C (U-kavisi / Kanca)"
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

print("\n\n==================== KONKAV GEOMETRİLER ÖZET TABLOSU ====================")
for r in results:
    print(f"Şekil: {r['shape']:<15} | E = {r['E']:.6f} (Çap: {r['diameter']:.4f}, %{r['improvement_pct']:.2f} kazanç) | Morfoloji: {r['type']}")

with open("results/concave_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

