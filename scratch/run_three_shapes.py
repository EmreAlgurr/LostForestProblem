"""
3 Yeni Geometri Testi:
1. tri_345 (3-4-5 Dik Üçgeni)
2. obtuse_asym (Asimetrik Geniş Açılı Üçgen)
3. trapezoid (İkizkenar Yamuk)

Ladder 6, 10, 14 ile çalıştırır ve morfolojiyi detaylı analiz eder.
"""
import sys, os, subprocess, json
import numpy as np

shapes = ["tri_345", "obtuse_asym", "trapezoid"]
results = []

for shape in shapes:
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", "6,10,14", "--gens", "250", "--pop", "60"]
    subprocess.run(cmd, check=True)
    
    # Tüm ladder kademelerini oku
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
        if len(sig_turns) >= 2:
            signs = np.sign(sig_turns)
            if signs[0] == signs[-1]:
                shape_type = "C (U-kavisi / Kanca)"
                sign_str = "+,+" if signs[0] > 0 else "-,-"
            else:
                shape_type = "Z (Zikzak)"
                sign_str = "+,-" if signs[0] > 0 else "-,+"
        else:
            shape_type = "V / Tek kırılmalı"
            sign_str = str([round(s, 1) for s in sig_turns])
            
        print(f"[{shape}] E = {E:.6f} (Kazanç: %{imp:.2f}) | Tip: {shape_type} | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        results.append({
            "shape": shape,
            "ladder": ladder_res,
            "E": E,
            "improvement_pct": imp,
            "type": shape_type,
            "sign_pattern": sign_str,
            "turns": deg_turns.tolist()
        })

print("\n\n==================== 3 YENİ GEOMETRİ ÖZET TABLOSU ====================")
for r in results:
    print(f"Şekil: {r['shape']:<15} | E = {r['E']:.6f} (%{r['improvement_pct']:.2f} kazanç) | Morfoloji: {r['type']}")

with open("results/three_shapes_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

