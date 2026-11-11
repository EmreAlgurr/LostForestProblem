"""
Faz geçişi taraması: iso_95, iso_100, iso_105, iso_110, iso_115
Her şekil için ladder 6, 10 çalıştırır ve geometriyi (Z mi C mi) analiz eder.
"""
import sys, os, subprocess, json
import numpy as np

angles = [95, 100, 105, 110, 115]
results = []

for ang in angles:
    shape = f"iso_{ang}"
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", "6,10", "--gens", "250", "--pop", "60"]
    subprocess.run(cmd, check=True)
    
    # Sonuç json'ını oku
    json_path = f"results/{shape}_n10.json"
    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        E = data["escape_length"]
        V = np.array(data["vertices"])
        # Açılar
        diffs = np.diff(V, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        m = lens > 1e-5
        diffs, lens = diffs[m], lens[m]
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        turns = (np.diff(headings) + np.pi) % (2*np.pi) - np.pi
        deg_turns = np.degrees(turns)
        
        # Z mi C mi analizi:
        # Önemli dönüşler (> 15 derece)
        sig_turns = deg_turns[np.abs(deg_turns) > 15]
        if len(sig_turns) >= 2:
            if (sig_turns[0] > 0 and sig_turns[-1] > 0) or (sig_turns[0] < 0 and sig_turns[-1] < 0):
                shape_type = "C (U-kavisi / Aynı yönlü dönüş)"
            else:
                shape_type = "Z (Zikzak / Zıt yönlü dönüş)"
        else:
            shape_type = "V veya tek kırılmalı"
            
        print(f"[{shape}] E = {E:.6f} | Tip: {shape_type} | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        results.append({
            "angle": ang,
            "shape": shape,
            "E": E,
            "type": shape_type,
            "turns": deg_turns.tolist()
        })

print("\n\n==================== FAZ GEÇİŞİ ÖZET TABLOSU ====================")
for r in results:
    print(f"Tepe Açısı: {r['angle']:>3}° | E = {r['E']:.6f} | Morfoloji: {r['type']}")

with open("results/phase_transition_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

