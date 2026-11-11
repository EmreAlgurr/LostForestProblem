"""
Mikroskop taraması: iso_96, iso_97, iso_98, iso_99
Faz geçişinin (Z -> C) tam koptuğu dereceyi bulma.
"""
import sys, os, subprocess, json
import numpy as np

angles = [96, 97, 98, 99]
results = []

for ang in angles:
    shape = f"iso_{ang}"
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", "6,10", "--gens", "250", "--pop", "60"]
    subprocess.run(cmd, check=True)
    
    json_path = f"results/{shape}_n10.json"
    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        E = data["escape_length"]
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
                shape_type = "C (U-kavisi / Aynı yönlü dönüş)"
                sign_str = "+,+" if signs[0] > 0 else "-,-"
            else:
                shape_type = "Z (Zikzak / Zıt yönlü dönüş)"
                sign_str = "+,-" if signs[0] > 0 else "-,+"
        else:
            shape_type = "Tek kırılmalı / Belirsiz"
            sign_str = str([round(s, 1) for s in sig_turns])
            
        print(f"[{shape}] E = {E:.6f} | Tip: {shape_type} | İşaretler: {sign_str} | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        results.append({
            "angle": ang,
            "shape": shape,
            "E": E,
            "type": shape_type,
            "sign_pattern": sign_str,
            "turns": deg_turns.tolist()
        })

print("\n\n==================== MİKROSKOPİK FAZ GEÇİŞİ ÖZETİ ====================")
for r in results:
    print(f"Tepe Açısı: {r['angle']:>2}° | E = {r['E']:.6f} | Tip: {r['type']} | İşaret: {r['sign_pattern']}")

with open("results/micro_sweep_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

