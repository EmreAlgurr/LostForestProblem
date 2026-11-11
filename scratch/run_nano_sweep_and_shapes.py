"""
1. 0.1 Derece Hassasiyetli Nanoboyut Taraması (97.1° - 97.9°)
2. Düzgün Beşgen (Pentagon, Çap 1.0)
3. 10-20-150 Aşırı Asimetrik ve Basık Üçgen (asym_150)
"""
import sys, os, subprocess, json
import numpy as np

# 1. Nanoboyut Taraması (97.1 - 97.9)
nano_angles = [97.1, 97.2, 97.3, 97.4, 97.5, 97.6, 97.7, 97.8, 97.9]
nano_results = []

for deg in nano_angles:
    shape = f"iso_{deg}"
    print(f"\n==================== {shape} BAŞLIYOR ====================", flush=True)
    cmd = [sys.executable, "main.py", "--shape", shape, "--ladder", "6,10", "--gens", "160", "--pop", "50"]
    subprocess.run(cmd, check=True)
    
    # n=10 sonucunu analiz et
    json_path = f"results/{shape}_n10.json"
    if os.path.exists(json_path):
        data = json.load(open(json_path, encoding="utf-8"))
        E = data["escape_length"]
        V = np.array(data["vertices"])
        
        diffs = np.diff(V, axis=0)
        lens = np.linalg.norm(diffs, axis=1)
        m = lens > 1e-4
        diffs, lens = diffs[m], lens[m]
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        turns = (np.diff(headings) + np.pi) % (2 * np.pi) - np.pi
        deg_turns = np.degrees(turns)
        
        sig_turns = deg_turns[np.abs(deg_turns) > 10.0]
        pos_count = np.sum(sig_turns > 0)
        neg_count = np.sum(sig_turns < 0)
        
        if pos_count > 0 and neg_count > 0:
            shape_type = "Z (Zikzak)"
            pattern = "+,-" if sig_turns[0] > 0 else "-,+"
        elif pos_count >= 2:
            shape_type = "C (U-Kavisi)"
            pattern = "+,+"
        elif neg_count >= 2:
            shape_type = "C (U-Kavisi)"
            pattern = "-,-"
        else:
            shape_type = "V / Diğer"
            pattern = "tekil"
            
        print(f"[{shape}] E = {E:.6f} | Tip: {shape_type} ({pattern}) | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        nano_results.append({
            "angle": deg,
            "shape": shape,
            "E": E,
            "type": shape_type,
            "pattern": pattern,
            "turns": deg_turns.tolist()
        })

with open("results/nano_sweep_summary.json", "w", encoding="utf-8") as f:
    json.dump(nano_results, f, indent=2)

# 2. Pentagon ve asym_150 (ladder 6, 10, 14)
other_shapes = ["pentagon", "asym_150"]
other_results = []

for shape in other_shapes:
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
            shape_type = "Zikzak / Yıldız / S"
        elif pos_count >= 2 or neg_count >= 2:
            shape_type = "C (U-Kavisi / Kanca)"
        else:
            shape_type = "V / Diğer"
            
        print(f"[{shape}] E = {E:.6f} (Çap: {diam:.4f}, Kazanç: %{imp:.2f}) | Tip: {shape_type} | Dönüşler: {np.round(deg_turns, 1)}", flush=True)
        other_results.append({
            "shape": shape,
            "ladder": ladder_res,
            "E": E,
            "diameter": diam,
            "improvement_pct": imp,
            "type": shape_type,
            "turns": deg_turns.tolist()
        })

with open("results/pentagon_and_asym150_summary.json", "w", encoding="utf-8") as f:
    json.dump(other_results, f, indent=2)

print("\n\nBÜTÜN TESTLER TAMAMLANDI!")

