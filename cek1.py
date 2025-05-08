import numpy as np
import pandas as pd
import ast

# --- 1. Load Membership Functions ---
def load_membership_functions(path):
    df = pd.read_csv(path)
    mf, cmap = {}, {}
    for _, r in df.iterrows():
        grp, g, setn = r['kategori'], r['Gejala'], r['Kategori']
        a, c = r['first'], r['second']
        b = (a + c) / 2.0
        mf.setdefault(g, {})[setn] = (a, b, c)
        cmap.setdefault(grp, set()).add(g)
    return mf, cmap

# --- 2. Load Rules with Weights ---
def load_rules_with_weights(path):
    df = pd.read_csv(path, dtype=str)
    rules = []
    for _, r in df.iterrows():
        try:
            conds = ast.literal_eval(r['vars'])
            weights = list(map(float, ast.literal_eval(r['weights'])))
        except Exception as e:
            print(f"Error parsing rule for {r['nama_penyakit']}: {e}")
            continue
        if len(weights) != len(conds):
            print(f"Skipping rule {r['nama_penyakit']} due to length mismatch.")
            continue
        rules.append((conds, weights, r['nama_penyakit']))
    return rules

# --- 3. Input Gejala from User ---
def get_user_inputs(mf, cmap):
    inp = {}
    for grp, gs in cmap.items():
        print(f"\nMasukkan input untuk gejala {grp.capitalize()}: ")
        for g in sorted(gs):
            if g not in mf:
                continue
            pts = [p for (a,b,c) in mf[g].values() for p in (a, b, c)]
            lo, hi = min(pts), max(pts)
            while True:
                try:
                    v = float(input(f"  - {g} ({lo:.1f}–{hi:.1f}): "))
                    if lo <= v <= hi:
                        inp[g] = v
                        break
                except ValueError:
                    pass
                print(f"    Masukkan angka antara {lo:.1f} dan {hi:.1f}.")
    return inp

# --- 4. Helpers ---
def var_and_set_name(tok):
    parts = tok.split('_')
    return '_'.join(parts[:-1]), parts[-1]

def trimf(x, params):
    a, b, c = params
    if a == b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

# --- 5. Weighted Fuzzy Inference with Trace and Match Boost ---
def fuzzy_inference_confidence_weighted_with_trace(inputs, mf, rules):
    fuzzy_vals = {}
    print("\n=== Fuzzifikasi (Membership Degrees) ===")
    for var, x in inputs.items():
        fuzzy_vals[var] = {}
        print(f"\n>> Gejala: {var} (nilai: {x})")
        for setn, params in mf.get(var, {}).items():
            mu = trimf(x, params)
            fuzzy_vals[var][setn] = mu
            print(f"   - {setn}: μ = {mu:.3f}")

    raw_degrees = {}
    print("\n=== Perhitungan Strength dengan Boost Berdasarkan Kecocokan ===")
    for conds, weights, disease in rules:
        print(f"\n>> Rule untuk Diagnosis: {disease}")
        base_strength = 0.0
        match_count = 0
        for c, w in zip(conds, weights):
            var, setn = var_and_set_name(c)
            mu = fuzzy_vals.get(var, {}).get(setn, 0.0)
            if mu > 0:
                match_count += 1
            contrib = mu * w
            base_strength += contrib
            print(f"   - {c}: μ = {mu:.3f}, bobot = {w:.2f}, kontribusi = {contrib:.3f}")
        # Boost: tambahkan faktor proporsional jumlah kecocokan
        match_ratio = match_count / len(conds) if conds else 0
        boosted_strength = base_strength * (1 + match_ratio)
        raw_degrees[disease] = boosted_strength
        print(f"   => Base Strength = {base_strength:.3f}, Matches = {match_count}/{len(conds)}, Boosted = {boosted_strength:.3f}")

    return fuzzy_vals, raw_degrees

# --- 6. Normalisasi Top N ---
def normalize_top_n(raw_degrees, n=3):
    sorted_raw = sorted(raw_degrees.items(), key=lambda x: -x[1])
    top_n = [d for d,_ in sorted_raw][:n]
    total_top = sum(raw_degrees[d] for d in top_n)
    if total_top > 0:
        return {d: (raw_degrees[d] / total_top) * 100 if d in top_n else 0.0 for d in raw_degrees}, top_n
    else:
        return {d: 0.0 for d in raw_degrees}, top_n

# --- 7. Main ---
if __name__ == "__main__":
    mf_file = "member_function_respirasi.csv"
    rule_file = "bobot_respirasi.csv"

    mf, cmap = load_membership_functions(mf_file)
    rules = load_rules_with_weights(rule_file)
    inputs = get_user_inputs(mf, cmap)

    fuzzy_vals, raw_degrees = fuzzy_inference_confidence_weighted_with_trace(inputs, mf, rules)
    confidences, top3 = normalize_top_n(raw_degrees, n=3)

    print("\n=== Diagnosis possibilities (Top 3) ===")
    for i, disease in enumerate(top3, start=1):
        print(f"  {i}) {disease} ({confidences[disease]:.1f}%)")
