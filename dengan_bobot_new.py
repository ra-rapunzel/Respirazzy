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

# --- 3. Input Gejala dari User ---
def get_user_inputs(mf, cmap):
    inp = {}
    for grp, gs in cmap.items():
        print(f"\nMasukkan input untuk gejala {grp.capitalize()}: ")
        for g in sorted(gs):
            if g not in mf:
                continue
            pts = [p for (a, b, c) in mf[g].values() for p in (a, b, c)]
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

# --- 4. Triangular Membership Function ---
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

# --- 5. Implikasi Mamdani ---
def implication(alpha, params, y):
    return np.minimum(alpha, np.array([trimf(val, params) for val in y]))

# --- 6. Defuzzifikasi MoM ---
def defuzzify_mom(y, mu):
    max_mu = np.max(mu)
    if max_mu == 0:
        return 0.0
    y_max = y[mu == max_mu]
    return (y_max[0] + y_max[-1]) / 2

# --- 7. Inference Lengkap ---
def fuzzy_inference_mamdani_weighted(inputs, mf, rules, output_mf, y_domain):
    fuzzy_vals = {}
    print("\n=== Fuzzifikasi ===")
    for var, x in inputs.items():
        fuzzy_vals[var] = {}
        print(f"\n>> Gejala: {var} (input: {x})")
        for setn, params in mf.get(var, {}).items():
            mu = trimf(x, params)
            fuzzy_vals[var][setn] = mu
            print(f"   - {setn}: μ = {mu:.3f}")

    aggregated = np.zeros_like(y_domain)
    per_disease = {}

    print("\n=== Evaluasi Rule, Implikasi, dan Agregasi ===")
    for conds, weights, disease in rules:
        match_vals = []
        for cond in conds:
            var, setn = var_and_set_name(cond)
            mu = fuzzy_vals.get(var, {}).get(setn, 0.0)
            match_vals.append(mu)
        if not match_vals:
            alpha = 0
        else:
            alpha = sum(mu * w for mu, w in zip(match_vals, weights)) / sum(weights)
        print(f"\n>> {disease} → α = {alpha:.3f}")

        # Implikasi dan agregasi
        clipped = implication(alpha, output_mf[disease], y_domain)
        aggregated = np.maximum(aggregated, clipped)
        if disease not in per_disease:
            per_disease[disease] = clipped
        else:
            per_disease[disease] = np.maximum(per_disease[disease], clipped)

    z_star = defuzzify_mom(y_domain, aggregated)
    return z_star, per_disease, aggregated

# --- 8. Get Top Diagnoses ---
def get_top_diagnoses(per_disease, y_domain):
    scores = {}
    for disease, mu in per_disease.items():
        max_mu = np.max(mu)
        if max_mu > 0:
            scores[disease] = max_mu
    sorted_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    total = sum(val for _, val in sorted_top)
    return [(d, v, 100 * v / top_total if top_total else 0) for d, v in sorted_top]

# --- 9. Main ---
if __name__ == "__main__":
    mf_file = "revisi_member_function.csv"
    rule_file = "rules_bobot_respirasi.csv"
    output_mf_file = "output_member_function.csv"

    mf, cmap = load_membership_functions(mf_file)
    rules = load_rules_with_weights(rule_file)
    inputs = get_user_inputs(mf, cmap)

    output_df = pd.read_csv(output_mf_file)
    output_mf = {r['penyakit']: (r['a'], r['b'], r['c']) for _, r in output_df.iterrows()}

    y_domain = np.linspace(0, 10, 1000)
    z_star, per_disease, aggregated = fuzzy_inference_mamdani_weighted(inputs, mf, rules, output_mf, y_domain)

    print("\n=== Hasil Defuzzifikasi (MoM) ===")
    print(f"  Skor Akhir: {z_star:.3f}")

    print("\n=== Top 3 Diagnosis ===")
    top3 = get_top_diagnoses(per_disease, y_domain)
    for i, (d, val, pct) in enumerate(top3, 1):
        print(f"  {i}) {d} → μ = {val:.3f} ({pct:.2f}%)")
