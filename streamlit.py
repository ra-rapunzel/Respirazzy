import streamlit as st
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

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
            st.error(f"Error parsing rule for {r['nama_penyakit']}: {e}")
            continue
        if len(weights) != len(conds):
            st.warning(f"Skipping rule {r['nama_penyakit']} due to length mismatch.")
            continue
        rules.append((conds, weights, r['nama_penyakit']))
    return rules

# --- 3. Input Gejala from User ---
def get_user_inputs(mf, cmap):
    inp = {}
    for grp, gs in cmap.items():
        st.subheader(f"{grp.capitalize()}:")
        for g in sorted(gs):
            if g not in mf:
                continue
            pts = [p for (a, b, c) in mf[g].values() for p in (a, b, c)]
            lo, hi = min(pts), max(pts)
            v = st.number_input(f"  - {g} ({lo:.1f}–{hi:.1f})", value=(lo + hi) / 2.0, min_value=lo, max_value=hi, step=0.1)
            inp[g] = v
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
    st.subheader("=== Fuzzifikasi (Membership Degrees) ===")
    for var, x in inputs.items():
        fuzzy_vals[var] = {}
        st.write(f"\n>> Gejala: {var} (nilai: {x})")
        for setn, params in mf.get(var, {}).items():
            mu = trimf(x, params)
            fuzzy_vals[var][setn] = mu
            st.write(f"   - {setn}: μ = {mu:.3f}")

    raw_degrees = {}
    st.subheader("\n=== Perhitungan Strength dengan Boost Berdasarkan Kecocokan ===")
    for conds, weights, disease in rules:
        st.write(f"\n>> Rule untuk Diagnosis: {disease}")
        base_strength = 0.0
        match_count = 0
        for c, w in zip(conds, weights):
            var, setn = var_and_set_name(c)
            mu = fuzzy_vals.get(var, {}).get(setn, 0.0)
            if mu > 0:
                match_count += 1
            contrib = mu * w
            base_strength += contrib
            st.write(f"   - {c}: μ = {mu:.3f}, bobot = {w:.2f}, kontribusi = {contrib:.3f}")
        # Boost: tambahkan faktor proporsional jumlah kecocokan
        match_ratio = match_count / len(conds) if conds else 0
        boosted_strength = base_strength * (1 + match_ratio)
        raw_degrees[disease] = boosted_strength
        st.write(f"   => Base Strength = {base_strength:.3f}, Matches = {match_count}/{len(conds)}, Boosted = {boosted_strength:.3f}")

    return fuzzy_vals, raw_degrees

# --- 6. Normalisasi Top N ---
def normalize_top_n(raw_degrees, n=3):
    sorted_raw = sorted(raw_degrees.items(), key=lambda x: -x[1])
    top_n = [d for d, _ in sorted_raw][:n]
    total_top = sum(raw_degrees[d] for d in top_n)
    if total_top > 0:
        return {
            d: (raw_degrees[d] / total_top) * 100 if d in top_n else 0.0
            for d in raw_degrees
        }, top_n
    else:
        return {d: 0.0 for d in raw_degrees}, []

# --- 7. Main ---
if __name__ == "__main__":
    st.set_page_config(page_title="Respirazzy", page_icon="🩺", layout="wide")

    # Custom CSS for styling
    custom_css = """
    <style>
    .stApp {
        background-image: linear-gradient(to bottom, #ADD8E6, #FFFFFF); /* Gradient dari biru muda ke putih */
        color: #000000;
    }
    .stTitle {
        color: #1F77B4;
    }
    .stSubheader {
        color: #1F77B4;
    }
    .stMarkdown p {
        color: #000000;
    }
    .stButton > button {
        background-color: #1F77B4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        width: 100%;
        margin-bottom: 10px;
    }
    .stButton > button:hover {
        background-color: #155799;
    }
    .stTextInput > div > div > input {
        background-color: white;
        color: #000000;
        border: 1px solid #1F77B4;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
    }
    .stNumberInput > div > div > input {
        background-color: white;
        color: #000000;
        border: 1px solid #1F77B4;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
    }
    .stCheckbox > label > span:first-child {
        color: #000000;
    }
    .stCheckbox > label > input[type="checkbox"] {
        accent-color: #1F77B4;
    }
    .stSidebar {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stSidebar .stButton > button {
        background-color: #1F77B4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        width: 100%;
        margin-bottom: 10px;
    }
    .stSidebar .stButton > button:hover {
        background-color: #155799;
    }
    /* Custom styling for table and pie chart */
    .stDataFrame {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Header Section
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: space-between;'>
            <div style='display: flex; align-items: center;'>
                <h1 style='margin: 0; color: #1F77B4; font-size: 24px;'>Respirazzy</h1>
            </div>
            <div>
                <a href="#home" style='text-decoration: none; color: #1F77B4; font-size: 18px; margin-right: 20px;'>Home</a>
                <a href="#diagnosis" style='text-decoration: none; color: #1F77B4; font-size: 18px; margin-right: 20px;'>Diagnosis</a>
                <a href="#about" style='text-decoration: none; color: #1F77B4; font-size: 18px;'>About</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.sidebar.button("Home", key="home_button"):
        st.session_state.page = "Home"
    if st.sidebar.button("Diagnosis", key="diagnosis_button"):
        st.session_state.page = "Diagnosis"
    if st.sidebar.button("About", key="about_button"):
        st.session_state.page = "About"

    # Load Membership Functions and Rules
    mf_file = "member_function_respirasi.csv"
    rule_file = "bobot_respirasi.csv"
    mf, cmap = load_membership_functions(mf_file)
    rules = load_rules_with_weights(rule_file)

    # Home Page
    if st.session_state.page == "Home":
        st.markdown("<div id='home'></div>", unsafe_allow_html=True)

        # Styling for the container
        st.markdown(
            """
            <style>
            .home-container {
                background-color: #FFFF;
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            .home-content {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .home-text {
                flex: 1;
                padding-right: 20px;
            }
            .home-image {
                flex: 1;
                text-align: center;
            }
            .home-container h1 {
                font-size: 36px;
                color: #1F77B4;
                margin-bottom: 10px;
            }
            .home-container p {
                font-size: 18px;
                color: #333;
                margin-bottom: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Container for home content
        with st.container():
            st.markdown(
                """
                <div class="home-container">
                    <div class="home-content">
                        <div class="home-image">
                            <img src="https://cdn.vectorstock.com/i/500p/41/86/anatomical-medical-scheme-respiratory-system-vector-26874186.jpg" alt="Respiratory Health Illustration" style="max-width: 100%; height: auto;">
                        </div>
                        <div class="home-text">
                            <h1>SMART DIAGNOSIS FOR RESPIRATORY HEALTH</h1>
                            <p>Detect 10 types of respiratory diseases instantly using fuzzy logic.</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Keep the existing button functionality
            if st.button("START DIAGNOSIS", key="start_diagnosis_button"):
                st.session_state.page = "Diagnosis"

    # Diagnosis Page
    elif st.session_state.page == "Diagnosis":
        st.title("Diagnosis Penyakit Respirasi menggunakan Logika Fuzzy")

        # Get User Inputs
        inputs = get_user_inputs(mf, cmap)

        # Perform Fuzzy Inference
        if st.button("Diagnosis", key="diagnosis_run_button"):
            fuzzy_vals, raw_degrees = fuzzy_inference_confidence_weighted_with_trace(inputs, mf, rules)
            confidences, top3 = normalize_top_n(raw_degrees, n=3)

            # Display Results
            st.subheader("Hasil Diagnosis")
            st.markdown("### Result")

            # Create DataFrame for Table
            data = {"Penyakit": list(confidences.keys()), "Kemungkinan (%)": list(confidences.values())}
            df = pd.DataFrame(data).sort_values(by="Kemungkinan (%)", ascending=False)

            # Display Table
            col1, col2 = st.columns([2, 1])
            with col1:
                for index, row in df.iterrows():
                    st.markdown(
                        f"<div style='display: flex; align-items: center;'>"
                        f"  <div style='flex: 1; padding-right: 10px;'>{row['Penyakit']}</div>"
                        f"  <div style='background-color: #1F77B4; color: white; padding: 5px 10px; border-radius: 5px;'>{row['Kemungkinan (%)']:.1f}%</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Create and Display Pie Chart
            with col2:
                fig, ax = plt.subplots()
                ax.pie(
                    df["Kemungkinan (%)"],
                    labels=df["Penyakit"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=plt.cm.Blues(np.linspace(0.3, 0.8, len(df))),
                    textprops={"color": "black"},
                )
                ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

    # About Page
    elif st.session_state.page == "About":
        st.markdown("<div id='about'></div>", unsafe_allow_html=True)
        
        # Styling for the about container
        st.markdown(
            """
            <style>
            .about-container {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 16px;
                padding: 40px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin: 20px 0;
                text-align: center;
            }
            .about-title {
                color: #1F77B4;
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 30px;
            }
            .about-text {
                color: #4A4A4A;
                font-size: 18px;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
            }
            .about-highlight {
                color: #1F77B4;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # About content
        st.markdown(
            """
            <div class="about-container">
                <div class="about-title">About</div>
                <div class="about-text">
                    <span class="about-highlight">Respirazzy</span> is a web-based Decision Support System (DSS) 
                    designed to assist in the early detection and classification of 
                    respiratory system diseases using Fuzzy Logic methodology. This 
                    intelligent system integrates user-inputted symptoms with a fuzzy 
                    inference engine to analyze and determine the most probable 
                    diagnosis among 10 types of respiratory diseases.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )