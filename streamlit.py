import streamlit as st
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Ini adalah informasi penyakit yang akan ditampilkan
DISEASE_INFO = {
    "PPOK": {
        "nama": "Penyakit Paru Obstruktif Kronis (PPOK)",
        "deskripsi": "Penyakit progresif yang menyebabkan kesulitan bernapas akibat peradangan saluran napas.",
        "gejala": ["Sesak napas kronis", "Batuk berdahak", "Mengi", "Rasa berat di dada"],
        "penanganan": ["Berhenti merokok", "Bronkodilator", "Terapi oksigen", "Rehabilitasi paru"]
    },
    "Asma": {
        "nama": "Asma",
        "deskripsi": "Kondisi kronis yang menyebabkan peradangan dan penyempitan saluran napas.",
        "gejala": ["Sesak napas", "Mengi", "Batuk-batuk", "Rasa tertekan di dada"],
        "penanganan": ["Inhaler", "Menghindari pemicu", "Obat anti-inflamasi", "Manajemen stress"]
    },
    # Tambahkan informasi penyakit lainnya di sini contoh seperti di atas
    "Pneumonia": {
        "nama": "Pneumonia",
        "deskripsi": "Infeksi yang menyebabkan peradangan pada kantung udara di satu atau kedua paru-paru.",
        "gejala": ["Batuk", "Demam", "Sesak napas", "Nyeri dada saat bernapas"],
        "penanganan": ["Antibiotik", "Obat penurun demam", "Istirahat yang cukup"]
    },
    "Bronkitis": {
        "nama": "Bronkitis",
        "deskripsi": "Peradangan pada saluran bronkus yang menghubungkan trakea dengan paru-paru.",
        "gejala": ["Batuk kering", "Sesak napas", "Nyeri dada", "Kelelahan"],
        "penanganan": ["Obat batuk", "Antibiotik (jika infeksi bakteri)", "Inhaler"]
    },
    "ISPA": {
        "nama": "ISPA",
        "deskripsi": "lorepisum.",
        "gejala": ["Batuk kering", "Sesak napas", "Nyeri dada"],
        "penanganan": ["Obat batuk", "Antibiotik (jika infeksi bakteri)", "Inhaler"]
    },
    "Common_Cold": {
        "nama": "Common Cold",
        "deskripsi": "lorepisum.",
        "gejala": ["Batuk kering", "Sesak napas", "Nyeri dada"],
        "penanganan": ["Obat batuk", "Antibiotik (jika infeksi bakteri)", "Inhaler"]
    },
    "Faringitis_Akut": {
        "nama": "Faringitis Akut",
        "deskripsi": "lorepisum.",
        "gejala": ["Batuk kering", "Sesak napas", "Nyeri dada"],
        "penanganan": ["Obat batuk", "Antibiotik (jika infeksi bakteri)", "Inhaler"]
    },
    "Tonsilitis_Akut": {
        "nama": "Tonsilitis Akut",
        "deskripsi": "lorepisum.",
        "gejala": ["Batuk kering", "Sesak napas", "Nyeri dada"],
        "penanganan": ["Obat batuk", "Antibiotik (jika infeksi bakteri)", "Inhaler"]
    },
}

# Konfigurasi halaman harus menjadi command Streamlit pertama
st.set_page_config(page_title="Respirazzy", page_icon="🩺", layout="wide")

# --- 1. Memuat Fungsi Keanggotaan ---
def load_membership_functions(path):
    """
    Memuat fungsi keanggotaan dari file CSV.
    Returns:
        mf: Dictionary fungsi keanggotaan
        cmap: Pemetaan kategori gejala
    """
    df = pd.read_csv(path)
    mf, cmap = {}, {}
    for _, r in df.iterrows():
        grp, g, setn = r['kategori'], r['Gejala'], r['Kategori']
        a, c = r['first'], r['second']
        b = (a + c) / 2.0
        mf.setdefault(g, {})[setn] = (a, b, c)
        cmap.setdefault(grp, set()).add(g)
    return mf, cmap

# --- 2. Memuat Aturan dengan Bobot ---
def load_rules_with_weights(path):
    """
    Memuat aturan fuzzy dan bobotnya dari file CSV.
    Returns:
        rules: List tuple (kondisi, bobot, nama_penyakit)
    """
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

# --- 3. Input Gejala dari Pengguna ---
def get_user_inputs(mf, cmap):
    """
    Mengambil input gejala dari pengguna melalui antarmuka Streamlit.
    Returns:
        inp: Dictionary nilai input untuk setiap gejala
    """
    inp = {}
    for grp, gs in cmap.items():
        st.subheader(f"{grp.title()}")
        col1, col2 = st.columns([1, 1])
        for i, g in enumerate(sorted(gs)):
            if g not in mf:
                continue
            pts = [p for (a, b, c) in mf[g].values() for p in (a, b, c)]
            lo, hi = min(pts), max(pts)
            default = (lo + hi) / 2
            step = 0.1 if hi - lo <= 10 else 0.5
            with col1 if i % 2 == 0 else col2:
                with st.container():
                    st.markdown(f'<div style="background-color: #f0f2f6; border-radius: 10px; margin-bottom: 10px;">', unsafe_allow_html=True)
                    st.markdown(f"**{g.title()} ({lo:.1f}–{hi:.1f})**")
                    value = st.number_input("", min_value=lo, max_value=hi, value=default, step=step, key=g)
                    inp[g] = value
                    st.markdown("</div>", unsafe_allow_html=True)
    return inp

# --- 4. Fungsi Pembantu ---
def var_and_set_name(tok):
    """Memisahkan nama variabel dan set dari token"""
    parts = tok.split('_')
    return '_'.join(parts[:-1]), parts[-1]

def trimf(x, params):
    """
    Menghitung derajat keanggotaan fungsi segitiga.
    Args:
        x: Nilai input
        params: Parameter fungsi segitiga (a,b,c)
    Returns:
        Derajat keanggotaan [0,1]
    """
    a, b, c = params
    if a == b == c:
        return 1.0 if x == a else 0.0
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)

# --- 5. Inferensi Fuzzy dengan Bobot dan Pelacakan ---
def fuzzy_inference_confidence_weighted_with_trace(inputs, mf, rules):
    """
    Melakukan inferensi fuzzy dengan bobot dan pelacakan proses.
    Returns:
        fuzzy_vals: Nilai fuzzy untuk setiap variabel
        raw_degrees: Derajat keanggotaan mentah untuk setiap penyakit
    """
    fuzzy_vals = {}
    for var, x in inputs.items():
        fuzzy_vals[var] = {}
        for setn, params in mf.get(var, {}).items():
            mu = trimf(x, params)
            fuzzy_vals[var][setn] = mu

    raw_degrees = {}
    for conds, weights, disease in rules:
        base_strength = 0.0
        match_count = 0
        for c, w in zip(conds, weights):
            var, setn = var_and_set_name(c)
            mu = fuzzy_vals.get(var, {}).get(setn, 0.0)
            if mu > 0:
                match_count += 1
            contrib = mu * w
            base_strength += contrib
        match_ratio = match_count / len(conds) if conds else 0
        boosted_strength = base_strength * (1 + match_ratio)
        raw_degrees[disease] = boosted_strength

    return fuzzy_vals, raw_degrees

# --- 6. Normalisasi N Teratas ---
def normalize_top_n(raw_degrees, n=3):
    """
    Menormalisasi dan mengambil n hasil teratas.
    Returns:
        confidences: Dictionary persentase kepercayaan
        top_n: List n penyakit teratas
    """
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

# --- 7. Program Utama ---
if __name__ == "__main__":
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
        color: #08B2FF !important;  
        border: 1px solid #1F77B4;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
        font-weight: 500;  /* Added font-weight for better visibility */
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
    if st.sidebar.button("Informasi", key="info_button"):
        st.session_state.page = "Informasi"
    if st.sidebar.button("About", key="about_button"):
        st.session_state.page = "About"

    # Memuat data dan inisialisasi
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
                            <h3>SMART DIAGNOSIS FOR RESPIRATORY HEALTH</h3>
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

            # Create DataFrame for Table - Only top 3
            top3_data = {
                "Penyakit": [disease for disease in top3],
                "Kemungkinan (%)": [confidences[disease] for disease in top3]
            }
            df = pd.DataFrame(top3_data)

            # Display Table and Chart in more compact layout
            col1, col2 = st.columns([1.2, 1])
            with col1:
                for index, row in df.iterrows():
                    st.markdown(
                        f"""
                        <div style='display: flex; align-items: center; margin-bottom: 5px; background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;'>
                            <div style='flex: 1;'>{row['Penyakit']}</div>
                            <div style='background-color: #1F77B4; color: white; padding: 5px 10px; border-radius: 5px; margin-left: 10px;'>{row['Kemungkinan (%)']:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col2:
                # Set figure with transparent background
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='none')
                ax.pie(
                    df["Kemungkinan (%)"],
                    labels=df["Penyakit"],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=plt.cm.Blues(np.linspace(0.3, 0.8, len(df))),
                    textprops={'color': 'black', 'fontsize': 9},
                    labeldistance=0.6,  # Bring labels closer to center
                    pctdistance=0.45,   # Bring percentages closer to center
                )
                ax.axis('equal')
                
                # Make plot background transparent
                fig.patch.set_alpha(0.0)
                ax.set_facecolor('none')
                
                # Add tight layout to remove extra whitespace
                plt.tight_layout()
                
                st.pyplot(fig)

    # Informasi Page
    elif st.session_state.page == "Informasi":
        st.title("Informasi Penyakit Pernapasan")
        
        # Styling for disease info cards
        st.markdown("""
        <style>
        .disease-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .disease-title {
            color: #1F77B4;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .section-title {
            color: #155799;
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        }
        .info-text {
            color: #333;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .symptom-list {
            list-style-type: disc;
            margin-left: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        # Tampilannya disini
        for disease_key, info in DISEASE_INFO.items():
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-title">{info['nama']}</div>
                <div class="section-title">Deskripsi:</div>
                <div class="info-text">{info['deskripsi']}</div>
                <div class="section-title">Gejala Utama:</div>
                <ul class="symptom-list">
                    {''.join(f'<li class="info-text">{gejala}</li>' for gejala in info['gejala'])}
                </ul>
                <div class="section-title">Penanganan:</div>
                <ul class="symptom-list">
                    {''.join(f'<li class="info-text">{penanganan}</li>' for penanganan in info['penanganan'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

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