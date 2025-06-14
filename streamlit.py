import streamlit as st
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

# Ini adalah informasi penyakit yang akan ditampilkan
DISEASE_INFO = {
    "Common_Cold": {
        "nama": "Common Cold (Flu Biasa)",
        "deskripsi": "Infeksi virus ringan yang menyerang saluran pernapasan atas, yaitu bagian hidung dan tenggorokan. Infeksi virus ini biasanya disebabkan oleh Rhinovirus.",
        "gejala": ["Hidung berair", "Hidung tersumbat", "Hidung gatal", "Bersin-bersin", "Batuk", "Demam"],
        "penanganan":"Istirahat yang cukup dan pemberian obat analgesik dan antipiretik sebagai pereda gejala seperti parasetamol dan dekongestan.",
        "pencegahan": "Mencuci tangan sebelum menyentuh wajah, menghindari kontak langsung dengan orang yang terinfeksi, dan menutup mulut saat bersin atau batuk",
    },

    "Influenza": {
        "nama": "Influenza",
        "deskripsi": "Infeksi virus akut yang menyerang sistem pernapasan, yaitu hidung, tenggorokan, hingga paru-paru. Penyakit ini disebabkan oleh infeksi virus RNA, terutama virus tipe A dan B.",
        "gejala": ["Demam tinggi", "Sakit kepala", "Hidung Berair", "Sakit Tenggorokan", "Batuk kering"],
        "penanganan":"Istirahat yang cukup, mengonsumsi cairan dalam jumlah banyak, serta pemberian obat sesuai gejala seperti antipiretik, dekongestan, dan antihistamin.",
        "pencegahan":"Menjaga kebersihan dan sanitasi, namun yang paling efektif adalah dengan vaksinasi influenza tahunan",
    },
    "Faringitis_Akut": {
        "nama": "Faringitis Akut",
        "deskripsi": "Infeksi atau peradangan pada bagian faring (tenggorokan), tepatnya bagian belakang tenggorokan yang menjadi penghubung antara rongga mulut dengan laring dan esofagus. Penyakit ini umumnya disebabkan oleh virus seperti Rhinovirus, Adenovirus, dan bakteri Streptococcus pyogenes",
        "gejala": ["Demam", "Sakit kepala", "Nyeri menelan", "Sakit tenggorokan", "Batuk"],
        "penanganan":"Penanganan umum istirahat yang cukup dan mengonsumsi air putih dalam jumlah cukup. Penanganan khusus disesuaikan dengan penyebabnya, karena infeksi virus maka dapat diberikan antivirus seperti Isoprinosine. Karena bakteri maka diberikan antibiotik seperti Amoksisilin atau Eritromisin untuk mencegah komplikasi dan mempercepat proses penyembuhan.",
        "pencegahan":"Memperbaiki pola makan dan menghindari makanan atau minuman yang dapat menimbulkan iritasi pada tenggorokan",
    },
    "Tonsilitis-Akut": {
        "nama": "Tonsilitis Akut",
        "deskripsi": "Kondisi saat tonsil atau amandel mengalami peradangan. Tonsil terletak di bagian belakang tenggorokan berupa dua jaringan limfoid. Penyebab umum tonsilitis akut berasal dari infeksi virus seperti denovirus atau virus Epstein--Barr, dan bisa juga disebabkan oleh bakteri seperti Streptococcus pyogenes.",
        "gejala": ["Demam", "Sakit kepala", "Nyeri menelan", "Sakit tenggorokan", "Pembengkakan amandel"],
        "penanganan":"Jika disebabkan oleh virus, maka dapat diberikan analgesik atau antipiretik seperti parasetamol, dan antivirus seperti Metisoprinal (pada gejala berat). Sedangkan jika disebabkan oleh bakteri, dapat diberikan antibiotik seperti Penisilin G Benzatin untuk mencegah komplikasi yang lebih parah.",
        "pencegahan": "Membiasakan hidup sehat, menjaga kebersihan, serta menghindari kontak dekat dengan penderita.",
    },
    "rinitis_alergi": {
        "nama": "Rinitis Alergi",
        "deskripsi": "Peradangan pada mukosa hidung (lapisan yang melapisi seluruh bagian dalam rongga hidung) yang disebabkan oleh reaksi alergi terhadap zat pemicu atau alergen, seperti debu. Penyakit ini tidak disebabkan oleh virus atau bakteri, melainkan oleh hipersensitivitas sistem imun.",
        "gejala": ["Hidung berair", "Hidung tersumbat", "Bersin", "Hidung gatal"],
        "penanganan": "Penggunaan antihistamin untuk meredakan gejala, kortikosteroid semprot hidung untuk mengurangi peradangan lokal, dan dalam beberapa kasus dilakukan imunoterapi alergen agar tubuh tidak bereaksi berlebihan terhadap alergen tertentu.",
    },
    "ispa": {
        "nama": "Infeksi Saluran Pernapasan Akut (ISPA)",
        "deskripsi": "Penyakit yang menyerang saluran pernapasan manusia. Infeksi dari virus dan bakteri, seperti rhinovirus, adenovirus, Streptococcus pneumoniae, dan Haemophilus influenzae, menjadi penyebab utama dari penyakit ini.",
        "gejala": ["Demam", "Sakit kepala", "Hidung berair", "Hidung tersumbat", "Sakit tenggorokan", "Sesak napas", "Batuk", "Napas mengi"],
        "penanganan": "Istirahat yang cukup dan mengonsumsi air putih. Apabila terdapat gejala batuk, dapat diberikan obat dekongestan. Apabila muncul demam, dapat diberikan obat antipiretik seperti parasetamol. Jika ISPA disebabkan oleh infeksi bakteri, maka disarankan pemberian antibiotik.",
    },
    "bronkitis": {
        "nama": "Bronkitis",
        "deskripsi": "Peradangan pada saluran penghubung antara batang tenggorokan dengan paru-paru, yaitu bronkus. Bronkitis dibedakan menjadi dua, yaitu bronkitis akut yang disebabkan oleh virus Rhinovirus dan Influenza virus, serta bronkitis kronis yang disebabkan oleh iritasi jangka panjang terhadap asap rokok, polusi udara, hingga bahan kimia.",
        "gejala": ["Demam", "Sesak napas", "Batuk", "Dahak", "Napas mengi"],
        "penanganan": "Untuk bronkitis kronis, penanganan dimulai dengan menghindari faktor pemicu. Untuk mencegah penurunan fungsi paru-paru dapat diberikan bronkodilator atau kortikosteroid inhalasi.",
    },
    "pneumonia": {
        "nama": "Pneumonia",
        "deskripsi": "Infeksi akut pada paru-paru, terutama pada kantong udara kecil (alveolus), yang menyebabkan bagian alveolus meradang hingga terisi cairan atau nanah. Hal ini menyebabkan proses pertukaran oksigen dan karbon dioksida dalam darah menjadi terganggu",
        "gejala": ["Demam tinggi", "Sesak napas", "Batuk", "Dahak", "Nyeri dada"],
        "penanganan": "Jika disebabkan oleh bakteri, maka dapat diberikan antibiotik seperti amoksisilin atau azitromisin. Apabila disebabkan oleh virus, pengobatannya lebih difokuskan pada perawatan suportif, seperti istirahat yang cukup, mengonsumsi cairan dalam jumlah cukup, dan pemberian antipiretik.",
    },
    "ppok": {
        "nama": "Penyakit Paru Obstruktif Kronik (PPOK)",
        "deskripsi": "Penyakit paru kronis yang ditandai dengan penyempitan saluran napas dan kerusakan jaringan paru-paru akibat paparan jangka panjang terhadap iritan berbahaya, terutama asap rokok",
        "gejala": ["Sesak napas", "Batuk", "Dahak", "Napas mengi"],
        "penanganan": "Berlangsung dalam jangka panjang, meliputi penghentian merokok, penggunaan bronkodilator (beta-agonis dan antikolinergik), penggunaan anti-inflamasi seperti kortikosteroid, inhalasi, dan PDE4 inhibitor, serta rehabilitasi paru,",
    },
    "asma": {
        "nama": "Asma",
        "deskripsi": " Gangguan pernapasan kronis yang ditandai dengan penyempitan saluran napas secara berulang sehingga penderita mengalami kesulitan bernapas",
        "gejala": ["Sesak napas", "Batuk", "Nyeri dada", "Napas mengi"],
        "penanganan": "Berlangsung jangka panjang biasanya menggunakan kortikosteroid inhalasi sebagai terapi utama, yang dapat dikombinasikan dengan long-acting beta agonist (LABA) untuk mengontrol gejala. Untuk serangan akut, digunakan short-acting beta agonist (SABA)} seperti salbutamol, dan pada kondisi berat dapat diberikan kortikosteroid sistemik atau terapi oksigen tambahan",
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

# --- 3. Triangular Membership Function ---
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

# --- 4. Implikasi Mamdani ---
def implication(alpha, params, y):
    return np.minimum(alpha, np.array([trimf(val, params) for val in y]))

# --- 5. Defuzzifikasi MoM ---
def defuzzify_mom(y, mu):
    max_mu = np.max(mu)
    if max_mu == 0:
        return 0.0
    y_max = y[mu == max_mu]
    return (y_max[0] + y_max[-1]) / 2

# --- 6. Inferensi Fuzzy dengan Bobot dan Pelacakan ---
def fuzzy_inference_mamdani_weighted(inputs, mf, rules, output_mf, y_domain):
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

    aggregated = np.zeros_like(y_domain)
    per_disease = {}

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

        clipped = implication(alpha, output_mf[disease], y_domain)
        aggregated = np.maximum(aggregated, clipped)
        if disease not in per_disease:
            per_disease[disease] = clipped
        else:
            per_disease[disease] = np.maximum(per_disease[disease], clipped)

    z_star = defuzzify_mom(y_domain, aggregated)
    return z_star, per_disease, aggregated


# --- 7. Normalisasi N Teratas ---
def get_top_diagnoses(per_disease, y_domain):
    """
    Menormalisasi dan mengambil n hasil teratas.
    Returns:
        confidences: Dictionary persentase kepercayaan
        top_n: List n penyakit teratas
    """
    scores = {}
    for disease, mu in per_disease.items():
        max_mu = np.max(mu)
        if max_mu > 0:
            scores[disease] = max_mu
    sorted_top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    top_total = sum(v for _, v in sorted_top)
    return [(d, v, 100 * v / top_total if top_total else 0) for d, v in sorted_top]

# --- 8. Input Gejala dari Pengguna ---

label_map = {
    "demam": "Suhu Tubuh (°C)",
    "sakit_kepala": "Sakit Kepala",
    "nyeri_menelan": "Nyeri Saat Menelan",
    "pembengkakan_amandel": "Pembengkakan Amandel",
    "sakit_tenggorokan": "Sakit Tenggorokan",
    "batuk": "Batuk",
    "dahak": "Dahak",
    "napas_mengi": "Napas Mengi",
    "nyeri_dada": "Nyeri Area Dada",
    "sesak_napas": "Sesak saat Bernapas",
    "bersin": "Bersin-Bersin",
    "hidung_berair": "Hidung Berair (Pilek)",
    "hidung_gatal": "Hidung Terasa Gatal",
    "hidung_tersumbat": "Hidung Tersumbat"
}

def get_user_inputs(mf, cmap):
    """
    Mengambil input gejala dari pengguna melalui antarmuka Streamlit.
    Returns:
    inp: Dictionary nilai input untuk setiap gejala
    """
    inp = {}
    for grp, gs in cmap.items():
        st.subheader(f"{grp.title()}")
        col1, col2, col3, col4 = st.columns(4)
        cols = [col1, col2, col3, col4]
        for i, g in enumerate(sorted(gs)):
            if g not in mf:
                continue
            pts = [p for (a, b, c) in mf[g].values() for p in (a, b, c)]
            lo, hi = min(pts), max(pts)
            default = lo
            step = 0.1 if hi - lo <= 10 else 0.5
            with cols[i % 4]:
                with st.container():
                    st.markdown(f'<div style="background-color: #f0f2f6; border-radius: 10px; margin-bottom: 10px; padding: 5px;">', unsafe_allow_html=True)
                    label = label_map.get(g, g.replace('_', ' ').title())
                    st.markdown(
                        f"<div style='font-weight: bold; font-size: 16px; color: #155799; margin-bottom: 5px;'>"
                        f"{label} ({lo:.1f}–{hi:.1f})</div>",
                        unsafe_allow_html=True
                    )
                    value = st.number_input(label="", min_value=lo, max_value=hi, value=default, step=step, key=g)
                    inp[g] = value
                    st.markdown("</div>", unsafe_allow_html=True)
    return inp

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
    mf = "revisi_member_function.csv"
    rules_file = "rules_bobot_respirasi.csv"
    output_mf_file = "output_member_function.csv"

    mf, cmap = load_membership_functions(mf)
    rules = load_rules_with_weights(rules_file)
    output_df = pd.read_csv(output_mf_file)
    output_mf = {r['penyakit']: (r['a'], r['b'], r['c']) for _, r in output_df.iterrows()}
    y_domain = np.linspace(0, 10, 1000)

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
                            <h3 style="font-size: 40px; font-weight: bold;">SMART DIAGNOSIS FOR RESPIRATORY HEALTH</h3>
                            <p style="font-size: 20px; color: #333;">Detect 10 types of respiratory diseases instantly using fuzzy inference system.</p>
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
        st.title("Diagnosis Penyakit Respirasi Menggunakan Fuzzy Inference System")
        st.markdown(
    """
    <div style='padding: 20px; background-color: #e6f2fa; border-left: 6px solid #1F77B4; border-radius: 8px; margin-top: 15px;'>
        <h4 style='color: #1F77B4; margin-bottom: 10px;'>📋 Petunjuk Pengisian</h4>
        <ul style='font-size: 16px; color: #1a1a1a; line-height: 1.6; margin-left: 20px;'>
            <li>Masukkan <strong>suhu tubuh</strong> Pasien (dalam °C) sesuai kondisi saat ini.</li>
            <li>Isi <strong>tingkat keparahan gejala</strong> lainnya pada rentang nilai <strong>0 hingga 10</strong> (semakin besar nilai maka gejala semakin parah).</li>
            <li>Jika Pasien <strong>tidak mengalami gejala tertentu</strong>, isi dengan nilai <strong>0</strong>.</li>
            <li>Input dapat berupa <strong>bilangan desimal</strong> (misalnya: 5.5, 7.0).</li>
            <li>Gunakan tombol <strong>–</strong> dan <strong>+</strong> di sisi kanan input untuk mengurangi atau menambah nilai.</li>
            <li>Setelah semua terisi, klik tombol <strong>“Diagnosis”</strong> untuk melihat hasil analisis.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

        # Get User Inputs
        inputs = get_user_inputs(mf, cmap)

        # Perform Fuzzy Inference
        if st.button("Diagnosis", key="diagnosis_run_button"):
            z_star, per_disease, aggregated = fuzzy_inference_mamdani_weighted(inputs, mf, rules, output_mf, y_domain)
            top3_result = get_top_diagnoses(per_disease, y_domain)

            # Display Results
            st.subheader("Hasil Diagnosis")

            # Create DataFrame for Table - Only top 3
            df = pd.DataFrame({
                "Penyakit": [d for d, _, _ in top3_result],
                "Kemungkinan (%)": [p for _, _, p in top3_result]
            })

            # Display Table and Chart in more compact layout
            col1, col2 = st.columns([1.2, 1])
            with col1:
                for index, row in df.iterrows():
                    st.markdown(
                        f"""
                            <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; background-color: rgba(255,255,255,0.1); padding: 12px; border-radius: 5px; font-size: 18px; font-weight: 500;'>
                            <div>{row['Penyakit'].capitalize()}</div>
                            <div style='background-color: #1F77B4; color: white; padding: 6px 12px; border-radius: 5px; font-size: 16px; font-weight: bold;'>{row['Kemungkinan (%)']:.1f}%</div>
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
                <div class="info-text">{info['penanganan']}</div>
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
                    <span class="about-highlight">Respirazzy</span> adalah Sistem Pendukung Keputusan (Decision Support System) berbasis website
                    yang dirancang untuk membantu dalam deteksi dini dan klasifikasi penyakit pada sistem pernapasan menggunakan metode Fuzzy Inference System.
                    Sistem cerdas ini mengintegrasikan gejala yang diinput oleh pengguna dengan untuk menganalisis dan menentukan diagnosis yang paling mungkin dari 10 jenis penyakit pernapasan.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )