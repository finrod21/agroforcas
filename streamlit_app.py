import json
import os
import pandas as pd
import streamlit as st
from models.growth_model import (
    predict_growth_stage,
    predict_harvest_days,
    evaluate_plant_condition,
    generate_care_recommendations,
    compute_health_score,
)

# App configurations
st.set_page_config(
    page_title="AgroForcast - AI Crop Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injected CSS for custom dark-mode glassmorphic aesthetics
st.markdown(
    """
    <style>
    /* Import Plus Jakarta Sans and Space Grotesk */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300..800;1,300..800&family=Space+Grotesk:wght@500;700&display=swap');
    
    /* Apply main font override */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Header fonts */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
    }
    
    /* Custom styled Streamlit metrics cards */
    div[data-testid="stMetric"] {
        background: rgba(18, 38, 23, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(16, 185, 129, 0.12) !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Override metric labels/values styles */
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }
    
    /* Input field styling */
    div[data-baseweb="input"], select, div[data-baseweb="select"], input, textarea {
        background-color: rgba(0, 0, 0, 0.25) !important;
        border: 1px solid rgba(16, 185, 129, 0.15) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    /* Buttons customization */
    button[kind="primary"], button[kind="secondaryFormSubmit"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover, button[kind="secondaryFormSubmit"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
        filter: brightness(1.1) !important;
    }
    
    /* Custom care cards */
    .care-card {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-left: 3px solid #10b981 !important;
        padding: 0.9rem 1.1rem !important;
        border-radius: 10px !important;
        margin-bottom: 0.7rem !important;
        font-size: 0.9rem !important;
        color: #94a3b8 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .care-card:hover {
        background: rgba(16, 185, 129, 0.05) !important;
        color: #f8fafc !important;
        transform: translateX(4px) !important;
        border-left-color: #34d399 !important;
    }
    
    /* Custom status badges */
    .status-badge {
        padding: 0.8rem 1.4rem !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        display: inline-flex !important;
        align-items: center;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    .status-sehat {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(52, 211, 153, 0.15) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        color: #34d399 !important;
    }
    .status-cukup {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(253, 224, 71, 0.1) 100%) !important;
        border: 1px solid rgba(234, 179, 8, 0.25) !important;
        color: #facc15 !important;
    }
    .status-perhatian {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(248, 113, 113, 0.12) 100%) !important;
        border: 1px solid rgba(239, 68, 68, 0.25) !important;
        color: #f87171 !important;
    }
    
    /* Glassmorphic alerts */
    .stAlert {
        background: rgba(18, 38, 23, 0.3) !important;
        border: 1px solid rgba(16, 185, 129, 0.12) !important;
        border-radius: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'weekly_data.json')
PLANTS = {
    'cabai': {
        'label': 'Cabai',
        'icon': '🌶️',
        'description': 'Tanaman cabai membutuhkan sinar matahari penuh, tanah gembur, dan penyiraman teratur.',
    },
    'tomat': {
        'label': 'Tomat',
        'icon': '🍅',
        'description': 'Tomat tumbuh baik pada suhu hangat dan kelembapan seimbang dengan pemupukan rutin.',
    },
    'terong': {
        'label': 'Terong',
        'icon': '🍆',
        'description': 'Terong membutuhkan paparan sinar matahari setidaknya 6 jam per hari dan drainase yang baik.',
    },
}


def load_data():
    if not os.path.exists(DATA_FILE):
        return {plant: [] for plant in PLANTS}

    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as data_file:
            return json.load(data_file)
    except Exception:
        return {plant: [] for plant in PLANTS}


def save_data(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as data_file:
        json.dump(data, data_file, ensure_ascii=False, indent=2)


# Load database
data = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='font-size: 2.2rem; color: #10b981; font-family: "Space Grotesk", sans-serif; margin-bottom: 5px;'>🌱 AgroForcast</h1>
        <p style='color: #64748b; font-size: 0.85rem;'>Dashboard AI Pemantauan Tanaman</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# Selection
selected_plant_key = st.sidebar.selectbox(
    "Pilih Komoditas Tanaman",
    list(PLANTS.keys()),
    format_func=lambda k: f"{PLANTS[k]['icon']} {PLANTS[k]['label']}",
)

plant_info = PLANTS[selected_plant_key]
records = data.get(selected_plant_key, [])

st.sidebar.markdown(
    f"""
    <div style='background-color: rgba(16, 185, 129, 0.08); padding: 15px; border-radius: 12px; border-left: 4px solid #10b981; margin-top: 15px;'>
        <h4 style='color: #34d399; margin-top: 0;'>{plant_info['icon']} Deskripsi Tanaman</h4>
        <p style='font-size: 0.88rem; color: #94a3b8; line-height: 1.6; margin-bottom: 0;'>{plant_info['description']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# Sidebar Guide
st.sidebar.markdown(
    """
    ### 📖 Panduan Penggunaan
    1. **Pilih Tanaman** Anda di menu dropdown di atas.
    2. **Catat Data Baru** secara berkala setiap minggu (Tab 3).
    3. **Periksa Hasil Analisis AI** mengenai kesehatan & perkiraan waktu panen (Tab 1).
    4. **Pantau Grafik Pertumbuhan** untuk melihat tren perkembangan (Tab 2).
    """
)

# ----------------- MAIN PANEL -----------------
# Header banner
st.markdown(
    f"""
    <div style='background: linear-gradient(135deg, #0d1e11 0%, #050b06 100%); padding: 30px; border-radius: 20px; border: 1px solid rgba(16, 185, 129, 0.15); margin-bottom: 25px;'>
        <h1 style='font-family: "Space Grotesk", sans-serif; font-size: 2.5rem; margin-top: 0; background: linear-gradient(135deg, #fff 30%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{plant_info['icon']} AgroForcast {plant_info['label']}</h1>
        <p style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 0;'>Pantau tumbuh kembang dan estimasi panen realtime berbasis teknologi AI terintegrasi.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs definitions (Flask tab deleted, back to 3 clean tabs)
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Ringkasan & Rekomendasi AI",
        "📈 Tren Tumbuh & Riwayat",
        "📝 Catat Data Mingguan",
    ]
)

# ----------------- TAB 1: SUMMARY & AI PREDICTIONS -----------------
with tab1:
    if records:
        last = records[-1]
        
        # Calculate AI health score
        health_score = compute_health_score(
            float(last.get('age_days', 0)),
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            float(last.get('temperature_c', 0)),
            float(last.get('humidity_pct', 0)),
        )
        
        # Predict growth stage
        growth_stage = predict_growth_stage(
            float(last.get('height_cm', 0)),
            int(last.get('leaf_count', 0)),
            health_score,
        )
        
        # Evaluate condition
        plant_condition = evaluate_plant_condition(
            float(last.get('age_days', 0)),
            float(last.get('temperature_c', 0)),
            float(last.get('humidity_pct', 0)),
        )
        
        # Generate care recommendations
        care_recommendations = generate_care_recommendations(
            plant_condition,
            float(last.get('temperature_c', 0)),
            float(last.get('humidity_pct', 0)),
            float(last.get('age_days', 0)),
        )
        
        # Predict harvest estimate
        try:
            harvest_estimate = predict_harvest_days(
                float(last.get('age_days', 0)),
                float(last.get('height_cm', 0)),
                int(last.get('leaf_count', 0)),
                float(last.get('humidity_pct', 0)),
                float(last.get('temperature_c', 0)),
                plant_condition,
                selected_plant_key,
            )
        except Exception:
            harvest_estimate = None

        # Custom Badge layout for condition
        badge_style = "status-sehat"
        if plant_condition == "Cukup Baik":
            badge_style = "status-cukup"
        elif plant_condition == "Perlu Perhatian":
            badge_style = "status-perhatian"

        st.markdown(
            f"""
            <div>
                <span style='font-size: 0.95rem; color: #94a3b8; font-weight: 600; margin-right: 10px;'>Status Kesehatan Terkini:</span>
                <span class='status-badge {badge_style}'>📊 {plant_condition}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics display columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Skor Kesehatan AI",
                value=f"{health_score} / 10",
                help="Dihitung otomatis berbasis perbandingan proporsional tinggi & daun terhadap usia tanaman.",
            )
        with col2:
            if harvest_estimate is not None:
                st.metric(
                    label="Estimasi Sisa Hari Panen",
                    value=f"{round(harvest_estimate, 1)} hari lagi",
                )
            else:
                st.metric(
                    label="Estimasi Sisa Hari Panen",
                    value="Belum tersedia",
                )
        with col3:
            st.metric(
                label="Total Log Pengamatan",
                value=f"{len(records)} Minggu",
            )

        st.info(f"🌿 **Prediksi Tahap Pertumbuhan Terakhir**: {growth_stage}")

        st.divider()
        st.subheader("🌱 Panduan & Rekomendasi Perawatan AI")
        
        # Recommendations Columns with premium cards styling
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        with rec_col1:
            st.markdown("### 💧 Penyiraman")
            for rec in care_recommendations['watering']:
                st.markdown(f"<div class='care-card'>✓ {rec}</div>", unsafe_allow_html=True)
                
        with rec_col2:
            st.markdown("### 🌾 Pemupukan")
            for rec in care_recommendations['fertilizer']:
                st.markdown(f"<div class='care-card'>✓ {rec}</div>", unsafe_allow_html=True)
                
        with rec_col3:
            st.markdown("### ⚙️ Tindakan Khusus")
            for rec in care_recommendations['special_actions']:
                # Clean up icon characters if present
                clean_rec = rec.replace("⚠️ ", "").replace("✓ ", "")
                icon = "⚠️" if "⚠️" in rec else "✓"
                st.markdown(f"<div class='care-card'>{icon} {clean_rec}</div>", unsafe_allow_html=True)

    else:
        st.warning(
            "Belum ada catatan data perkembangan untuk tanaman ini. Silakan buka tab **Catat Data Mingguan** untuk menambahkan data pertama Anda."
        )


# ----------------- TAB 2: HISTORY & CHART -----------------
with tab2:
    if records:
        df = pd.DataFrame(records)
        
        st.subheader("📈 Tren Tumbuh Kembang Tanaman")
        
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Perkembangan Tinggi Tanaman (cm)**")
            st.line_chart(df.set_index('week')['height_cm'], color="#10b981")
            
        with chart_col2:
            st.markdown("**Penambahan Jumlah Daun**")
            st.line_chart(df.set_index('week')['leaf_count'], color="#06b6d4")

        st.subheader("📋 Riwayat Lengkap Pencatatan")
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("🗑/ Hapus Catatan Perkembangan")
        
        # Deletion logic UI
        weeks_list = [
            f"{idx}: {row['week']} (Usia: {row['age_days']} hari, Tinggi: {row['height_cm']} cm)"
            for idx, row in enumerate(records)
        ]
        selected_record_to_delete = st.selectbox(
            "Pilih catatan mingguan yang ingin dihapus secara permanen:",
            weeks_list,
        )
        if st.button("Hapus Catatan Terpilih", type="primary"):
            idx_to_delete = int(selected_record_to_delete.split(":")[0])
            deleted_week = records[idx_to_delete]['week']
            records.pop(idx_to_delete)
            data[selected_plant_key] = records
            save_data(data)
            st.success(f"Catatan untuk {deleted_week} berhasil dihapus secara permanen!")
            st.rerun()
    else:
        st.info("Riwayat kosong. Data akan ditampilkan di sini setelah Anda melakukan pencatatan.")


# ----------------- TAB 3: RECORD DATA -----------------
with tab3:
    st.subheader("📝 Catat Ukuran Tanaman Mingguan")
    
    with st.form("add_weekly_record_form", clear_on_submit=True):
        default_week = f"Minggu {len(records) + 1}"
        
        week = st.text_input(
            "Periode Minggu",
            value=default_week,
            help="Contoh: Minggu 1, Minggu 2, dst.",
        )
        
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            age_days = st.number_input(
                "Usia Tanaman (hari)",
                min_value=0,
                step=1,
                help="Jumlah hari sejak penyemaian atau penanaman.",
            )
            height_cm = st.number_input(
                "Tinggi Tanaman (cm)",
                min_value=0.0,
                step=0.1,
                help="Tinggi tanaman diukur dari pangkal batang.",
            )
            leaf_count = st.number_input(
                "Jumlah Helai Daun",
                min_value=0,
                step=1,
                help="Hitung seluruh daun yang telah mekar sempurna.",
            )
            
        with col_form2:
            humidity_pct = st.number_input(
                "Kelembapan Lingkungan (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=0.1,
                help="Kelembapan udara di sekitar tempat budidaya.",
            )
            temperature_c = st.number_input(
                "Suhu Lingkungan (°C)",
                min_value=0.0,
                value=27.0,
                step=0.1,
                help="Suhu udara rata-rata harian.",
            )

        submit_btn = st.form_submit_button("💾 Simpan Catatan Perkembangan")
        
        if submit_btn:
            if not week.strip():
                st.error("Nama minggu/periode tidak boleh kosong!")
            else:
                new_record = {
                    'week': week.strip(),
                    'age_days': int(age_days),
                    'height_cm': float(height_cm),
                    'leaf_count': int(leaf_count),
                    'humidity_pct': float(humidity_pct),
                    'temperature_c': float(temperature_c),
                }
                
                records.append(new_record)
                data[selected_plant_key] = records
                save_data(data)
                
                st.success(f"Berhasil menyimpan data untuk {week.strip()}!")
                st.rerun()
