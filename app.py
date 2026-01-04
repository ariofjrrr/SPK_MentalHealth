import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ===========================
# CONFIG
# ===========================
st.set_page_config(page_title="SPK Kesehatan Mental", layout="wide")

# ===========================
# STYLE
# ===========================
st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
body, .block-container {
    background: linear-gradient(135deg, #fbc2eb, #a6c1ee, #fcd5ce);
    font-family:'Helvetica Neue', 'Segoe UI', sans-serif;
    padding-left:2rem;
    padding-right:2rem;
}
.title {font-size:36px; font-weight:800; text-align:center; color:#111827; margin-bottom:2px;}
.subtitle {font-size:14px; text-align:center; color:#111827; margin-bottom:25px;}
.card {border-radius:20px; padding:15px; margin-bottom:10px; background:white; box-shadow:0 6px 25px rgba(0,0,0,0.12); transition: transform 0.2s;}
.card:hover {transform: translateY(-6px); box-shadow:0 10px 30px rgba(0,0,0,0.15);}
.step-card {border-radius:15px; padding:15px; text-align:center; margin-bottom:12px; background: linear-gradient(135deg,#fcb045,#fd1d1d,#833ab4); color:white; font-weight:600; box-shadow:0 3px 12px rgba(0,0,0,0.08);}
.step-card:hover {transform: translateY(-5px);}
.step-title {font-size:15px; margin-bottom:5px;}
.step-desc {font-size:13px; color:white;}
.kpi-card {border-radius:15px; padding:20px; margin-bottom:20px; text-align:center; font-weight:700; color:white; box-shadow:0 4px 15px rgba(0,0,0,0.1);}
.kpi-card:hover {transform: translateY(-3px);}
.kpi-value {font-size:24px;}
.kpi-label {font-size:13px;}
.rank-card {border-radius:15px; padding:14px; background:white; color:#111827; margin-bottom:12px; box-shadow:0 2px 10px rgba(0,0,0,0.05); border-left:6px solid;}
.footer {margin-top:40px; text-align:center; font-size:12px; color:#111827;}
@media only screen and (max-width:768px) {.block-container {padding-left:1rem; padding-right:1rem;}}
</style>
""", unsafe_allow_html=True)

# ===========================
# HEADER & PANDUAN
# ===========================
st.markdown("<div class='title'>SPK Kesehatan Mental</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analisis Prioritas Intervensi · Metode AHP–TOPSIS</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><h3>Panduan Penggunaan Sistem</h3></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])
col1.markdown("<div class='step-card'><div class='step-title'>1. Unggah Data</div><div class='step-desc'>Unggah dataset CSV aktivitas, emosi, interaksi individu.</div></div>", unsafe_allow_html=True)
col2.markdown("<div class='step-card'><div class='step-title'>2. Pemrosesan</div><div class='step-desc'>Sistem menghitung prioritas menggunakan AHP & TOPSIS.</div></div>", unsafe_allow_html=True)
col3.markdown("<div class='step-card'><div class='step-title'>3. Hasil Analisis</div><div class='step-desc'>Lihat ranking, prioritas & rekomendasi interaktif.</div></div>", unsafe_allow_html=True)

# ===========================
# UPLOAD DATA
# ===========================
file = st.file_uploader("Unggah Dataset CSV", type="csv")
st.markdown("<div style='margin-bottom:10px; font-size:14px;'>Contoh dataset: <a href='https://github.com/ariofjrrr/SPK_MentalHealth/blob/main/MH.csv' target='_blank'>Download CSV</a></div>", unsafe_allow_html=True)

if file:
    df_original = pd.read_csv(file)

    # Tambahkan kolom Alternatif sesuai urutan asli data: U1, U2, ...
    df_original.insert(0, "Alternatif", [f"U{i+1}" for i in range(len(df_original))])

    # ===========================
    # PREPROCESS
    # ===========================
    cols = ["daily_screen_time_min","social_media_time_min","negative_interactions_count","positive_interactions_count","sleep_hours","physical_activity_min","anxiety_level","stress_level","mood_level"]
    missing_cols = [c for c in cols if c not in df_original.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}. Pastikan dataset sesuai format.")
        st.stop()

    df = df_original.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # ===========================
    # AGREGAT KRITERIA
    # ===========================
    df["C1"] = (df["daily_screen_time_min"] + df["social_media_time_min"]) / 2
    df["C2"] = df["negative_interactions_count"] / (df["negative_interactions_count"] + df["positive_interactions_count"] + 1e-6)
    df["C3"] = df["sleep_hours"]
    df["C4"] = df["physical_activity_min"]
    df["C5"] = (df["anxiety_level"] + df["stress_level"] + (10 - df["mood_level"])) / 3

    kriteria = ["C1", "C2", "C3", "C4", "C5"]
    bobot = {"C1": 0.18, "C2": 0.32, "C3": 0.11, "C4": 0.07, "C5": 0.32}
    benefit = ["C3", "C4"]

    # ===========================
    # TOPSIS
    # ===========================
    norm = df[kriteria].copy()
    for c in kriteria:
        norm[c] = df[c] / np.sqrt(np.sum(df[c]**2) + 1e-8)

    terbobot = norm.copy()
    for c in kriteria:
        terbobot[c] *= bobot[c]

    A_pos = terbobot.max().copy()
    A_neg = terbobot.min().copy()
    for c in kriteria:
        if c not in benefit:
            A_pos[c] = terbobot[c].min()
            A_neg[c] = terbobot[c].max()

    df["D+"] = np.sqrt(((terbobot - A_pos) ** 2).sum(axis=1))
    df["D-"] = np.sqrt(((terbobot - A_neg) ** 2).sum(axis=1))
    df["Ci"] = df["D-"] / (df["D+"] + df["D-"] + 1e-8)

    # Hitung ranking
    df["Rank"] = df["Ci"].rank(ascending=False, method='min').astype(int)

    # Buat df untuk ranking (diurutkan berdasarkan Ci tertinggi)
    df_ranked = df.sort_values("Ci", ascending=False).reset_index(drop=True)

    # ===========================
    # KPI
    # ===========================
    total = len(df)
    high = (df["Ci"] >= 0.95).sum()
    medium = ((df["Ci"] < 0.95) & (df["Ci"] >= 0.80)).sum()
    low = (df["Ci"] < 0.80).sum()
    max_ci = df["Ci"].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"<div class='kpi-card' style='background: linear-gradient(135deg,#fcb045,#fd1d1d,#833ab4)'><div class='kpi-value'>{total}</div><div class='kpi-label'>Total Alternatif</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi-card' style='background:#fcb045'><div class='kpi-value'>{high}</div><div class='kpi-label'>Prioritas Tinggi</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi-card' style='background:#fd1d1d'><div class='kpi-value'>{medium}</div><div class='kpi-label'>Prioritas Sedang</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi-card' style='background:#833ab4'><div class='kpi-value'>{low}</div><div class='kpi-label'>Prioritas Rendah</div></div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='kpi-card' style='background: linear-gradient(135deg,#fcb045,#fd1d1d,#833ab4)'><div class='kpi-value'>{max_ci:.4f}</div><div class='kpi-label'>Skor Tertinggi</div></div>", unsafe_allow_html=True)

    # ===========================
    # TABS
    # ===========================
    tab1, tab2, tab3 = st.tabs(["Data", "Proses", "Hasil"])

    with tab1:
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("<div class='card'><h4>Dataset Mentah dengan Agregat Kriteria</h4></div>", unsafe_allow_html=True)
            display_cols = ["Alternatif"] + cols + kriteria
            df_display = df[display_cols].copy()  # Urutan asli U1 sampai akhir
            numeric_cols = df_display.select_dtypes(include=[np.number]).columns
            styler = df_display.style.format({col: "{:.2f}" for col in numeric_cols})
            st.dataframe(styler, use_container_width=True)

        with col2:
            st.markdown("<div class='card'><h4>Bobot Kriteria (AHP)</h4></div>", unsafe_allow_html=True)
            bobot_df = pd.DataFrame.from_dict(bobot, orient="index", columns=["Bobot"])
            st.table(bobot_df.style.format("{:.2f}"))

    with tab2:
        st.markdown("<div class='card'><h4>Matriks Normalisasi Vektor</h4></div>", unsafe_allow_html=True)
        st.dataframe(norm.style.format("{:.2f}"), use_container_width=True)  # Urutan asli

        st.markdown("<div class='card'><h4>Matriks Terbobot</h4></div>", unsafe_allow_html=True)
        st.dataframe(terbobot.style.format("{:.2f}"), use_container_width=True)  # Urutan asli

        solusi_ideal = pd.DataFrame({
            "Kriteria": kriteria,
            "Jenis": ["Benefit" if c in benefit else "Cost" for c in kriteria],
            "Ideal Positif (A⁺)": A_pos.values,
            "Ideal Negatif (A⁻)": A_neg.values
        })
        st.markdown("<div class='card'><h4>Solusi Ideal TOPSIS</h4></div>", unsafe_allow_html=True)
        st.table(solusi_ideal.style.format({"Ideal Positif (A⁺)": "{:.2f}", "Ideal Negatif (A⁻)": "{:.2f}"}))

        st.markdown("<div class='card'><h4>Jarak ke Solusi Ideal (D+, D-, Ci)</h4></div>", unsafe_allow_html=True)
        jarak_df = df[["Alternatif", "D+", "D-", "Ci", "Rank"]].copy()  # Urutan asli + Rank
        jarak_styler = jarak_df.style.format({"D+": "{:.2f}", "D-": "{:.2f}", "Ci": "{:.2f}"})
        st.dataframe(jarak_styler, use_container_width=True)

    with tab3:
        st.markdown("<div class='card'><h4>Perankingan Alternatif (Urut dari Prioritas Tertinggi)</h4></div>", unsafe_allow_html=True)
        rank_cols = st.columns(2)
        for idx, r in enumerate(df_ranked.itertuples()):
            color = "#fcb045" if r.Ci >= 0.95 else "#fd1d1d" if r.Ci >= 0.80 else "#833ab4"
            with rank_cols[idx % 2]:
                st.markdown(f"<div class='rank-card' style='border-left-color:{color}'><b>{r.Alternatif}</b><br>Rank {r.Rank} | Skor Ci = {r.Ci:.4f}</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'><h4>Grafik Skor Preferensi (Ci) - Urutan Alternatif Asli</h4></div>", unsafe_allow_html=True)
        fig = px.bar(df, x="Alternatif", y="Ci", color="Ci",  # df = urutan asli
                     color_continuous_scale=["#833ab4","#fd1d1d","#fcb045"],
                     template="plotly_white", height=500)
        fig.update_layout(xaxis_title="Alternatif (U1 sampai akhir)", yaxis_title="Skor Ci (semakin tinggi semakin prioritas)")
        st.plotly_chart(fig, use_container_width=True)

# ===========================
# FOOTER
# ===========================
st.markdown("<div class='footer'>© 2026 Ario Fajar Dharmawan · SPK Kesehatan Mental dengan AHP-TOPSIS</div>", unsafe_allow_html=True)