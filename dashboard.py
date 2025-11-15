import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
import json 

# --- 0. KONFIGURASI HALAMAN & DATA ---

st.set_page_config(
    page_title="Dashboard Finansial Gen Z",
    page_icon="💡",
    layout="wide"
)

# --- (TETAP) KUSTOMISASI CSS (FONT POPPINS) ---
st.markdown("""
    <style>
    /* (BARU) Impor font Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* (PERUBAHAN) Mengganti font utama aplikasi ke Poppins */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Judul Utama Halaman */
    h1 {
        font-family: 'Poppins', sans-serif; 
        font-weight: 600;
        color: #1a1a1a;
        text-align: center
    }
    
    /* Judul Bagian (Temuan #1, #2, #3) */
    h2 {
        font-family: 'Poppins', sans-serif; 
        font-weight: 600;
        color: #0068c9; /* Warna biru yang bagus */
        border-bottom: 2px solid #f0f2f6; /* Garis bawah subtil */
        padding-bottom: 5px;
        text-align: center;
    }

    /* Judul Grafik (Grafik 1, 2, 3) */
    h3 {
        font-family: 'Poppins', sans-serif; 
        font-weight: 600;
        color: #333333;
        text-align: center;
    }

    /* (BARU) Kustomisasi Kartu KPI (st.metric) */
    [data-testid="stMetric"] {
        border: 1px solid #0068c9; /* Border biru */
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        background-color: #f0f2f6; /* Warna latar sedikit agar 'kotak'-nya terlihat */
    }
    
    /* (PERUBAHAN) Memaksa label KPI (Judul) untuk rata tengah */
    [data-testid="stMetricLabel"] {
        display: flex;              /* ubah jadi flex container */
        justify-content: center;    /* isi rata tengah horizontal */
        text-align: center;         /* backup alignment */
        width: 100%; 
    }

    /* (BARU) Subtitle di bawah Judul Utama */
    .centered-subtitle {
        text-align: center;
        color: #444; /* Sedikit lebih lembut dari hitam */
        font-size: 1.1em; /* Sedikit lebih besar */
    }

    /* (PERUBAHAN) Kotak Info Petunjuk */
    /* Kustomisasi seluruh box alert */
    .stAlert {
        border-radius: 10px;
        border: 1px solid #0068c9;
        background-color: #e6f0fa;
        margin-top: 15px;
    }

    /* Kustomisasi teks di dalam alert */
    .stAlert > div {
        font-size: 0.9em !important;   /* paksa override */
        /* text-align: center; */     /* Hapus rata tengah agar sebaris */
    }
    
    /* Tombol Radio Navigasi di Sidebar */
    .stRadio [role="radio"] {
        border: 2px solid #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    
    .stRadio [role="radio"]:has(input:checked) {
        background-color: #e6f0fa; /* Warna biru muda saat dipilih */
        border-color: #0068c9;
    }

    </style>
    """, unsafe_allow_html=True)


warnings.filterwarnings('ignore')

# --- (REVISI BESAR) Fungsi Load Data (Caching) ---
@st.cache_data
def load_data():
    """
    Memuat data MIKRO (survey & profile) untuk visualisasi insight regresi.
    """
    try:
        # Data untuk insight Literasi, Impulsif, Gender, PDRB, Adopsi
        df_survey = pd.read_csv('survey_lengkap_CLEAN.csv', encoding='utf-8-sig')
        
        # --- (REVISI) PERBAIKAN "0 RESPONDEN" ---
        # Normalisasi 'Province of Origin' di df_survey
        if 'Province of Origin' in df_survey.columns:
             df_survey['Province of Origin'] = df_survey['Province of Origin'].str.strip().str.lower()
        # ----------------------------------------
        
        # Terjemahkan nama kolom teknis ke bahasa manusia
        df_survey = df_survey.rename(columns={
            'PDRB (Ribu Rp)': 'Kekayaan Rata-Rata (PDRB)',
            'Skor_Literasi': 'Skor Melek Finansial',
            'Skor_Impulsif': 'Skor Impulsif',
            'Skor_Adopsi_Fintech': 'Skor Adopsi Fintech'
        })

        # --- PERUBAHAN UTAMA (BINNING) ---
        
        # 1. Buat grup untuk 'Skor Melek Finansial' (3 grup)
        try:
            literasi_labels = ["Rendah", "Sedang", "Tinggi"] 
            df_survey['Grup Melek Finansial'] = pd.qcut(
                df_survey['Skor Melek Finansial'].rank(method='first'), 
                q=3, 
                labels=literasi_labels 
            )
        except Exception as e:
            st.warning(f"Gagal membuat grup literasi: {e}")
            df_survey['Grup Melek Finansial'] = "N/A" # Fallback

        # 2. Buat grup untuk 'Kekayaan Rata-Rata (PDRB)' (3 grup)
        try:
            pdrb_labels = ["Rendah", "Menengah", "Tinggi"] 
            df_survey['Grup Kekayaan (PDRB)'] = pd.qcut(
                df_survey['Kekayaan Rata-Rata (PDRB)'].rank(method='first'), 
                q=3, 
                labels=pdrb_labels 
            )
        except Exception as e:
            st.warning(f"Gagal membuat grup PDRB: {e}")
            df_survey['Grup Kekayaan (PDRB)'] = "N/A" # Fallback
            
    except FileNotFoundError:
        st.error("File 'survey_lengkap_CLEAN.csv' tidak ditemukan.")
        df_survey = None
        
    # (BARU) Load data regional untuk Halaman 2
    try:
        regional = pd.read_csv('regional_clean.csv', encoding='utf-8-sig')
        regional['Provinsi'] = regional['Provinsi'].str.strip().str.lower()
    except FileNotFoundError:
        st.error("File 'regional_clean.csv' tidak ditemukan. Halaman 2 tidak akan berfungsi.")
        regional = None
        
    try:
        # Data untuk insight Kecemasan vs Pendapatan
        df_profile = pd.read_csv('profile_lengkap_CLEAN.csv', encoding='utf-8-sig')
        
        # Normalisasi nama provinsi di file profile (untuk Halaman 2)
        if 'province' in df_profile.columns:
            df_profile['province'] = df_profile['province'].str.strip().str.lower()
        # (REVISI) Baris 'Province of Origin' dihapus dari sini karena salah
             
        df_profile = df_profile.rename(columns={
            'financial_anxiety_score': 'Skor Kecemasan',
            'income_clean': 'Pendapatan',
            'age': 'Usia'
        })

        # 3. Buat grup untuk 'Pendapatan' (3 Grup)
        try:
            pendapatan_labels = ["Rendah", "Sedang", "Tinggi"] 
            df_profile['Grup Pendapatan'] = pd.qcut(
                df_profile['Pendapatan'].rank(method='first'),
                q=3, 
                labels=pendapatan_labels
            )
        except Exception as e:
            st.warning(f"Gagal membuat grup pendapatan: {e}")
            df_profile['Grup Pendapatan'] = "N/A"

        # 4. Buat grup untuk 'Usia' (2 Grup)
        try:
            usia_labels = ["Muda", "Tua"] 
            df_profile['Grup Usia'] = pd.qcut(
                df_profile['Usia'].rank(method='first'),
                q=2, 
                labels=usia_labels
            )
        except Exception as e:
            st.warning(f"Gagal membuat grup usia: {e}")
            df_profile['Grup Usia'] = "N/A"

    except FileNotFoundError:
        st.error("File 'profile_lengkap_CLEAN.csv' tidak ditemukan.")
        df_profile = None

    # (REVISI) Inisialisasi df_master
    df_master = None
    
    # (REVISI) Load data Master Klaster (dari Fina/Aza)
    try:
        df_master = pd.read_csv('df_master_cluster.csv', encoding='utf-8-sig')
        
        # TAHAP 1: Normalisasi Awal (Nama di CSV)
        df_master['Provinsi'] = df_master['Provinsi'].str.strip().str.lower()

        # TAHAP 2: KAMUS LATITUDE/LONGITUDE
        # Ini menggantikan kebutuhan file GeoJSON
        lat_lon_map = {
            'aceh': (4.6951, 96.7494),
            'sumatera utara': (2.1154, 99.5451),
            'sumatera barat': (-0.7392, 100.8000),
            'riau': (0.2933, 101.7068),
            'jambi': (-1.6101, 103.6131),
            'sumatera selatan': (-3.3194, 103.9144),
            'bengkulu': (-3.5778, 102.3464),
            'lampung': (-4.5586, 105.4068),
            'kepulauan bangka belitung': (-2.7410, 106.4406), # Dari CSV Anda
            'kepulauan riau': (3.9457, 108.1428),
            'dki jakarta': (-6.2088, 106.8456), # Dari CSV Anda
            'jawa barat': (-6.9175, 107.6191),
            'jawa tengah': (-7.1509, 110.1403),
            'di yogyakarta': (-7.7956, 110.3695), # Dari CSV Anda
            'jawa timur': (-7.5361, 112.2384),
            'banten': (-6.4058, 106.0640),
            'bali': (-8.4095, 115.1889),
            'nusa tenggara barat': (-8.6529, 117.3616),
            'nusa tenggara timur': (-8.6574, 121.0794),
            'kalimantan barat': (-0.2787, 111.4753),
            'kalimantan tengah': (-1.6815, 113.3823),
            'kalimantan selatan': (-3.0926, 115.2838),
            'kalimantan timur': (0.5387, 116.4194),
            'kalimantan utara': (2.9796, 116.3330),
            'sulawesi utara': (0.6247, 123.9750),
            'sulawesi tengah': (-1.4300, 121.4456),
            'sulawesi selatan': (-3.6447, 119.9421),
            'sulawesi tenggara': (-4.1444, 122.1746),
            'gorontalo': (0.6999, 122.4467),
            'sulawesi barat': (-2.8548, 119.2317),
            'maluku': (-3.2384, 130.1453),
            'maluku utara': (1.5709, 127.8087),
            'papua barat': (-1.3362, 133.1747), # Dari CSV Anda
            'papua': (-4.2699, 138.0804),
            'papua tengah': (-4.0, 136.0),           
            'papua pegunungan': (-4.15, 139.4),      
            'papua selatan': (-6.8, 139.25),         
            'papua barat daya': (-0.86, 131.25)
        }

        # TAHAP 3: Buat kolom Lat dan Lon di df_master
        df_master['Latitude'] = df_master['Provinsi'].map(lambda x: lat_lon_map.get(x, (None, None))[0])
        df_master['Longitude'] = df_master['Provinsi'].map(lambda x: lat_lon_map.get(x, (None, None))[1])
        
        # Cek jika ada yang gagal di-mapping
        if df_master['Latitude'].isnull().any():
            st.sidebar.warning("Ada provinsi di CSV yang tidak ada di kamus Lat/Lon:")
            st.sidebar.dataframe(df_master[df_master['Latitude'].isnull()]['Provinsi'])
            
        # Terjemahkan nama kolom teknis (kode lama Anda, sudah benar)
        df_master = df_master.rename(columns={
            'PDRB (Ribu Rp)_x': 'Kekayaan Rata-Rata (PDRB)',
            'Outstanding Pinjaman (Rp miliar)_x': 'Total Pinjaman Beredar (Miliar)',
            'TWP 90%_x': 'Risiko Kredit Macet (%)'
        })
        
    except FileNotFoundError:
        st.sidebar.warning("File 'df_master_cluster.csv' (Data Klaster) belum ada.")
        # df_master akan tetap None
    except Exception as e:
        st.error(f"Gagal memuat df_master_cluster: {e}")

    # (REVISI) Hapus 'geojson' dari return
    return df_survey, df_profile, regional, df_master

# --- (REVISI) Load Semua Data ---
# Hapus 'geojson' dari proses unpack
df_survey, df_profile, regional, df_master = load_data()

# Jika data inti gagal dimuat, hentikan aplikasi
if df_survey is None or df_profile is None or regional is None:
    st.error("Data CSV inti (profile, survey, regional) gagal dimuat. Harap periksa file.")
    st.stop()
    
# --- (TETAP) Hitung KPI Global ---
try:
    total_responden_survey = len(df_survey)
    total_responden_profile = len(df_profile)
    avg_kecemasan_nasional = df_profile['Skor Kecemasan'].mean()
except Exception as e:
    st.error(f"Gagal menghitung KPI: {e}")
    total_responden_survey = 0
    total_responden_profile = 0
    avg_kecemasan_nasional = 0


# --- (TETAP) Sidebar Navigasi ---
st.sidebar.title("Fitur Dashboard")
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "Halaman 1: Dashboard Nasional", 
        "Halaman 2: Dashboard Regional"
    ]
)

# ===================================================================
# 📌 HALAMAN 1 – INSIGHT UTAMA (NASIONAL)
# ===================================================================
if page == "Halaman 1: Dashboard Nasional":
    
    st.title("Dashboard Kondisi Finansial Gen Z di Indonesia")
    # (TETAP) Menggunakan st.markdown dengan HTML/CSS kustom
    st.markdown(
        "<div class='centered-subtitle'>Dashboard ini memvisualisasikan hasil analisis korelasi dan regresi seluruh responden, serta clustering untuk provinsi.</div>", 
        unsafe_allow_html=True
    )
    
    # --- (TETAP) PETUNJUK PENGGUNA (STRATEGI UX KITA) ---
    st.info(
        "**Petunjuk:** Arahkan mouse Anda ke ikon tanda tanya `(?)` di sebelah "
        "setiap judul grafik untuk melihat penjelasan singkat",
    )

    # --- (TETAP) Tiga Kartu KPI ---
    st.markdown("---")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric(
            label="**Jumlah Responden (Survei Psikologis)**", 
            value=f"{total_responden_survey:,.0f}"
        )
    with col_kpi2:
        st.metric(
            label="**Jumlah Responden (Profil Finansial)**", 
            value=f"{total_responden_profile:,.0f}"
        )
    with col_kpi3:
        st.metric(
            label="**Skor Kecemasan Finansial Gen Z**", 
            value=f"{avg_kecemasan_nasional:.2f} / 5",
            help="Skor rata-rata kecemasan finansial nasional dari 1.000 responden profil."
        )
    
    # --- (TETAP) TEMUAN #1: PETA KLASTER (SCATTER_GEO) ---
    st.markdown("---") 
    
    if df_master is None:
        st.warning("⚠️ **Data Klaster Belum Ada**\n\nFile `df_master_cluster.csv` tidak ditemukan. Peta tidak dapat ditampilkan.")
    else:
        st.subheader(
            "Peta Segmentasi Provinsi (Peta Titik)",
            help=(
                "Peta ini mengelompokkan provinsi berdasarkan persona finansial Gen Z. "
                "Ukuran titik menunjukkan total pinjaman. "
                "Arahkan mouse ke titik untuk melihat detail klaster dan skornya."
            )
        )
        
        # Cek apakah kolom Lat/Lon ada (dibuat di load_data)
        if 'Latitude' not in df_master.columns or df_master['Latitude'].isnull().all():
            st.error("Gagal membuat peta titik. Kolom Latitude/Longitude tidak ditemukan atau kosong.")
            st.info("Pastikan kamus `lat_lon_map` di dalam `load_data()` sudah benar.")
        else:
            # Buat peta gelembung (scatter_geo)
            fig_map = px.scatter_geo(
                df_master,
                lat='Latitude',
                lon='Longitude',
                color="Cluster_Best", # Kolom klaster Anda
                hover_name="Provinsi",
                size="Total Pinjaman Beredar (Miliar)", 
                projection="mercator", 
                hover_data={
                    "Skor_Literasi": ":.2f",
                    "Skor_Rasa_Aman": ":.2f",
                    "Kekayaan Rata-Rata (PDRB)": ":,.0f",
                    "Risiko Kredit Macet (%)": ":.2f",
                    "Latitude": False, 
                    "Longitude": False
                }
            )
            
            # Zoom ke Indonesia
            fig_map.update_geos(
                scope='asia', 
                center=dict(lat=-2, lon=118), 
                lataxis_range=[-11, 6], 
                lonaxis_range=[94, 142] 
            )
            
            fig_map.update_layout(
                margin={"r":0,"t":10,"l":0,"b":0}, 
                height=500,
                font_family="Poppins"
            )
            st.plotly_chart(fig_map, use_container_width=True)

    
    # --- (REVISI) BAGIAN BARU UNTUK BEDAH CLUSTER ---
    st.markdown("---")
    
    if df_master is None:
        st.warning("⚠️ **Data Klaster Belum Ada**\n\nGrafik bedah cluster tidak dapat ditampilkan.")
    else:
        # Siapkan data: Agregasi rata-rata per cluster
        try:
            # Kolom yang ingin kita bandingkan
            cluster_cols = ['Risiko Kredit Macet (%)', 'Kekayaan Rata-Rata (PDRB)', 'Total Pinjaman Beredar (Miliar)']
            
            # Hitung rata-rata dari 3 faktor itu, dikelompokkan per cluster
            df_cluster_grouped = df_master.groupby('Cluster_Best')[cluster_cols].mean().reset_index()

            # Buat 3 kolom untuk 3 grafik
            col_c1, col_c2, col_c3 = st.columns(3)

            with col_c1:
                st.subheader(
                    "Rata-rata Risiko Kredit (%)",
                    help="Grafik ini membuktikan julukan 'Padat & Berisiko' untuk Cluster 1, yang memiliki rata-rata gagal bayar tertinggi."
                )
                fig_risk = px.bar(
                    df_cluster_grouped,
                    x='Cluster_Best',
                    y='Risiko Kredit Macet (%)',
                    color='Cluster_Best', # Warnai bar berdasarkan cluster
                    labels={'Cluster_Best': 'Cluster', 'Risiko Kredit Macet (%)': 'Rata-rata Risiko (%)'}
                )
                fig_risk.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins", showlegend=False)
                fig_risk.update_traces(marker_cornerradius=5)
                st.plotly_chart(fig_risk, use_container_width=True)

            with col_c2:
                st.subheader(
                    "Rata-rata PDRB",
                    help="Grafik ini membuktikan julukan 'Pusat Ekonomi' untuk Cluster 2 (DKI Jakarta), yang memiliki PDRB jauh lebih tinggi."
                )
                fig_pdrb = px.bar(
                    df_cluster_grouped,
                    x='Cluster_Best',
                    y='Kekayaan Rata-Rata (PDRB)',
                    color='Cluster_Best',
                    labels={'Cluster_Best': 'Cluster', 'Kekayaan Rata-Rata (PDRB)': 'Rata-rata PDRB (Ribu Rp)'}
                )
                fig_pdrb.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins", showlegend=False)
                fig_pdrb.update_traces(marker_cornerradius=5)
                st.plotly_chart(fig_pdrb, use_container_width=True)

            with col_c3:
                st.subheader(
                    "Rata-rata Pinjaman Beredar",
                    help="Grafik ini membuktikan julukan 'Fintech Menengah' untuk Cluster 0, yang volume pinjamannya jauh lebih rendah."
                )
                fig_loan = px.bar(
                    df_cluster_grouped,
                    x='Cluster_Best',
                    y='Total Pinjaman Beredar (Miliar)',
                    color='Cluster_Best',
                    labels={'Cluster_Best': 'Cluster', 'Total Pinjaman Beredar (Miliar)': 'Rata-rata Pinjaman (Miliar Rp)'}
                )
                fig_loan.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins", showlegend=False)
                fig_loan.update_traces(marker_cornerradius=5)
                st.plotly_chart(fig_loan, use_container_width=True)
        
        except KeyError as e:
            st.error(f"Gagal membuat grafik cluster. Kolom tidak ditemukan: {e}. Periksa `df_master_cluster.csv`.")
        except Exception as e:
            st.error(f"Terjadi error saat membuat grafik cluster: {e}")

    # --- (REVISI) Judul diubah menjadi Temuan #3 ---
    st.markdown("---")
    
    # --- Baris 1 ---
    col1, col2 = st.columns(2)
    
    with col1:
        # --- GRAFIK 1: Literasi vs Impulsif (BAR CHART 3 GRUP) ---
        st.subheader(
            "Literasi adalah 'Rem' Impulsif",
            help=(
                "Grup dengan Literasi Finansial tinggi memiliki Skor Impulsif paling rendah."
            )
        )
        
        df_literasi_grouped = df_survey.groupby('Grup Melek Finansial')['Skor Impulsif'].mean(numeric_only=True).reset_index()
        literasi_order = ["Rendah", "Sedang", "Tinggi"]
        
        avg_impulsif = df_survey['Skor Impulsif'].mean()
        
        fig1 = px.bar(
            df_literasi_grouped, 
            x='Grup Melek Finansial', 
            y='Skor Impulsif',
            color='Grup Melek Finansial',
            category_orders={"Grup Melek Finansial": literasi_order}, 
            labels={
                'Skor Impulsif': 'Rata-rata Skor Impulsif',
                'Grup Melek Finansial': 'Grup Melek Finansial'
            }
        )
        fig1.add_hline(
            y=avg_impulsif, 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Rata-rata"
        )
        fig1.update_traces(marker_cornerradius=5)
        fig1.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_family="Poppins"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # --- GRAFIK 2: Gender vs Impulsif (BAR CHART) ---
        st.subheader(
            "Laki-laki Lebih Impulsif",
            help=(
                "Secara rata-rata, responden Laki-laki (Male) memiliki Skor Impulsif sedikit lebih tinggi daripada Perempuan (Female)."
            )
        )
        
        df_gender_impulse = df_survey.groupby('Gender')['Skor Impulsif'].mean(numeric_only=True).reset_index()
        avg_impulsif_gender = df_survey['Skor Impulsif'].mean()

        fig2 = px.bar(
            df_gender_impulse, 
            x='Gender', 
            y='Skor Impulsif',
            color='Gender',
            labels={'Skor Impulsif': 'Rata-rata Skor Impulsif'}
        )
        fig2.add_hline(
            y=avg_impulsif_gender, 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Rata-rata"
        )
        fig2.update_traces(marker_cornerradius=5)
        fig2.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Poppins"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- (BARU) Baris 2 ---
    col3, col4 = st.columns(2)

    with col3:
        # --- (PINDAH) GRAFIK 3: PDRB vs Adopsi Fintech ---
        st.subheader(
            "Daerah PDRB Tinggi Lebih Mengadopsi Fintech",
            help=(
                "Gen Z yang tinggal di daerah PDRB Tinggi memiliki rata-rata Skor Adopsi Fintech yang paling tinggi."
            )
        )
        
        df_pdrb_grouped = df_survey.groupby('Grup Kekayaan (PDRB)')['Skor Adopsi Fintech'].mean(numeric_only=True).reset_index()
        pdrb_order = ["Rendah", "Menengah", "Tinggi"]

        avg_adopsi = df_survey['Skor Adopsi Fintech'].mean()
        
        fig3 = px.bar(
            df_pdrb_grouped,
            x='Grup Kekayaan (PDRB)',
            y='Skor Adopsi Fintech',
            color='Grup Kekayaan (PDRB)',
            category_orders={"Grup Kekayaan (PDRB)": pdrb_order},
            labels={
                'Kekayaan Rata-Rata (PDRB)': 'Grup Kekayaan Daerah (PDRB)',
                'Skor Adopsi Fintech': 'Rata-rata Skor Adopsi'
            }
        )
        fig3.add_hline(
            y=avg_adopsi, 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Rata-rata"
        )
        fig3.update_traces(marker_cornerradius=10)
        fig3.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Poppins"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # --- (PINDAH) GRAFIK 4: Pendapatan vs Kecemasan ---
        st.subheader(
            "Pendapatan Tidak Mempengaruhi Kecemasan",
            help=(
                "Baik dari tingkat pendapatan tinggi, sedang, dan rendah, semuanya memiliki Skor Kecemasan yang sama."
            )
        )
        
        df_pendapatan_grouped = df_profile.groupby('Grup Pendapatan')['Skor Kecemasan'].mean(numeric_only=True).reset_index()
        avg_kecemasan = df_profile['Skor Kecemasan'].mean()
        pendapatan_order = ["Rendah", "Sedang", "Tinggi"]
        
        fig4 = px.bar(
            df_pendapatan_grouped,
            x='Grup Pendapatan',
            y='Skor Kecemasan',
            color='Grup Pendapatan',
            category_orders={"Grup Pendapatan": pendapatan_order},
            labels={
                'Grup Pendapatan': 'Grup Pendapatan',
                'Skor Kecemasan': 'Rata-rata Skor Kecemasan'
            }
        )
        fig4.add_hline(
            y=avg_kecemasan, 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Rata-rata"
        )
        fig4.update_traces(marker_cornerradius=10)
        fig4.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Poppins"
        )
        st.plotly_chart(fig4, use_container_width=True)
        
    
    st.markdown("---")


# ===================================================================
# 📌 HALAMAN 2 – BEDAH PROVINSI (LOKAL)
# ===================================================================
elif page == "Halaman 2: Dashboard Regional":
    
    st.title("Dashboard Regional")
    st.markdown(
        "<div class='centered-subtitle'>Lihat lebih dalam data psikologis, finansial, dan risiko di tiap provinsi.</div>", 
        unsafe_allow_html=True
    )
    
    # --- (TETAP) Filter Dropdown Utama ---
    st.markdown("---")
    
    list_provinsi = sorted(regional['Provinsi'].unique().tolist())
    
    selected_province = st.selectbox(
        "Pilih Provinsi untuk Dibedah:",
        options=list_provinsi,
        index=list_provinsi.index("jawa barat") if "jawa barat" in list_provinsi else 0,
        label_visibility="collapsed" # Sembunyikan label, judul sudah jelas
    )
    st.markdown("---")

    # --- (TETAP) Filtering Data Dinamis ---
    df_profile_filtered = df_profile[df_profile['province'] == selected_province]
    df_survey_filtered = df_survey[df_survey['Province of Origin'] == selected_province]
    regional_filtered = regional[regional['Provinsi'] == selected_province]

    # Cek jika ada data
    if df_profile_filtered.empty and df_survey_filtered.empty:
        st.warning(f"Tidak ditemukan data responden individu (profil & survei) untuk {selected_province.title()}.")
        st.stop()
        
    # --- (REVISI) MENAMBAHKAN "gap='large'" UNTUK JARAK ANTAR KOLOM ---
    col1, col2, col3 = st.columns(3, gap="large")

    # --- KOLOM 1: PROFIL PSIKOLOGIS (DARI SURVEY) ---
    with col1:
        st.header(f"Profil Psikologis")
        st.caption(f"Berdasarkan {len(df_survey_filtered)} responden survei di {selected_province.title()}")
        
        if not df_survey_filtered.empty:
            # Grafik 1: Distribusi Skor Melek Finansial
            st.subheader(
                "Skor Melek Finansial",
                help="Distribusi skor pemahaman finansial (1-5) di provinsi ini."
            )
            # <-- REVISI: marginal="box" DIHAPUS
            fig1_h2 = px.histogram(df_survey_filtered, x='Skor Melek Finansial') 
            fig1_h2.update_traces(marker_cornerradius=5) # Ini sekarang aman
            
            # <-- REVISI: bargap ditambahkan untuk jarak antar batang
            fig1_h2.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins", bargap=0.3)
            st.plotly_chart(fig1_h2, use_container_width=True)

            # --- (REVISI) MENGGANTI BOXPLOT MENJADI BARPLOT RATA-RATA ---
            st.subheader(
                "Gender vs Rasa Aman (Rata-rata)", 
                help="Perbandingan skor rata-rata rasa aman finansial (1-5) antara gender."
            )
            df_gender_safety = df_survey_filtered.groupby('Gender')['Skor_Rasa_Aman'].mean(numeric_only=True).reset_index()
            
            fig3_h2 = px.bar( 
                df_gender_safety, 
                x='Gender', 
                y='Skor_Rasa_Aman', 
                color='Gender',
                labels={'Skor_Rasa_Aman': 'Rata-rata Skor Rasa Aman'}
            )
            fig3_h2.update_traces(marker_cornerradius=5) 
            fig3_h2.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
            st.plotly_chart(fig3_h2, use_container_width=True)

        else:
            st.info(f"Tidak ada data responden survei untuk {selected_province.title()}.")

    # --- KOLOM 2: PROFIL FINANSIAL (DARI PROFIL) ---
    with col2:
        st.header(f"Profil Finansial")
        st.caption(f"Berdasarkan {len(df_profile_filtered)} responden profil di {selected_province.title()}")
        
        if not df_profile_filtered.empty:
            # Grafik 2: Distribusi Skor Kecemasan
            st.subheader(
                "Skor Kecemasan",
                help="Distribusi skor kecemasan finansial (1-5) di provinsi ini."
            )
            # <-- REVISI: marginal="box" DIHAPUS (INI MEMPERBAIKI ERROR VALUEERROR)
            fig2_h2 = px.histogram(df_profile_filtered, x='Skor Kecemasan', color_discrete_sequence=['#E74C3C'])
            fig2_h2.update_traces(marker_cornerradius=5) # Ini sekarang aman
            
            # <-- REVISI: bargap ditambahkan untuk jarak antar batang
            fig2_h2.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins", bargap=0.3)
            st.plotly_chart(fig2_h2, use_container_width=True)

            # Grafik 5 & 6: Donut Charts
            st.subheader(
                "Penggunaan Fintech & Pinjaman",
                help="Aplikasi favorit dan tujuan utama responden jika mengambil pinjaman."
            )
            fig5_h2 = px.pie(df_profile_filtered, names='main_fintech_app', hole=0.4, title='E-Wallet Favorit')
            fig5_h2.update_traces(textposition='inside', textinfo='percent+label')
            fig5_h2.update_layout(height=250, margin={"r":0,"t":40,"l":0,"b":0}, font_family="Poppins", showlegend=False)
            st.plotly_chart(fig5_h2, use_container_width=True)

            fig6_h2 = px.pie(df_profile_filtered, names='loan_usage_purpose', hole=0.4, title='Tujuan Pinjaman')
            fig6_h2.update_traces(textposition='inside', textinfo='percent+label')
            fig6_h2.update_layout(height=250, margin={"r":0,"t":40,"l":0,"b":0}, font_family="Poppins", showlegend=False)
            st.plotly_chart(fig6_h2, use_container_width=True)

        else:
            st.info(f"Tidak ada data responden profil untuk {selected_province.title()}.")

    # --- KOLOM 3: RISIKO & PERILAKU GENDER ---
    with col3:
        st.header(f"Risiko & Perilaku")
        st.caption("Data risiko regional & perbandingan gender.")

        # --- (REVISI) Grafik 8: KPI Risiko Regional dengan Penjelasan ---
        st.subheader(
            "Metrik Risiko Regional",
            help="Data pinjaman (P2P) regional dari OJK/BPS."
        )
        if not regional_filtered.empty:
            st.metric(
                label="**Kredit Macet (TWP 90%)**", 
                value=f"{regional_filtered['TWP 90%'].values[0]:.2f} %",
                help="Tingkat Wanprestasi 90 hari. Persentase pinjaman P2P yang gagal bayar lebih dari 90 hari. Semakin tinggi, semakin berisiko."
            )
            st.metric(
                label="**Total Pinjaman Beredar**", 
                value=f"Rp {regional_filtered['Outstanding Pinjaman (Rp miliar)'].values[0]:,.0f} Miliar",
                help="Total uang yang masih dipinjam (belum lunas) oleh semua peminjam P2P di provinsi ini."
            )
            st.metric(
                label="**Jumlah Peminjam Aktif**", 
                value=f"{regional_filtered['Jumlah Penerima Pinjaman (akun)'].values[0]:,.0f} Akun",
                help="Jumlah akun peminjam unik yang saat ini memiliki pinjaman yang belum lunas."
            )
        else:
            st.info(f"Tidak ada data risiko regional untuk {selected_province.title()}.")
        
        # Grafik 7: Rata-rata Pinjaman per Gender
        if not df_profile_filtered.empty:
            st.subheader(
                "Rata-rata Pinjaman (Gender)",
                help="Rata-rata outstanding loan per gender di provinsi ini."
            )
            df_gender_loan = df_profile_filtered.groupby('gender')['outstanding_loan'].mean(numeric_only=True).reset_index()
            fig7_h2 = px.bar(
                df_gender_loan, 
                x='gender', 
                y='outstanding_loan', 
                color='gender', 
                labels={'outstanding_loan': 'Rata-rata Pinjaman', 'gender': 'Gender'}
            )
            fig7_h2.update_traces(marker_cornerradius=5)
            fig7_h2.update_layout(height=250, plot_bgcolor='rgba(0,0,0,0)', font_family="Poppins")
            st.plotly_chart(fig7_h2, use_container_width=True)
        else:
            st.info(f"Tidak ada data responden profil untuk {selected_province.title()}.")