import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Analisis Clustering Cafe", layout="wide")
st.markdown("<h1 style='text-align: center;'>📊 Sistem Clustering Penjualan Cafe</h1>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload File CSV Rekap Penjualan", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. Baca file (otomatis deteksi pemisah koma atau titik koma)
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # 2. PROSES OTOMATIS: Ubah semua nama kolom jadi HURUF BESAR agar seragam
        # Ini fungsinya supaya mau input 'produk' atau 'PRODUK' tetap terbaca
        df.columns = df.columns.str.strip().str.upper()
        
        # 3. Cek apakah kolom yang dibutuhkan ada (setelah diseragamkan ke huruf besar)
        if 'JUMLAH' in df.columns and 'HARGA' in df.columns:
            
            # 4. Bersihkan data angka (hapus Rp, titik ribuan, dll)
            for col in ['JUMLAH', 'HARGA']:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('Rp', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Hapus baris yang datanya kosong atau rusak
            df = df.dropna(subset=['JUMLAH', 'HARGA'])

            if not df.empty:
                st.subheader("✅ Data Berhasil Dimuat")
                st.dataframe(df.head(), use_container_width=True)

                # 5. Proses K-Means
                X = df[['JUMLAH', 'HARGA']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=3, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X_scaled)

                # 6. Grafik
                st.subheader("📈 Hasil Clustering")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='JUMLAH', y='HARGA', hue='Cluster', palette='bright', s=150, ax=ax)
                st.pyplot(fig)
                st.success("Analisis Berhasil! Grafik sudah muncul di atas.")
            else:
                st.error("Data tidak mengandung angka yang valid untuk dihitung.")
        else:
            # Jika kolom tidak ditemukan, beri tahu user kolom apa saja yang terdeteksi
            st.error(f"Kolom 'JUMLAH' atau 'HARGA' tidak ditemukan.")
            st.warning(f"Kolom yang terdeteksi di file kamu: {list(df.columns)}")
            st.info("Saran: Pastikan judul kolom di Excel kamu adalah PRODUK, JUMLAH, dan HARGA.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")