import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Pengaturan Tampilan
st.set_page_config(page_title="Analisis Clustering Cafe", layout="wide")

st.markdown("<h1 style='text-align: center;'>📊 Sistem Clustering Penjualan Cafe</h1>", unsafe_allow_html=True)
st.markdown("---")

# 2. Sidebar untuk Upload
st.sidebar.header("Konfigurasi")
uploaded_file = st.sidebar.file_uploader("Upload File CSV Rekap Penjualan", type=["csv"])

if uploaded_file is not None:
    try:
        # Membaca file tanpa skip baris (agar judul kolom terbaca)
        df = pd.read_csv(uploaded_file)
        
        # --- PROSES PEMBERSIHAN DATA (Agar Tak Error) ---
        # Hapus baris yang benar-benar kosong
        df = df.dropna(how='all')
        
        # Bersihkan kolom JUMLAH dan HARGA dari simbol Rp, titik, atau spasi
        for col in ['JUMLAH', 'HARGA']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('Rp', '', regex=False)
                    df[col] = df[col].str.replace('.', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.strip()
                
                # Ubah jadi angka, jika gagal ubah jadi NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Hapus baris yang kolom angka-nya rusak (NaN)
        df = df.dropna(subset=['JUMLAH', 'HARGA'])
        
        if not df.empty:
            # Tampilkan data awal
            st.subheader("✅ Data Berhasil Dimuat")
            st.write(f"Total data yang valid: {len(df)} baris")
            st.dataframe(df.head(10), use_container_width=True)

            # --- PROSES K-MEANS ---
            X = df[['JUMLAH', 'HARGA']]
            
            # Standarisasi data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Menjalankan K-Means dengan 3 Cluster
            kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
            df['Cluster'] = kmeans.fit_predict(X_scaled)

            # --- VISUALISASI ---
            st.markdown("---")
            st.subheader("📈 Hasil Clustering Penjualan")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df, x='JUMLAH', y='HARGA', 
                hue='Cluster', palette='bright', 
                s=150, style='Cluster', ax=ax
            )
            
            plt.title("Pengelompokan Produk Berdasarkan Jumlah & Harga")
            plt.xlabel("Jumlah Terjual")
            plt.ylabel("Harga Produk")
            st.pyplot(fig)

            # Tampilkan Tabel Hasil
            st.subheader("📄 Detail Hasil Cluster")
            st.dataframe(df, use_container_width=True)
            
            st.success("Analisis selesai! Data telah dikelompokkan ke dalam 3 cluster.")
            
        else:
            st.error("Error: Data kosong setelah dibersihkan. Pastikan kolom JUMLAH dan HARGA berisi angka.")

    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
        st.info("Pastikan file CSV memiliki kolom: PRODUK, JUMLAH, HARGA")
else:
    st.info("👋 Silakan upload file CSV melalui menu di sebelah kiri untuk memulai.")