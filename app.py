import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman
st.set_page_config(page_title="Data Mining Cafe Bumbum", layout="wide")

st.title("📊 Sistem Clustering Penjualan Cafe")
st.markdown("---")

# 1. Fitur Upload File
uploaded_file = st.sidebar.file_uploader("Upload File Rekap Penjualan (CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca data dengan melewati 2 baris judul seperti struktur filemu
    df = pd.read_csv(uploaded_file, skiprows=2)
    
    # Menampilkan 2 kolom di web
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📁 Data Terdeteksi")
        st.write(df.head(10)) # Menampilkan 10 data teratas
        st.info(f"Total Produk: {len(df)}")

    # 2. Proses Data Mining
    # Kita hanya ambil kolom angka: JUMLAH (Quantity) dan HARGA (Omzet)
    X = df[['JUMLAH', 'HARGA']]

    # Normalisasi agar data seimbang (Skala Jumlah vs Rupiah)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar untuk pengaturan Algoritma
    st.sidebar.header("Pengaturan K-Means")
    k = st.sidebar.slider("Tentukan Jumlah Cluster (K)", 2, 5, 3)
    
    # 3. Eksekusi K-Means
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    with col2:
        st.subheader("📈 Visualisasi Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Membuat scatter plot yang lebih cantik
        scatter = sns.scatterplot(
            data=df, x='JUMLAH', y='HARGA', 
            hue='Cluster', palette='viridis', s=100, ax=ax
        )
        ax.set_title(f"Pengelompokan {k} Cluster Produk")
        ax.set_xlabel("Jumlah Terjual (Quantity)")
        ax.set_ylabel("Total Pendapatan (Rupiah)")
        st.pyplot(fig)

    st.markdown("---")
    
    # 4. Tabel Hasil & Kesimpulan
    st.subheader("📋 Hasil Pengelompokan Produk")
    
    # Menampilkan tabel yang bisa diurutkan
    st.dataframe(df[['PRODUK', 'JUMLAH', 'HARGA', 'Cluster']].sort_values(by='Cluster'), use_container_width=True)

    # 5. Penjelasan Sederhana untuk Dosen
    st.success("💡 **Cara Membaca Hasil:**")
    st.write("- **Cluster dengan HARGA & JUMLAH tinggi:** Produk Unggulan (Laris & Menguntungkan).")
    st.write("- **Cluster dengan JUMLAH tinggi tapi HARGA rendah:** Produk Pendukung (Murah tapi Laris).")
    st.write("- **Cluster dengan HARGA & JUMLAH rendah:** Produk Perlu Evaluasi (Kurang Laku).")

else:
    st.warning("Silakan upload file CSV melalui sidebar di sebelah kiri untuk memulai analisis.")
    st.info("Catatan: Pastikan file CSV memiliki kolom PRODUK, JUMLAH, dan HARGA.")