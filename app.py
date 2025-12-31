import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. SINKRONISASI JUDUL (Sesuai Judul Jurnal) ---
st.set_page_config(page_title="Clustering Penjualan K-Means", layout="wide")

st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 15px; border: 2px solid #4CAF50;'>
        <h1 style='color: #1b5e20; font-size: 2.1em;'>Rancang Bangun Aplikasi Clustering Penjualan Makanan dan Minuman Menggunakan Algoritma K-Means</h1>
        <hr style='border: 1px solid #4CAF50;'>
        <p style='font-size: 1.3em; color: #333; font-weight: bold;'>Studi Kasus: Cafe Bumbum</p>
        <p style='font-size: 1em; color: #666;'>Implementasi Data Mining untuk Optimalisasi Strategi Penjualan</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# --- 2. SIDEBAR CONFIGURATION ---
st.sidebar.header("Input Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah File Rekap Penjualan (Format CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Deteksi separator otomatis
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        content = raw_bytes.decode('utf-8', errors='ignore')
        sep = ';' if content.count(';') > content.count(',') else ','
        
        # Deep Scan untuk mencari baris header
        raw_df = pd.read_csv(uploaded_file, sep=sep, header=None, engine='python').astype(str)
        target_row = None
        keywords = ['PRODUK', 'ITEM', 'BARANG', 'JUMLAH', 'QTY', 'HARGA', 'PRICE']
        
        for i, row in raw_df.iterrows():
            combined_text = "".join(row.values).replace(" ", "").upper()
            if any(key in combined_text for key in keywords):
                target_row = i
                break
        
        if target_row is not None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep, skiprows=target_row, engine='python')
            
            # Normalisasi nama kolom (Menghapus karakter non-alfanumerik)
            df.columns = df.columns.str.strip().str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
            
            # Mapping Kolom Otomatis
            col_map = {}
            mapping_logic = {
                'PRODUK': ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'MENU'],
                'JUMLAH': ['JUMLAH', 'QTY', 'QUANTITY', 'TERJUAL', 'UNIT'],
                'HARGA': ['HARGA', 'PRICE', 'NILAI', 'TOTAL']
            }
            
            for key, syns in mapping_logic.items():
                for c in df.columns:
                    if any(s in c for s in syns):
                        col_map[key] = c
                        break

            if len(col_map) >= 3:
                df_final = df[[col_map['PRODUK'], col_map['JUMLAH'], col_map['HARGA']]].copy()
                df_final.columns = ['PRODUK', 'JUMLAH', 'HARGA']
                
                # Pembersihan data numerik (Menghapus Rp, titik, dsb)
                for col in ['JUMLAH', 'HARGA']:
                    df_final[col] = df_final[col].astype(str).str.replace(r'[^\d]', '', regex=True)
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                
                df_final = df_final.dropna(subset=['JUMLAH', 'HARGA'])

                if not df_final.empty:
                    # --- 3. PROSES ALGORITMA K-MEANS ---
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Parameter Algoritma")
                    n_clusters = st.sidebar.slider("Jumlah Cluster (K)", 2, 5, 3)

                    X = df_final[['JUMLAH', 'HARGA']]
                    # Transformasi data (Standardisasi)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Inisialisasi dan Fit Model K-Means
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
                    df_final['Cluster'] = kmeans.labels_

                    # --- 4. VISUALISASI HASIL CLUSTERING ---
                    st.markdown("---")
                    st.subheader("Visualisasi Hasil Clustering Penjualan")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df_final, x='JUMLAH', y='HARGA', hue='Cluster', palette='viridis', s=200, ax=ax, style='Cluster')
                    plt.title(f"Persebaran Data Produk Berdasarkan Cluster (K={n_clusters})")
                    plt.xlabel("Jumlah Terjual (Quantity)")
                    plt.ylabel("Harga (Price)")
                    st.pyplot(fig)
                    
                    st.subheader("Data Hasil Analisis Clustering")
                    st.dataframe(df_final, use_container_width=True)

                    # --- 5. KESIMPULAN DAN HASIL ANALISIS ---
                    st.markdown("---")
                    st.subheader("Interpretasi Hasil Analisis (Kesimpulan)")
                    
                    # Logika penentuan cluster terlaris vs kurang laku
                    cluster_summary = df_final.groupby('Cluster')['JUMLAH'].mean().sort_values(ascending=False)
                    id_laris = cluster_summary.index[0]
                    id_kurang = cluster_summary.index[-1]
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.success(f"### 🌟 Cluster Produk Terlaris")
                        st.write("Daftar produk dengan tingkat volume penjualan tertinggi:")
                        st.write(df_final[df_final['Cluster'] == id_laris]['PRODUK'].unique()[:15])
                    with c2:
                        st.warning(f"### 📉 Cluster Produk Kurang Laku")
                        st.write("Daftar produk dengan tingkat volume penjualan terendah:")
                        st.write(df_final[df_final['Cluster'] == id_kurang]['PRODUK'].unique()[:15])
                    
                    st.info("**Informasi Metodologi:** Analisis ini dilakukan secara otomatis menggunakan perhitungan jarak Euclidean dalam algoritma K-Means.")
                else:
                    st.error("Data numerik tidak ditemukan setelah proses pembersihan.")
            else:
                st.error("Struktur kolom (Produk, Jumlah, Harga) tidak terdeteksi.")
        else:
            st.error("Header dataset tidak ditemukan. Pastikan file CSV memiliki baris judul kolom.")
            
    except Exception as e:
        st.error(f"Kesalahan Sistem: {e}")
else:
    st.info("👋 Selamat Datang. Silakan unggah dataset penjualan (CSV) untuk memulai proses clustering.")