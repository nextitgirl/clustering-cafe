import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PENGATURAN TAMPILAN ---
st.set_page_config(page_title="Clustering Penjualan", layout="wide")

st.markdown(
    """
    <div style='text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 15px; border: 2px solid #4CAF50;'>
        <h1 style='color: #1b5e20; font-size: 2.1em;'>Rancang Bangun Aplikasi Clustering Penjualan Makanan dan Minuman Menggunakan Algoritma K-Means</h1>
        <hr style='border: 1px solid #4CAF50; width: 80%;'>
        <p style='font-size: 1.1em; color: #555;'>Aplikasi ini otomatis menyesuaikan dengan format file CSV Anda.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📁 Menu Data")
uploaded_file = st.sidebar.file_uploader("Upload Rekap Penjualan (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. BACA FILE SEBAGAI TEKS UNTUK MENCARI POSISI HEADER
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore').splitlines()
        
        # Cari baris mana yang mengandung kata kunci (Produk/Jumlah/Harga)
        target_row = 0
        keywords = ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'JUMLAH', 'QTY', 'HARGA', 'PRICE']
        
        for i, line in enumerate(content):
            if any(key in line.upper() for key in keywords):
                target_row = i
                break
        
        # 2. DETEKSI PEMISAH (SEMI-KOLON ATAU KOMA)
        uploaded_file.seek(0)
        sample = "\n".join(content[target_row:target_row+2])
        separator = ';' if sample.count(';') > sample.count(',') else ','
        
        # 3. BACA DATA DENGAN SKIPROWS OTOMATIS
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, sep=separator, skiprows=target_row, engine='python')
        
        # Bersihkan nama kolom agar seragam
        df_raw.columns = df_raw.columns.astype(str).str.strip().str.upper()
        
        # 4. MAPPING KOLOM SECARA CERDAS
        col_p = next((c for c in df_raw.columns if any(k in c for k in ['PRODUK', 'ITEM', 'BARANG', 'NAMA'])), None)
        col_j = next((c for c in df_raw.columns if any(k in c for k in ['JUMLAH', 'QTY', 'QUANTITY', 'TERJUAL'])), None)
        col_h = next((c for c in df_raw.columns if any(k in c for k in ['HARGA', 'PRICE', 'TOTAL', 'NILAI'])), None)

        if col_p and col_j and col_h:
            df = df_raw[[col_p, col_j, col_h]].copy()
            df.columns = ['PRODUK', 'JUMLAH', 'HARGA']
            
            # Bersihkan data angka (Hapus Rp, titik, atau koma yang salah)
            for col in ['JUMLAH', 'HARGA']:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()

            if not df.empty:
                # --- PROSES K-MEANS ---
                st.sidebar.markdown("---")
                n_clusters = st.sidebar.slider("Tentukan Jumlah Cluster (K)", 2, 5, 3)
                
                X = df[['JUMLAH', 'HARGA']]
                X_scaled = StandardScaler().fit_transform(X)
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
                df['Cluster'] = kmeans.labels_

                # --- VISUALISASI ---
                st.success(f"✅ Berhasil memproses data dari baris ke-{target_row + 1}")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(data=df, x='JUMLAH', y='HARGA', hue='Cluster', palette='viridis', s=200, ax=ax, style='Cluster')
                plt.title("Hasil Clustering Penjualan")
                st.pyplot(fig)
                
                st.dataframe(df, use_container_width=True)
                
                # --- KESIMPULAN ---
                avg_sales = df.groupby('Cluster')['JUMLAH'].mean().sort_values(ascending=False)
                laris = df[df['Cluster'] == avg_sales.index[0]]['PRODUK'].unique()
                kurang = df[df['Cluster'] == avg_sales.index[-1]]['PRODUK'].unique()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"### 🌟 Cluster Terlaris\n" + "\n".join([f"- {p}" for p in laris[:10]]))
                with c2:
                    st.warning(f"### 📉 Cluster Kurang Laku\n" + "\n".join([f"- {p}" for p in kurang[:10]]))
            else:
                st.error("Isi kolom bukan angka yang valid.")
        else:
            st.error("Kolom Produk, Jumlah, atau Harga tidak ditemukan.")
            
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Silakan unggah file CSV Anda.")