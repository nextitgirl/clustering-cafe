import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION (Judul Sesuai Jurnal) ---
st.set_page_config(page_title="Aplikasi Clustering K-Means", layout="wide")

st.markdown(
    """
    <div style='text-align: center; padding: 25px; background-color: #f8f9fa; border-radius: 15px; border: 2px solid #4CAF50;'>
        <h1 style='color: #1b5e20; font-size: 2.1em;'>Rancang Bangun Aplikasi Clustering Penjualan Makanan dan Minuman Menggunakan Algoritma K-Means</h1>
        <hr style='border: 1px solid #4CAF50; width: 80%;'>
        <p style='font-size: 1.1em; color: #555;'>Solusi cerdas untuk analisis strategi penjualan UMKM.</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# --- 2. LOGIKA UNGGAH DATA ---
st.sidebar.header("⚙️ Menu Utama")
uploaded_file = st.sidebar.file_uploader("Unggah File Rekap Penjualan (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # STEP 1: Deteksi Pemisah (Koma atau Titik Koma) secara otomatis
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        sep = ';' if content.count(';') > content.count(',') else ','
        uploaded_file.seek(0)
        
        # STEP 2: Baca semua data sebagai string dulu untuk dibersihkan
        df_raw = pd.read_csv(uploaded_file, sep=sep, engine='python').astype(str)
        
        # STEP 3: Bersihkan nama kolom (Hapus spasi, jadikan huruf besar semua)
        # Ini agar "Produk", "produk ", dan "PRODUK" dibaca sama oleh sistem
        df_raw.columns = df_raw.columns.str.strip().str.upper()
        
        # STEP 4: Cari kolom menggunakan kata kunci (Fuzzy Matching)
        col_p = next((c for c in df_raw.columns if any(k in c for k in ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'MENU'])), None)
        col_j = next((c for c in df_raw.columns if any(k in c for k in ['JUMLAH', 'QTY', 'QUANTITY', 'TERJUAL', 'UNIT', 'JML'])), None)
        col_h = next((c for c in df_raw.columns if any(k in c for k in ['HARGA', 'PRICE', 'NILAI', 'TOTAL', 'HARG'])), None)

        if col_p and col_j and col_h:
            # Ambil datanya
            df = df_raw[[col_p, col_j, col_h]].copy()
            df.columns = ['PRODUK', 'JUMLAH', 'HARGA']
            
            # STEP 5: Bersihkan data angka (Hapus Rp, titik ribuan, spasi, dsb)
            for col in ['JUMLAH', 'HARGA']:
                # Menghapus apapun yang bukan angka (0-9)
                df[col] = df[col].str.replace(r'[^\d]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Buang baris yang kosong setelah dibersihkan
            df = df.dropna()

            if not df.empty:
                st.sidebar.markdown("---")
                n_clusters = st.sidebar.slider("Tentukan Jumlah Cluster (K)", 2, 5, 3)

                # --- 3. PROSES K-MEANS ---
                X = df[['JUMLAH', 'HARGA']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
                df['Cluster'] = kmeans.labels_

                # --- 4. OUTPUT HASIL ---
                st.success(f"✅ Sistem berhasil mengenali kolom: **{col_p}**, **{col_j}**, dan **{col_h}**.")
                
                tab1, tab2 = st.tabs(["📈 Visualisasi Cluster", "📄 Tabel Data"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df, x='JUMLAH', y='HARGA', hue='Cluster', palette='viridis', s=200, ax=ax, style='Cluster')
                    plt.title("Hasil Pengelompokan Penjualan")
                    st.pyplot(fig)
                
                with tab2:
                    st.dataframe(df, use_container_width=True)

                # --- 5. KESIMPULAN ---
                st.markdown("---")
                st.subheader("💡 Kesimpulan Hasil Clustering")
                
                # Cari cluster terlaris berdasarkan rata-rata jumlah
                avg_sales = df.groupby('Cluster')['JUMLAH'].mean().sort_values(ascending=False)
                id_laris = avg_sales.index[0]
                id_kurang = avg_sales.index[-1]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.success("### 🌟 Cluster Terlaris")
                    st.write(df[df['Cluster'] == id_laris]['PRODUK'].unique()[:10])
                with c2:
                    st.warning("### 📉 Cluster Perlu Promo")
                    st.write(df[df['Cluster'] == id_kurang]['PRODUK'].unique()[:10])
            else:
                st.error("Data ditemukan, tapi kolom jumlah/harga tidak berisi angka yang benar.")
        else:
            st.error("Sistem gagal menemukan kolom Produk, Jumlah, atau Harga.")
            st.info(f"Kolom yang terdeteksi di file Anda: {list(df_raw.columns)}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan unggah file CSV data penjualan Anda untuk memulai.")