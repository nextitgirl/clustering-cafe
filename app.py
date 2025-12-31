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
        <p style='font-size: 1.1em; color: #555;'>Silakan masukkan data penjualan Anda. Pastikan data yang Anda unggah memiliki kolom Produk, Jumlah, dan Harga.</p>
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
        # Deteksi separator secara otomatis (Koma atau Titik Koma)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        content = raw_bytes.decode('utf-8', errors='ignore')
        sep = ';' if content.count(';') > content.count(',') else ','
        
        # Scan data untuk mencari baris header yang mengandung keyword
        raw_df = pd.read_csv(uploaded_file, sep=sep, header=None, engine='python').astype(str)
        target_row = None
        keywords = ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'JUMLAH', 'QTY', 'QUANTITY', 'HARGA', 'PRICE']
        
        for i, row in raw_df.iterrows():
            row_text = " ".join(row.values).upper()
            if any(key in row_text for key in keywords):
                target_row = i
                break
        
        # Baca ulang file mulai dari baris header yang ditemukan
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep, skiprows=target_row if target_row is not None else 0, engine='python')
        
        # Bersihkan nama kolom (Huruf besar, hapus spasi, hapus simbol)
        df.columns = df.columns.str.strip().str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
        
        # --- LOGIKA PEMETAAN KOLOM CERDAS ---
        col_map = {}
        logics = {
            'PRODUK': ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'MENU', 'DESKRIPSI'],
            'JUMLAH': ['JUMLAH', 'QTY', 'QUANTITY', 'TERJUAL', 'UNIT', 'VOL', 'JML'],
            'HARGA': ['HARGA', 'PRICE', 'NILAI', 'TOTAL', 'HARG', 'PRC']
        }
        
        for key, synonyms in logics.items():
            for c in df.columns:
                if any(syn in c for syn in synonyms):
                    col_map[key] = c
                    break

        # Cek apakah 3 kolom inti ditemukan
        if len(col_map) >= 3:
            df_final = df[[col_map['PRODUK'], col_map['JUMLAH'], col_map['HARGA']]].copy()
            df_final.columns = ['PRODUK', 'JUMLAH', 'HARGA']
            
            # Bersihkan angka dari simbol (Rp, titik ribuan, koma desimal)
            for col in ['JUMLAH', 'HARGA']:
                # Hapus semua kecuali angka dan titik
                df_final[col] = df_final[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
            
            df_final = df_final.dropna(subset=['JUMLAH', 'HARGA'])

            if not df_final.empty:
                # --- 3. PROSES K-MEANS ---
                st.sidebar.markdown("---")
                n_clusters = st.sidebar.slider("Tentukan Jumlah Cluster (K)", 2, 5, 3)

                X = df_final[['JUMLAH', 'HARGA']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
                df_final['Cluster'] = kmeans.labels_

                # --- 4. OUTPUT HASIL ---
                st.success(f"✅ Analisis Berhasil: Memproses {len(df_final)} baris data.")
                
                tab1, tab2 = st.tabs(["📈 Visualisasi Grafik", "📄 Tabel Data"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df_final, x='JUMLAH', y='HARGA', hue='Cluster', palette='viridis', s=200, ax=ax, style='Cluster')
                    plt.title(f"Visualisasi Clustering (K-Means)")
                    st.pyplot(fig)
                
                with tab2:
                    st.dataframe(df_final, use_container_width=True)

                # --- 5. KESIMPULAN STRATEGIS ---
                st.markdown("---")
                st.subheader("💡 Hasil Analisis Berdasarkan Kelompok (Cluster)")
                
                cluster_sum = df_final.groupby('Cluster')['JUMLAH'].mean().sort_values(ascending=False)
                id_laris = cluster_sum.index[0]
                id_kurang = cluster_sum.index[-1]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.success(f"### 🌟 Cluster Produk Terlaris")
                    st.write("Produk-produk ini memiliki volume penjualan paling tinggi (Bintang):")
                    st.write(df_final[df_final['Cluster'] == id_laris]['PRODUK'].unique()[:15])
                with c2:
                    st.warning(f"### 📉 Cluster Produk Perlu Promo")
                    st.write("Produk-produk ini memiliki volume penjualan rendah (Evaluasi):")
                    st.write(df_final[df_final['Cluster'] == id_kurang]['PRODUK'].unique()[:15])
                
                st.info(f"Metode: Euclidean Distance. Cluster Terlaris memiliki rata-rata penjualan {cluster_sum.iloc[0]:.1f} unit.")
            else:
                st.error("Gagal memproses angka. Pastikan kolom Jumlah dan Harga berisi data numerik.")
        else:
            st.error("Kolom Produk, Jumlah, atau Harga tidak ditemukan.")
            st.info(f"Kolom yang terdeteksi: {list(df.columns)}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
else:
    st.info("👋 Selamat Datang. Silakan unggah file CSV data penjualan Anda di sidebar sebelah kiri.")