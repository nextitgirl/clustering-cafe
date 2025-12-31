import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- TAMPILAN ---
st.set_page_config(page_title="Analisis Penjualan UMKM", layout="wide")

st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 15px; border: 2px solid #4CAF50;'>
        <h1 style='color: #1b5e20;'>📊 Sistem Analisis Penjualan Otomatis</h1>
        <p style='font-size: 1.1em; color: #333;'>Bantu UMKM berkembang dengan data. Cukup upload, sistem kami yang bekerja!</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("📁 Unggah Data Anda")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. DETEKSI PEMISAH
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        separator = ';' if content.count(';') > content.count(',') else ','
        uploaded_file.seek(0)
        
        # 2. BACA DATA DENGAN DEEP SCAN
        raw_df = pd.read_csv(uploaded_file, sep=separator, header=None, engine='python').astype(str)
        
        target_row = None
        keywords = ['PRODUK', 'ITEM', 'BARANG', 'JUMLAH', 'QTY', 'HARGA', 'PRICE']
        
        for i, row in raw_df.iterrows():
            row_str = " ".join(row.values).upper()
            if any(key in row_str for key in keywords):
                target_row = i
                break
        
        if target_row is not None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=separator, skiprows=target_row, engine='python')
            df.columns = df.columns.str.strip().str.upper()
            
            # Mapping Kolom Otomatis
            col_map = {}
            mapping_logic = {
                'PRODUK': ['PRODUK', 'ITEM', 'BARANG', 'NAMA'],
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
                
                # Pembersihan angka
                for col in ['JUMLAH', 'HARGA']:
                    df_final[col] = df_final[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                
                df_final = df_final.dropna()

                if not df_final.empty:
                    # --- FITUR SLIDER K ---
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("Pengaturan Kelompok")
                    n_clusters = st.sidebar.slider("Pilih Jumlah Kelompok (K)", 2, 5, 3)

                    # --- K-MEANS ---
                    X = df_final[['JUMLAH', 'HARGA']]
                    X_scaled = StandardScaler().fit_transform(X)
                    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10).fit(X_scaled)
                    df_final['Kelompok'] = kmeans.labels_

                    # --- VISUALISASI ---
                    st.markdown("---")
                    st.subheader("📈 Peta Kelompok Penjualan")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df_final, x='JUMLAH', y='HARGA', hue='Kelompok', palette='viridis', s=200, ax=ax, style='Kelompok')
                    plt.title(f"Persebaran Produk dalam {n_clusters} Kelompok")
                    st.pyplot(fig)
                    
                    # --- TABEL DATA ---
                    st.subheader("📄 Tabel Hasil Analisis")
                    st.dataframe(df_final, use_container_width=True)

                    # --- FITUR BARU: KESIMPULAN OTOMATIS ---
                    st.markdown("---")
                    st.subheader("💡 Kesimpulan Strategis")
                    
                    # Hitung rata-rata jumlah terjual per kelompok
                    cluster_summary = df_final.groupby('Kelompok')['JUMLAH'].mean().sort_values(ascending=False)
                    
                    # Kelompok Terlaris (Rata-rata penjualan tertinggi)
                    id_laris = cluster_summary.index[0]
                    produk_laris = df_final[df_final['Kelompok'] == id_laris]['PRODUK'].unique()
                    
                    # Kelompok Kurang Laku (Rata-rata penjualan terendah)
                    id_kurang = cluster_summary.index[-1]
                    produk_kurang = df_final[df_final['Kelompok'] == id_kurang]['PRODUK'].unique()

                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.success(f"### 🌟 Produk Paling Laku (Kelompok {id_laris+1})")
                        st.write("Produk ini memiliki minat yang sangat tinggi. Pastikan stok selalu tersedia!")
                        for p in produk_laris[:10]: # Tampilkan max 10 produk
                            st.write(f"- {p}")
                    
                    with c2:
                        st.warning(f"### 📉 Produk Kurang Laku (Kelompok {id_kurang+1})")
                        st.write("Produk ini penjualannya masih rendah. Pertimbangkan untuk memberikan promo atau evaluasi harga.")
                        for p in produk_kurang[:10]: # Tampilkan max 10 produk
                            st.write(f"- {p}")
                    
                    st.info("Catatan: Kesimpulan ini diambil berdasarkan rata-rata jumlah produk yang terjual dalam satu bulan.")

                else:
                    st.error("Data ditemukan tapi isinya bukan angka yang valid.")
            else:
                st.error("Gagal memetakan kolom. Pastikan ada judul PRODUK, JUMLAH, dan HARGA.")
        else:
            st.error("Sistem tidak menemukan baris judul kolom.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan unggah file CSV Anda.")