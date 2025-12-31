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
        
        # 2. BACA DATA DENGAN DEEP SCAN (Mencari judul kolom yang benar)
        # Kami membaca data tanpa header dulu untuk mencari di mana judul asli berada
        raw_df = pd.read_csv(uploaded_file, sep=separator, header=None, engine='python').astype(str)
        
        target_row = None
        keywords = ['PRODUK', 'ITEM', 'BARANG', 'JUMLAH', 'QTY', 'HARGA', 'PRICE']
        
        for i, row in raw_df.iterrows():
            row_str = " ".join(row.values).upper()
            if any(key in row_str for key in keywords):
                target_row = i
                break
        
        if target_row is not None:
            # Baca ulang mulai dari baris yang ditemukan
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
                    st.success(f"✅ Data Terdeteksi di baris ke-{target_row+1}")
                    st.dataframe(df_final.head(), use_container_width=True)

                    # --- K-MEANS ---
                    X = df_final[['JUMLAH', 'HARGA']]
                    X_scaled = StandardScaler().fit_transform(X)
                    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
                    df_final['Kelompok'] = kmeans.labels_

                    # --- VISUALISASI ---
                    st.markdown("---")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.scatterplot(data=df_final, x='JUMLAH', y='HARGA', hue='Kelompok', palette='viridis', s=200, ax=ax)
                    st.pyplot(fig)
                    st.success("Analisis Selesai!")
                else:
                    st.error("Data ditemukan tapi isinya bukan angka yang valid.")
            else:
                st.error("Gagal memetakan kolom Produk, Jumlah, dan Harga.")
        else:
            st.error("Sistem tidak menemukan baris yang berisi judul kolom (Produk, Jumlah, Harga).")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("👋 Silakan unggah file CSV Anda.")