import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- TAMPILAN UTAMA ---
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
st.sidebar.info("Gunakan file CSV hasil ekspor dari Excel atau aplikasi kasir Anda.")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. DETEKSI PEMISAH (Koma atau Titik Koma)
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        separator = ';' if content.count(';') > content.count(',') else ','
        uploaded_file.seek(0)
        
        df = pd.read_csv(uploaded_file, sep=separator, engine='python')
        
        # 2. NORMALISASI NAMA KOLOM (Hapus spasi, jadikan huruf besar)
        df.columns = df.columns.str.strip().str.upper()
        
        # 3. LOGIKA CERDAS MENCARI KOLOM
        # Sistem akan mencari kata yang mirip-mirip
        keywords = {
            'PRODUK': ['PRODUK', 'ITEM', 'BARANG', 'NAMA', 'MENU', 'DESKRIPSI'],
            'JUMLAH': ['JUMLAH', 'QTY', 'QUANTITY', 'TERJUAL', 'UNIT', 'VOL'],
            'HARGA': ['HARGA', 'PRICE', 'NILAI', 'TOTAL', 'OMZET', 'SALES']
        }

        found_cols = {}
        for key, synonyms in keywords.items():
            for col in df.columns:
                if any(syn in col for syn in synonyms):
                    found_cols[key] = col
                    break

        # 4. EKSEKUSI JIKA KOLOM DITEMUKAN
        if len(found_cols) >= 3:
            # Gunakan kolom yang ditemukan
            df_final = df[[found_cols['PRODUK'], found_cols['JUMLAH'], found_cols['HARGA']]].copy()
            df_final.columns = ['PRODUK', 'JUMLAH', 'HARGA']
            
            # 5. PEMBERSIHAN DATA ANGKA (Anti-Error Karakter Aneh)
            for col in ['JUMLAH', 'HARGA']:
                # Hapus semua yang bukan angka/titik (seperti Rp, spasi, dll)
                df_final[col] = df_final[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

            df_final = df_final.dropna()

            if not df_final.empty:
                st.success(f"✅ Berhasil mengenali kolom: {found_cols['PRODUK']}, {found_cols['JUMLAH']}, & {found_cols['HARGA']}")
                
                with st.expander("🔍 Lihat Data Terdeteksi"):
                    st.dataframe(df_final.head(10), use_container_width=True)

                # 6. ANALISIS CLUSTERING (K-MEANS)
                X = df_final[['JUMLAH', 'HARGA']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
                df_final['Kelompok'] = kmeans.fit_predict(X_scaled)

                # 7. VISUALISASI AESTHETIC
                st.markdown("---")
                st.subheader("📈 Peta Kelompok Penjualan Produk")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(
                    data=df_final, x='JUMLAH', y='HARGA', 
                    hue='Kelompok', palette='viridis', s=200, ax=ax, style='Kelompok'
                )
                plt.title("Perbandingan Jumlah Terjual vs Harga")
                st.pyplot(fig)

                # 8. KESIMPULAN SEDERHANA UNTUK PEMILIK USAHA
                st.subheader("💡 Tips Strategi Usaha")
                col1, col2, col3 = st.columns(3)
                
                # Cari kelompok mana yang paling laku (Rata-rata jumlah tertinggi)
                top_group = df_final.groupby('Kelompok')['JUMLAH'].mean().idxmax()
                
                with col1:
                    st.info("**Kelompok Terlaris**")
                    st.write(df_final[df_final['Kelompok'] == top_group]['PRODUK'].head(5).values)
                
                with col2:
                    st.warning("**Perlu Promo**")
                    low_group = df_final.groupby('Kelompok')['JUMLAH'].mean().idxmin()
                    st.write(df_final[df_final['Kelompok'] == low_group]['PRODUK'].head(5).values)
                
                with col3:
                    st.success("**Produk Premium**")
                    high_price = df_final.groupby('Kelompok')['HARGA'].mean().idxmax()
                    st.write(df_final[df_final['Kelompok'] == high_price]['PRODUK'].head(5).values)

            else:
                st.error("Data terdeteksi, tapi tidak ada angka yang bisa diolah. Pastikan kolom Jumlah dan Harga berisi angka.")
        else:
            st.error("Gagal mendeteksi kolom secara otomatis.")
            st.info(f"Kolom yang ada di file Anda: {list(df.columns)}")
            st.markdown("**Saran:** Pastikan file Anda punya judul kolom seperti 'Produk', 'Jumlah', dan 'Harga'.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.markdown(
        """
        <div style='text-align: center; margin-top: 50px;'>
            <p style='color: #666;'>👋 Halo! Silakan unggah rekap penjualan bulanan Anda (format .CSV) pada menu di samping kiri.</p>
        </div>
        """,
        unsafe_allow_html=True
    )