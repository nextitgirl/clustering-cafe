import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PENGATURAN TAMPILAN ---
st.set_page_config(page_title="Analisis Penjualan Produk", layout="wide")

# --- PESAN SAMBUTAN AESTHETIC ---
st.markdown(
    """
    <div style='text-align: center; padding: 25px; background-color: #f0f2f6; border-radius: 15px; border-left: 10px solid #4CAF50;'>
        <h1 style='color: #2e7d32;'>👋 Selamat Datang di Sistem Analisis Penjualan 👋</h1>
        <p style='font-size: 1.2em; color: #555;'>Solusi cerdas untuk mengetahui produk yang <b>Paling Laku</b> dan <b>Kurang Diminati</b> di usaha Anda.</p>
        <p style='font-size: 1.1em;'>Cukup masukkan data satu bulan, dan biarkan sistem bekerja untuk Anda!</p>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# --- INSTRUKSI PENGGUNAAN ---
with st.expander("📖 Baca Petunjuk Penggunaan (Penting!)"):
    st.markdown("""
    1.  **Siapkan File:** Pastikan Anda memiliki file CSV (bisa disimpan dari Excel).
    2.  **Isi Data:** Pastikan ada judul kolom **Produk**, **Jumlah**, dan **Harga** (huruf besar/kecil tidak masalah).
    3.  **Upload:** Klik tombol di samping kiri untuk memasukkan file Anda.
    4.  **Hasil:** Sistem akan mengelompokkan produk Anda menjadi beberapa kelompok (Cluster) secara otomatis.
    """)

st.sidebar.header("📂 Menu Data")
uploaded_file = st.sidebar.file_uploader("Upload Rekap Penjualan (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data dengan deteksi pemisah otomatis
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Seragamkan nama kolom (Hapus spasi & Jadikan Huruf Besar)
        df.columns = df.columns.str.strip().str.upper()
        
        # Cari kolom yang mirip dengan PRODUK, JUMLAH, dan HARGA
        col_p = next((c for c in df.columns if 'PRODUK' in c or 'NAMA' in c or 'ITEM' in c), None)
        col_j = next((c for c in df.columns if 'JUMLAH' in c or 'QTY' in c or 'TERJUAL' in c), None)
        col_h = next((c for c in df.columns if 'HARGA' in c or 'PRICE' in c or 'TOTAL' in c), None)

        if col_p and col_j and col_h:
            df = df.rename(columns={col_p: 'PRODUK', col_j: 'JUMLAH', col_h: 'HARGA'})
            
            # Pembersihan data angka dari simbol Rp, titik, atau koma
            for col in ['JUMLAH', 'HARGA']:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[^\d]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=['JUMLAH', 'HARGA'])
            
            if not df.empty:
                st.subheader("✅ Data Berhasil Ditemukan")
                st.dataframe(df[['PRODUK', 'JUMLAH', 'HARGA']].head(10), use_container_width=True)
                
                # --- PROSES CLUSTERING ---
                X = df[['JUMLAH', 'HARGA']]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                st.sidebar.markdown("---")
                n_clusters = st.sidebar.slider("Tentukan Jumlah Kelompok", 2, 4, 3)
                
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                df['Kelompok'] = kmeans.fit_predict(X_scaled)

                # --- VISUALISASI ---
                st.markdown("---")
                st.subheader("📈 Grafik Pengelompokan Penjualan")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x='JUMLAH', y='HARGA', hue='Kelompok', palette='deep', s=200, ax=ax)
                plt.title("Peta Persebaran Produk", fontsize=15)
                st.pyplot(fig)

                # --- KESIMPULAN ---
                st.subheader("📋 Ringkasan Analisis")
                for i in range(n_clusters):
                    subset = df[df['Kelompok'] == i]
                    st.write(f"**Kelompok {i+1}:** Terdiri dari {len(subset)} jenis produk.")
                
                st.success("Analisis selesai! Sekarang Anda tahu produk mana yang harus diperbanyak stoknya.")
            else:
                st.error("Data Anda tidak mengandung angka yang bisa dihitung. Mohon periksa file Anda.")
        else:
            st.error("Sistem tidak menemukan kolom Produk, Jumlah, atau Harga. Silakan sesuaikan judul kolom di file Anda.")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan teknis: {e}")
else:
    st.warning("⚠️ Silakan masukkan data Anda di sebelah kiri. Pastikan data sudah benar agar hasil maksimal!")