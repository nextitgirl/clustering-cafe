import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Data Mining Cafe Bumbum", layout="wide")
st.title("📊 Sistem Clustering Penjualan Cafe")

uploaded_file = st.sidebar.file_uploader("Upload File Rekap Penjualan (CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca data tanpa skip baris
    df = pd.read_csv(uploaded_file)
    
    # Bersihkan data: hapus baris kosong & pastikan angka valid
    df = df.dropna()
    df['JUMLAH'] = pd.to_numeric(df['JUMLAH'], errors='coerce')
    df['HARGA'] = pd.to_numeric(df['HARGA'], errors='coerce')
    df = df.dropna(subset=['JUMLAH', 'HARGA'])

    st.write("### Data Terdeteksi", df.head())

    # Proses K-Means
    X = df[['JUMLAH', 'HARGA']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualisasi
    st.write("### Grafik Hasil Clustering")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='JUMLAH', y='HARGA', hue='Cluster', palette='viridis', s=100, ax=ax)
    st.pyplot(fig)
    
    st.success("Analisis Selesai!")