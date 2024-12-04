import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.title("Prediksi Cluster Obesitas dengan K-Means")
st.write("Masukkan data pada fitur di bawah ini untuk memprediksi cluster obesitas.")

model = joblib.load('kmeans_model.pkl')

# Sidebar
st.sidebar.header("Input Data :")
age = st.sidebar.slider("Age", min_value=5, max_value=100, step=1, value=25)
gender = st.sidebar.selectbox(
    "Gender",
    options=[0, 1],
    index=0,
    format_func=lambda x: "Female" if x == 0 else "Male" 
)
height = st.sidebar.slider("Height (cm)", min_value=50.0, max_value=250.0, step=0.1, value=170.0)
weight = st.sidebar.slider("Weight (kg)", min_value=10.0, max_value=200.0, step=0.1, value=70.0)
favc = st.sidebar.selectbox(
    "Frequent Consumption of High Calorie Food (FAVC)",
    options=[0, 1],
    index=0,
    format_func=lambda x: "No" if x == 0 else "Yes"
)
ch2o = st.sidebar.slider("Daily Water Intake (liters)", min_value=0.1, max_value=10.0, step=0.1, value=2.0)
family_history = st.sidebar.selectbox(
    "Family History with Overweight",
    options=[0, 1],
    index=0,
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# Menambahkan penjelasan di area utama
st.write("""
    Aplikasi ini memprediksi cluster obesitas berdasarkan data yang dimasukkan.
    Silakan pilih nilai untuk setiap fitur di sidebar untuk melihat hasil prediksi.
""")

# Ketika tombol prediksi ditekan
if st.button("Prediksi"):
    # Format input untuk model
    new_data = np.array([[age, gender, height, weight, favc, ch2o, family_history]])

    # Prediksi cluster
    predicted_cluster = model.predict(new_data)

    # Menampilkan hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.write(f"Data baru masuk ke cluster : {predicted_cluster[0]}")
    st.subheader("Visualisasi Data")
    
    # Membuat DataFrame dengan data input baru
    input_df = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Predicted Cluster': predicted_cluster
    })

    # Scatter plot untuk menampilkan distribusi cluster berdasarkan Age dan Weight
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Weight', hue='Predicted Cluster', palette='viridis', data=input_df, s=100, ax=ax)
    ax.set_title('Distribusi Cluster Obesitas Berdasarkan Usia dan Berat Badan')
    ax.set_xlabel('Usia (Tahun)')
    ax.set_ylabel('Berat Badan (kg)')
    st.pyplot(fig)

    # Menampilkan histogram distribusi cluster berdasarkan umur
    st.subheader("Distribusi Cluster Berdasarkan Umur")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(input_df['Age'], bins=10, kde=True, color='blue', ax=ax)
    ax.set_title('Distribusi Umur pada Data Prediksi')
    ax.set_xlabel('Umur')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)

    # Menampilkan histogram distribusi cluster berdasarkan berat badan
    st.subheader("Distribusi Cluster Berdasarkan Berat Badan")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(input_df['Weight'], bins=10, kde=True, color='green', ax=ax)
    ax.set_title('Distribusi Berat Badan pada Data Prediksi')
    ax.set_xlabel('Berat Badan (kg)')
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)