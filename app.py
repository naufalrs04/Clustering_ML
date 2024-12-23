from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.title("Clustering Obesitas")

model = joblib.load('kmeans_model.pkl') 
pca = joblib.load('pca_model.pkl')
original_data = pd.read_csv('obesity_data.csv')
original_data['Cluster'] = model.labels_

cluster_stats = original_data.groupby('Cluster').agg({
    'Age': 'mean',
    'Height': 'mean',
    'Weight': 'mean',
    'FCVC': 'mean',
    'NCP': 'mean',
    'CH2O': 'mean',
    'FAF': 'mean',
    'TUE': 'mean',
})

#Input Data
st.sidebar.header("Input Data:")
age = st.sidebar.slider("Age", min_value=5, max_value=100, step=1, value=25)
gender = st.sidebar.selectbox(
    "Gender",
    options=['Female', 'Male'],
    index=0
)
height = st.sidebar.slider("Height (m)", min_value=1.0, max_value=2.0, step=0.01, value=1.65)
weight = st.sidebar.slider("Weight (kg)", min_value=10.0, max_value=200.0, step=0.1, value=70.0)
fcvc = st.sidebar.slider("FCVC (Frequency of Consumption of Vegetables)", min_value=0.0, max_value=10.0, step=0.1, value=4.0)
ncp = st.sidebar.slider("NCP (Number of Main Meals)", min_value=0.0, max_value=10.0, step=0.1, value=2.0)
ch2o = st.sidebar.slider("Daily Water Intake (CH2O)", min_value=0.1, max_value=10.0, step=0.1, value=2.0)
faf = st.sidebar.slider("FAF (Frequency of Physical Activity)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
tue = st.sidebar.slider("TUE (Time Using Technology Devices)", min_value=0.0, max_value=10.0, step=0.1, value=6.0)
cal = st.sidebar.selectbox(
    "Consumption of Alcohol (CALC)",
    options=['Always', 'Frequently', 'Sometimes', 'no'],
    index=3
)
favc = st.sidebar.selectbox(
    "Frequent Consumption of High Calorie Food (FAVC)",
    options=['no', 'yes'],
    index=0
)
scc = st.sidebar.selectbox(
    "Do you suffer from Stress (SCC)?",
    options=['no', 'yes'],
    index=0
)
smoke = st.sidebar.selectbox(
    "Do you Smoke (SMOKE)?",
    options=['no', 'yes'],
    index=0
)
family_history = st.sidebar.selectbox(
    "Family History with Overweight",
    options=['no', 'yes'],
    index=0
)
caec = st.sidebar.selectbox(
    "Consumption of Carbohydrates (CAEC)",
    options=['Always', 'Frequently', 'Sometimes', 'no'],
    index=3
)
mtrans = st.sidebar.selectbox(
    "Mode of Transportation (MTRANS)",
    options=['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'],
    index=0
)

gender_mapping = {'Female': 0, 'Male': 1}
cal_mapping = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
favc_mapping = {'no': 0, 'yes': 1}
scc_mapping = {'no': 0, 'yes': 1}
smoke_mapping = {'no': 0, 'yes': 1}
family_history_mapping = {'no': 0, 'yes': 1}
caec_mapping = {'Always': 0, 'Frequently': 1, 'Sometimes': 2, 'no': 3}
mtrans_mapping = {'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 'Public_Transportation': 3, 'Walking': 4}

gender = gender_mapping[gender]
cal = cal_mapping[cal]
favc = favc_mapping[favc]
scc = scc_mapping[scc]
smoke = smoke_mapping[smoke]
family_history = family_history_mapping[family_history]
caec = caec_mapping[caec]
mtrans = mtrans_mapping[mtrans]

input_data = pd.DataFrame({
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CH2O': [ch2o],
    'FAF': [faf],
    'TUE': [tue],
    'Gender': [gender],
    'CALC': [cal],
    'FAVC': [favc],
    'SCC': [scc],
    'SMOKE': [smoke],
    'family_history_with_overweight': [family_history],
    'CAEC': [caec],
    'MTRANS': [mtrans]
})

st.write("""
    Aplikasi ini memprediksi cluster obesitas berdasarkan data yang dimasukkan.
    Silakan pilih nilai untuk setiap fitur di sidebar untuk melihat hasil prediksi.
""")

if st.button("Prediksi"):
    pca_data = pca.transform(input_data)

    predicted_cluster = model.predict(pca_data)

    st.subheader("Hasil Prediksi:")
    st.write(f"Data baru masuk ke cluster : {predicted_cluster[0]}")

    input_df = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Predicted Cluster': predicted_cluster
    })

    st.subheader("Visualisasi Clustering setelah PCA")
    pca_transformed = pca.transform(original_data.drop(columns=['Cluster']))

    plt.figure(figsize=(12, 8))
    plt.scatter(
        pca_transformed[:, 0], 
        pca_transformed[:, 1], 
        c=original_data['Cluster'], 
        cmap='viridis', 
        s=50, 
        alpha=0.7, 
        edgecolor='k'
    )
    plt.title("Clustering dengan KMeans setelah PCA", fontsize=16)
    plt.xlabel("Fitur 1 (PCA)", fontsize=12)
    plt.ylabel("Fitur 2 (PCA)", fontsize=12)
    plt.colorbar(label='Cluster')
    plt.grid(alpha=0.3)

    new_pca_point = pca.transform(input_data)
    plt.scatter(
        new_pca_point[0, 0], 
        new_pca_point[0, 1], 
        color='red', 
        marker='X', 
        s=200, 
        label='Data Baru', 
        edgecolor='k'
    )

    plt.legend(fontsize=10)
    st.pyplot(plt)

    st.subheader("Rata-Rata Per Cluster")
    st.write(cluster_stats)

    st.subheader("Visualisasi Clustering Berdasarkan Tinggi dan Berat Badan")
    plt.figure(figsize=(12, 8))

    for cluster in range(model.n_clusters):
        cluster_data = original_data[original_data['Cluster'] == cluster]
        plt.scatter(
            cluster_data['Height'], 
            cluster_data['Weight'], 
            label=f'Cluster {cluster}', 
            alpha=0.7, 
            edgecolor='k'
        )

    plt.scatter(
        height, 
        weight, 
        color='red', 
        marker='X', 
        s=200, 
        label='Data Baru', 
        alpha=1, 
        edgecolor='k'
    )

    plt.title('Distribusi Cluster Berdasarkan Usia dan Berat Badan', fontsize=16)
    plt.xlabel('Tinggi Badan (m)', fontsize=12)
    plt.ylabel('Berat Badan (kg)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    st.pyplot(plt)

    st.subheader("Boxplot Fitur Numerikal Berdasarkan Cluster")
    scaler = MinMaxScaler()
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    numerical_data = original_data[numerical_features]
    numerical_scaled = pd.DataFrame(scaler.fit_transform(numerical_data), columns=numerical_features)

    input_data_scaled = scaler.transform(input_data[numerical_features])
    input_data_scaled = pd.DataFrame(input_data_scaled, columns=numerical_features)
    input_data_scaled['Type'] = 'Data Baru'
    numerical_scaled['Type'] = 'Data Asli' 

    combined_data = pd.concat([numerical_scaled, input_data_scaled])

    long_data = pd.melt(
        combined_data,
        id_vars=['Type'], 
        var_name='Feature',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_data, x='Feature', y='Value', hue='Type', palette={'Data Asli': 'blue', 'Data Baru': 'red'})
    plt.title('Distribusi Gabungan Fitur dengan Data Baru', fontsize=16)
    plt.xlabel('Fitur', fontsize=12)
    plt.ylabel('Nilai (Ternormalisasi)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(alpha=0.3)
    st.pyplot(plt)

    st.subheader("Boxplot Fitur Kategorikal Berdasarkan Cluster")

    categorical_features = ['Gender','CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight','CAEC','MTRANS']
    scaler = MinMaxScaler()
    categorical_data = original_data[categorical_features]
    categorical_scaled = pd.DataFrame(scaler.fit_transform(categorical_data), columns=categorical_features)
    
    input_categorical_scaled = pd.DataFrame([{
        'Gender': gender, 
        'CALC': cal, 
        'FAVC': favc, 
        'SCC': scc, 
        'SMOKE': smoke, 
        'family_history_with_overweight': family_history,
        'CAEC': caec,
        'MTRANS': mtrans,
    }], columns=categorical_features)

    categorical_scaled['Type'] = 'Data Asli'
    input_categorical_scaled['Type'] = 'Data Baru'

    combined_categorical = pd.concat([categorical_scaled, input_categorical_scaled])

    long_categorical_data = pd.melt(
        combined_categorical,
        id_vars=['Type'],
        var_name='Feature',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long_categorical_data, x='Feature', y='Value', hue='Type', palette={'Data Asli': 'blue', 'Data Baru': 'red'})
    plt.title('Distribusi Gabungan Fitur Kategorikal dengan Data Baru', fontsize=16)
    plt.xlabel('Fitur', fontsize=12)
    plt.ylabel('Nilai (Ternormalisasi)', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(alpha=0.3)
    st.pyplot(plt)