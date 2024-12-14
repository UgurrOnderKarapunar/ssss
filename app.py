import streamlit as st
import joblib
import pandas as pd
from pyngrok import ngrok

# Authenticate ngrok using your authtoken (you've already done this)
ngrok.set_auth_token('2kCfQjgie0e8E2ogR4uHUZyVOvc_6RxHQp4PZuWVqYcKxyu6f')

# Start the ngrok tunnel to expose the Streamlit app publicly
public_url = ngrok.connect(8501)  # The default port for Streamlit
st.write(f"Streamlit app is live at: {public_url}")

# Load the trained model and preprocessor (replace the path as needed)
model_pipeline = joblib.load("/content/model_pipeline.joblib")  # Update path to your model

# Define Streamlit app layout
st.title('Yolcu Sayısı Prediction')

# Inputs from the user
hat = st.selectbox(
    "Hat",
    [
        'ÇENGELKÖY-KABATAŞ', 'KABATAŞ-ÇENGELKÖY', 'KINALIADA-BURGAZADA-HEYBELİADA-BÜYÜKADA-MALTEPE',
        'MALTEPE-BÜYÜKADA-HEYBELİADA-BURGAZADA-KINALIADA', 'EYÜP-SÜTLÜCE-HASKÖY-FENER-KASIMPAŞA-KADIKÖY',
        'KADIKÖY-KASIMPAŞA-FENER-HASKÖY-SÜTLÜCE-EYÜP', 'BEŞİKTAŞ-KABATAŞ-KARAKÖY-KASIMPAŞA-HASKÖY-SÜTLÜCE-EYÜP',
    ]
)

gemi_tipi = st.selectbox("Gemi Tipi", ['Kiralık Motor', 'Vapur', "Araba Vapuru", "Deniz Taksi", "Deniz Dolmuşu"])
kar_zarar = st.selectbox("Kar-Zarar", ['Kar', 'Zarar'])

mil = st.number_input("Mil", min_value=0, max_value=10000, value=0)
ortalama_yakıt = st.number_input("Ortalama Kullanılan Yakıt(Lt)", min_value=0.0, max_value=10000.0, value=0.0)
yakıt_masrafı = st.number_input("Yakıt Masrafı", min_value=0.0, max_value=10000.0, value=0.0)
litre_ücreti = st.number_input("Litre Ücreti", min_value=0.0, max_value=1000.0, value=0.0)

# Collect the data in a DataFrame
input_data = pd.DataFrame({
    'Hat': [hat],
    'Gemi Tipi': [gemi_tipi],
    'Kar-Zarar': [kar_zarar],
    'Mil': [mil],
    'Ortalama Kullanılan Yakıt(Lt)': [ortalama_yakıt],
    'Yakıt Masrafı': [yakıt_masrafı],
    'Litre Ücreti': [litre_ücreti],
    'Tek Seferde  Kullanılan Toplam Yakıt(Lt)': [0],  # Placeholder value
    'Kar Oranı': [0],  # Placeholder value
    'Kar Miktarı': [0],  # Placeholder value
    'Saatlik Yolcu Ücreti': [0],  # Placeholder value
    'Ortalama Yolcu Başına Ücret': [0]  # Placeholder value
})

# Make the prediction
prediction = model_pipeline.predict(input_data)

# Display the prediction result
st.write(f"Predicted Yolcu Sayısı: {prediction[0]}")
