from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel
import os

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Menambahkan CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Ganti dengan domain frontend atau "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menentukan path absolut untuk model dan scaler
model_path = os.path.join(os.getcwd(), 'app', 'model_xgboost.pkl')
scaler_path = os.path.join(os.getcwd(), 'app', 'scaler.pkl')

# Load model dan scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Schema data input untuk validasi request
class LeadData(BaseModel):
    lead_status: int
    response_time: int
    interaction_level: int
    source: str

# Definisikan endpoint POST untuk prediksi
@app.post("/predict")
async def predict(lead_data: LeadData):
    try:
        # Menyusun data input dalam format DataFrame
        input_data = pd.DataFrame([{
            'lead_status': lead_data.lead_status,
            'response_time': lead_data.response_time,
            'interaction_level': lead_data.interaction_level,
            'source': lead_data.source
        }])

        # Encode categorical variables (sumber lead)
        input_data = pd.get_dummies(input_data)
        required_columns = ['response_time', 'interaction_level', 'lead_status', 'source_ad', 'source_email', 'source_referral', 'source_search engine', 'source_social media']

        # Menambahkan kolom yang hilang
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Menyusun ulang kolom agar sesuai dengan urutan yang benar
        input_data = input_data[required_columns]

        # Scaling pada kolom numerik
        input_data[['response_time', 'interaction_level', 'lead_status']] = scaler.transform(input_data[['response_time', 'interaction_level', 'lead_status']])

        # Prediksi menggunakan model
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        # Mengambil nilai prediksi dan probabilitas langsung sebagai scalar
        predicted_class = int(prediction)  # Nilai scalar
        predicted_probability = float(probability[0][1])  # Nilai scalar dari proba

        # Mengembalikan hasil prediksi
        return {"prediction": predicted_class, "probability": predicted_probability}
    
    except Exception as e:
        # Menangani error dan memberikan respon 500 jika ada masalah
        raise HTTPException(status_code=500, detail=str(e))

