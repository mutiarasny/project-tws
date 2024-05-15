from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Definisikan struktur data input menggunakan Pydantic BaseModel
class SleepData(BaseModel):
    Gender: str
    Age: int
    Occupation: str
    Sleep_Duration: float
    Quality_of_Sleep: int
    Physical_Activity_Level: int
    Stress_Level: int
    BMI_Category: str
    Blood_Pressure: str
    Heart_Rate: int
    Daily_Steps: int

# Memuat model prediksi dari file pickle
model = joblib.load("SleepHealth_pipeline.pkl")

# Definisikan route untuk prediksi
@app.post("/predict")
async def predict_sleep_disorder(data: SleepData):
    # Ubah data masukan menjadi format yang diperlukan oleh model
    input_data = [[ data.Gender, data.Age, data.Occupation, data.Sleep_Duration, data.Quality_of_Sleep, data.Physical_Activity_Level, data.Stress_Level,
    data.BMI_Category, data.Blood_Pressure, data.Heart_Rate, data.Daily_Steps]]
    
    # Konversi menjadi DataFrame
    input_data_df = pd.DataFrame(input_data, columns=['Gender','Age' ,'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                                                      'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps'])

    # Lakukan prediksi dengan model
    prediction = model.predict(input_data_df)
    
    # Kembalikan hasil prediksi sebagai respons API
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)