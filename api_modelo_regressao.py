import fastapi
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Instância do FastAPI
app = FastAPI()

# Validar input do usuário

class request_body(BaseModel):
    horas_irrigacao : float


# Carregar modelo de machine learning
modelo_irrigacao = joblib.load('./model_predict_irrigation_area.pkl')

# Definir rota
@app.post('/predict')
# Definir método
def predict(data: request_body):
    # Preparar dados
    input_feature = [[data.horas_irrigacao]]

    # Realizar a predição
    y_pred = modelo_irrigacao.predict(input_feature)[0].astype(int)

    # Retornar valor
    return {
        'Área irrigada': y_pred.tolist()
        }
