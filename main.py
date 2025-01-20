from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import pickle
from fastapi.staticfiles import StaticFiles

import os

app = FastAPI()

with open('turismo_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)
with open('activities_columns.pkl', 'rb') as activities_file:
    activities_columns = pickle.load(activities_file)
with open('transport_columns.pkl', 'rb') as transport_file:
    transport_columns = pickle.load(transport_file)
with open('city_dept_mapping.pkl', 'rb') as city_dept_file:
    city_dept_mapping = pickle.load(city_dept_file)
with open('name_activities_mapping.pkl', 'rb') as mapping_file:
    name_activities_mapping = pickle.load(mapping_file)

clima_mapping = {'Cálido': 3, 'Desierto': 2, 'Templado': 1, 'Frío': 0}
costo_mapping = {'Ultra': 4, 'Alto': 3, 'Moderado': 2, 'Bajo': 1}
edades_mapping = {"Todo público": 0, "Adultos": 1}

app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/", response_class=HTMLResponse)
async def form():
    html_file = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    return FileResponse(html_file)

class UserInput(BaseModel):
    clima: str
    costo: str
    transporte: str
    edades: str
    actividades: str

@app.post("/recomendar")
async def recomendar(user_input: UserInput):
    clima = clima_mapping.get(user_input.clima)
    costo = costo_mapping.get(user_input.costo)
    edades = edades_mapping.get(user_input.edades)

    if clima is None or costo is None or edades is None:
        return {"error": "Entrada inválida en una de las opciones. Verifica tus respuestas."}

    transportes_usuario = user_input.transporte.split(", ")
    transport_input = {transport: 1 if transport in transportes_usuario else 0 for transport in transport_columns}
    actividades_usuario = user_input.actividades.split(", ")
    activity_input = {activity: 1 if activity in actividades_usuario else 0 for activity in activities_columns}

    user_input_dict = {
        "Clima": label_encoders["Clima"].transform([clima])[0],
        "Estimated_Cost": label_encoders["Estimated_Cost"].transform([costo])[0],
        "Ages": label_encoders["Ages"].transform([edades])[0],
    }
    user_input_dict.update(transport_input)
    user_input_dict.update(activity_input)

    user_input_df = pd.DataFrame([user_input_dict])

    missing_cols = set(model.feature_names_in_) - set(user_input_df.columns)
    for col in missing_cols:
        user_input_df[col] = 0
    user_input_df = user_input_df[model.feature_names_in_]

  
    predicted_site = model.predict(user_input_df)[0]
    site_images = {
        'Nevado del Ruiz': [
            'images/nevado_del_ruiz1.jpg',
            'images/nevado_del_ruiz2.jpg',
            'images/nevado_del_ruiz3.jpg'
        ],
        'Playa': [
            'images/playa1.jpg',
            'images/playa2.jpg',
            'images/playa3.jpg',
        ],
        'Monte Alto': [
            "images/monte_alto1.jpg",
            "images/monte_alto2.jpg",
            "images/monte_alto3.jpg"
        ],
        'Zoológico de Cali': [
            "images/zoologicodecali1.jpg",
        ],
        "Casa Terracota": [
            "images/casaterracota.jpg",
        ],
        'Museo del Oro de Bogotá':[
            'images/museooro.jpg',
        ],
        'Punta Gallinas':[
            'images/puntagallina.jpg',
        ]
    }
    
    image = site_images.get(predicted_site, "images/default.jpg")

    city_dept = city_dept_mapping.get(predicted_site, {"City": "Desconocida", "Department": "Desconocido"})
    predicted_site_index = 0
    activities = name_activities_mapping.get(predicted_site, ["Actividades no disponibles."])
    return {
        "recomendación": f"{predicted_site}",
        "ubicación": {
            "ciudad": city_dept['City'],
            "departamento": city_dept['Department']
        },
        "actividades": activities, "imagenes": image
        }


