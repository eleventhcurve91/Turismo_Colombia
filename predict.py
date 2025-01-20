import pandas as pd
import pickle

with open('turismo_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoders.pkl', 'rb') as encoder_file:
    label_encoders = pickle.load(encoder_file)
with open('activities_columns.pkl', 'rb') as activities_file:
    activities_columns = pickle.load(activities_file)

clima_mapping = {'Cálido': 3, 'Desierto': 2, 'Templado': 1, 'Frío': 0}
costo_mapping = {'Ultra': 4, 'Alto': 3, 'Moderado': 2,'Bajo': 1}
edades_mapping = {"Todo público": 0, "Adultos": 1}
# transport_mapping = {
#     "A pie": "A pie",
#     "Vehiculo particular": "Vehiculo particular",
#     "Transporte publico": "Transporte publico",
#     "Avion": "Avion",
#     "Barco": "Barco",
#     "Canoa": "Canoa",
#     "Teleférico": "Teleférico"
# }

def get_valid_input(prompt, options):
    while True:
        print(prompt)
        user_input = input("Tu elección: ").strip()
        if user_input in options:
            return options[user_input]
        print(f"Error: '{user_input}' no es una opción válida. Por favor, elige entre: {list(options.keys())}")

print("\nPor favor, responde las siguientes preguntas para recomendarte un sitio turístico:")
clima = get_valid_input("¿Qué clima prefieres? (Cálido, Frío, Templado, Desierto): ", clima_mapping)
costo = get_valid_input("¿Cuál es tu presupuesto? (Bajo, Medio, Alto): ", costo_mapping)
transporte = get_valid_input("¿Qué tipo de transporte prefieres? (Barco, Avión, Transporte público): ", transport_mapping)
edades = get_valid_input("¿Cuál es el rango de edades? (Todos, Mayor de 18): ", edades_mapping)
actividades = input("¿Qué tipo de actividades te interesan? (Ejemplo: playa, aventura, arqueología): ").strip()

user_input = {
    "Clima": label_encoders["Clima"].transform([clima])[0],
    "Estimated_Cost": label_encoders["Estimated_Cost"].transform([costo])[0],
    "Transport_Type": label_encoders["Transport_Type"].transform([transporte])[0],
    "Ages": label_encoders["Ages"].transform([edades])[0],
}

user_activities = actividades.split(", ")
activity_input = {activity: 1 if activity in user_activities else 0 for activity in activities_columns}

user_input.update(activity_input)
user_input_df = pd.DataFrame([user_input])

missing_cols = set(model.feature_names_in_) - set(user_input_df.columns)
for col in missing_cols:
    user_input_df[col] = 0
user_input_df = user_input_df[model.feature_names_in_]

predicted_site = model.predict(user_input_df)[0]
print(f"\nEl sitio turístico recomendado para ti es: {predicted_site}")
