import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv(r'C:\Users\busto\OneDrive\Área de Trabalho\Turismo_Colombia.csv')

df = df.dropna()
df = df.drop(['Opinions', 'Durations', 'ID_destino'], axis=1)

city_dept_mapping = df.groupby('Name')[['City', 'Department']].first().to_dict(orient='index')
name_activities_mapping = (df.groupby("Name")["Activities"].apply(lambda x: list(set(", ".join(x).split(", ")))).to_dict())


cost_mapping = {'$$$$': 4, '$$$': 3, '$$': 2, '$': 1}
df['Estimated_Cost'] = df['Estimated_Cost'].map(cost_mapping).fillna(0).astype(int)

clima_mapping = {'Cálido': 3, 'Desierto': 2, 'Templado': 1, 'Frío': 0}
df['Clima'] = df['Clima'].map(clima_mapping)

ages_mapping = {"Todo público": 0, "Adultos": 1}
df['Ages'] = df['Ages'].map(ages_mapping).astype(int)

X_interest = ['Clima', 'Activities', 'Estimated_Cost', 'Transport_Type', 'Ages', 'Name']
data_filtered = df[X_interest].copy()

label_encoders = {}
for column in ["Clima", "Estimated_Cost", "Ages"]:
    le = LabelEncoder()
    data_filtered[column] = le.fit_transform(data_filtered[column])
    label_encoders[column] = le

data_filtered['Transport_Type'] = data_filtered['Transport_Type'].astype(str)
transport_expanded = data_filtered['Transport_Type'].str.get_dummies(sep=', ')

data_filtered['Activities'] = data_filtered['Activities'].astype(str)
activities_expanded = data_filtered['Activities'].str.get_dummies(sep=', ')

data_processed = pd.concat([data_filtered.drop(columns=["Transport_Type", "Activities"]), transport_expanded, activities_expanded], axis=1)

balanced_data = pd.concat([
    resample(data_processed[data_processed["Name"] == name], replace=True, n_samples=200, random_state=42)
    for name in data_processed["Name"].unique()
])

X = balanced_data.drop(columns=["Name"])
y = balanced_data["Name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('turismo_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)

activities_columns = list(activities_expanded.columns)
with open('activities_columns.pkl', 'wb') as activities_file:
    pickle.dump(activities_columns, activities_file)

transport_columns = list(transport_expanded.columns)
with open('transport_columns.pkl', 'wb') as transport_file:
    pickle.dump(transport_columns, transport_file)
    
city_dept_mapping = df.groupby('Name')[['City', 'Department']].first().to_dict(orient='index')
with open('city_dept_mapping.pkl', 'wb') as city_dept_file:
    pickle.dump(city_dept_mapping, city_dept_file)
    
with open('name_activities_mapping.pkl', 'wb') as mapping_file:
    pickle.dump(name_activities_mapping, mapping_file)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print("Columnas de actividades procesadas:")
print(activities_expanded.columns)
print("Columnas de transporte procesadas:")
print(transport_expanded.columns)
