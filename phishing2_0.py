import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pickle


# Leer datos del archivo CSV
data = pd.read_csv("dataset.csv")

# Seleccionar las características importantes para el modelo
features = ['length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_slash', 'nb_comma', 'nb_dollar', 'nb_space', 'http_in_path']

# Separar características y etiquetas (clases)
X = data[features]
y = data['status']

# Imputar valores faltantes
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)



# Entrenar el modelo de Random Forest
model = RandomForestClassifier(random_state=0).fit(X, y)
print(model.feature_importances_)

for feature, importance in zip(features, model.feature_importances_):
    print(feature, importance)
# Guardar el modelo en un archivo pickle
pickle.dump(model, open('model.pkl', 'wb'))
