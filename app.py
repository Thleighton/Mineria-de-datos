import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_sellingprice_web.pkl")

app = Flask(__name__)

# Cargar el modelo una sola vez al arrancar
modelo = joblib.load(MODEL_PATH)

# Categorias conocidas por el OneHotEncoder (para poblar los desplegables)
ohe = modelo.named_steps["preprocesador"].named_transformers_["categoricas"]
MAKES, BODIES, TRANSMISSIONS = (list(c) for c in ohe.categories_)

# Orden exacto de columnas que espera el pipeline
FEATURES = ["year", "condition", "odometer", "mmr", "make", "body", "transmission"]


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    valores = {}

    if request.method == "POST":
        try:
            valores = {
                "year": int(request.form["year"]),
                "condition": float(request.form["condition"]),
                "odometer": float(request.form["odometer"]),
                "mmr": float(request.form["mmr"]),
                "make": request.form["make"],
                "body": request.form["body"],
                "transmission": request.form["transmission"],
            }
            X = pd.DataFrame([[valores[f] for f in FEATURES]], columns=FEATURES)
            prediccion = float(modelo.predict(X)[0])
        except (ValueError, KeyError):
            error = "Revisa que todos los campos numericos esten completos y sean validos."

    return render_template(
        "index.html",
        makes=MAKES,
        bodies=BODIES,
        transmissions=TRANSMISSIONS,
        prediccion=prediccion,
        error=error,
        valores=valores,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
