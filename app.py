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


def estimar_mmr(valores):
    antiguedad = max(0, 2026 - valores["year"])
    mmr = 32000 - (antiguedad * 1500)
    mmr += (valores["condition"] - 25) * 280
    mmr -= valores["odometer"] * 0.055

    if valores["transmission"].lower() == "automatic":
        mmr += 900

    body = valores["body"].lower()
    if "suv" in body:
        mmr += 2200
    elif "sedan" in body:
        mmr += 600
    elif "truck" in body or "pickup" in body:
        mmr += 2800
    elif "hatch" in body:
        mmr -= 300

    make = valores["make"].lower()
    if make in {"bmw", "mercedes-benz", "audi", "lexus", "infiniti", "acura", "cadillac"}:
        mmr += 5000
    elif make in {"toyota", "honda", "subaru"}:
        mmr += 1800
    elif make in {"kia", "hyundai", "chevrolet", "ford", "nissan"}:
        mmr += 700

    return round(max(1000, mmr))


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    valores = {
        "year": 2015,
        "condition": 30,
        "odometer": 50000,
        "make": MAKES[0],
        "body": BODIES[0],
        "transmission": TRANSMISSIONS[0],
    }
    valores["mmr"] = estimar_mmr(valores)

    if request.method == "POST":
        try:
            valores.update({
                "year": int(request.form["year"]),
                "condition": float(request.form["condition"]),
                "odometer": float(request.form["odometer"]),
                "make": request.form["make"],
                "body": request.form["body"],
                "transmission": request.form["transmission"],
            })
            valores["mmr"] = estimar_mmr(valores)
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
