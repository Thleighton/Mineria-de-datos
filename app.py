import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_sellingprice_web.pkl")
MMR_LOOKUP_PATH = os.path.join(BASE_DIR, "mmr_lookup.csv")

app = Flask(__name__)

# Cargar el modelo una sola vez al arrancar
modelo = joblib.load(MODEL_PATH)
mmr_lookup = pd.read_csv(MMR_LOOKUP_PATH)

for col in ["make", "model", "body", "transmission"]:
    mmr_lookup[col] = mmr_lookup[col].fillna("").astype(str).str.strip().str.lower()

mmr_lookup["year"] = pd.to_numeric(mmr_lookup["year"], errors="coerce").astype("Int64")
mmr_lookup["mmr_estimado"] = pd.to_numeric(mmr_lookup["mmr_estimado"], errors="coerce")
mmr_lookup["cantidad_registros"] = pd.to_numeric(
    mmr_lookup["cantidad_registros"],
    errors="coerce",
).fillna(1)

MMR_MEDIANA_GENERAL = float(mmr_lookup["mmr_estimado"].median())

# Categorias conocidas por el OneHotEncoder (para poblar los desplegables)
ohe = modelo.named_steps["preprocesador"].named_transformers_["categoricas"]
MAKES, BODIES, TRANSMISSIONS = (list(c) for c in ohe.categories_)
MODELS = sorted(m for m in mmr_lookup["model"].dropna().unique() if m)

# Orden exacto de columnas que espera el pipeline
FEATURES = ["year", "condition", "odometer", "mmr", "make", "body", "transmission"]


def normalizar_texto(valor):
    return str(valor or "").strip().lower()


def calcular_mmr_promedio(filas):
    if filas.empty:
        return None

    pesos = filas["cantidad_registros"].clip(lower=1)
    return float((filas["mmr_estimado"] * pesos).sum() / pesos.sum())


def buscar_mmr_desde_dataset(valores):
    year = int(valores["year"])
    make = normalizar_texto(valores["make"])
    model = normalizar_texto(valores["model"])
    body = normalizar_texto(valores["body"])
    transmission = normalizar_texto(valores["transmission"])

    filtros = [
        {"year": year, "make": make, "model": model, "body": body, "transmission": transmission},
        {"year": year, "make": make, "model": model},
        {"year": year, "make": make, "body": body, "transmission": transmission},
        {"year": year, "make": make},
        {"make": make},
    ]

    for filtro in filtros:
        filas = mmr_lookup
        for columna, valor in filtro.items():
            filas = filas[filas[columna] == valor]

        mmr = calcular_mmr_promedio(filas)
        if mmr is not None:
            return round(mmr)

    return round(MMR_MEDIANA_GENERAL)


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    valores = {
        "year": 2015,
        "condition": 30,
        "odometer": 50000,
        "make": MAKES[0],
        "model": MODELS[0] if MODELS else "",
        "body": BODIES[0],
        "transmission": TRANSMISSIONS[0],
    }
    valores["mmr"] = buscar_mmr_desde_dataset(valores)

    if request.method == "POST":
        try:
            valores.update({
                "year": int(request.form["year"]),
                "condition": float(request.form["condition"]),
                "odometer": float(request.form["odometer"]),
                "make": request.form["make"],
                "model": request.form["model"],
                "body": request.form["body"],
                "transmission": request.form["transmission"],
            })
            valores["mmr"] = buscar_mmr_desde_dataset(valores)
            X = pd.DataFrame([[valores[f] for f in FEATURES]], columns=FEATURES)
            prediccion = float(modelo.predict(X)[0])
        except (ValueError, KeyError):
            error = "Revisa que todos los campos numericos esten completos y sean validos."

    return render_template(
        "index.html",
        makes=MAKES,
        models=MODELS,
        bodies=BODIES,
        transmissions=TRANSMISSIONS,
        prediccion=prediccion,
        error=error,
        valores=valores,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
