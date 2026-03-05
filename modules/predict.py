import os
import json
import glob
import dill
import pandas as pd
from datetime import datetime


def load_model():
    model_path = max(glob.glob("data/models/*.pkl"), key=os.path.getmtime)
    with open(model_path, "rb") as f:
        model = dill.load(f)
    return model


def load_test_data():

    files = glob.glob("data/test/*.json")

    data = []

    for file in files:

        with open(file) as f:
            data.append(json.load(f))

    return pd.DataFrame(data)


def make_predictions(model, df):

    car_ids = df["id"]

    df = df.drop(columns=["id"])

    preds = model.predict(df)

    return pd.DataFrame({
        "car_id": car_ids,
        "pred": preds
    })


def save_predictions(preds):

    os.makedirs("data/predictions", exist_ok=True)

    filename = f"data/predictions/preds_{datetime.now().strftime('%Y%m%d%H%M')}.csv"

    preds.to_csv(filename, index=False)

    print(f"Predictions saved to {filename}")


def predict():

    model = load_model()

    df = load_test_data()

    preds = make_predictions(model, df)

    save_predictions(preds)


if __name__ == "__main__":
    predict()