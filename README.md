# Airflow HW: Car Price Prediction

DAG: `car_price_prediction`

## Project structure
- `dags/hw_dag.py` — Airflow DAG (tasks: pipeline -> predict)
- `modules/` — training and prediction logic

## Result
After DAG run, predictions are saved to `data/predictions/preds_<timestamp>.csv`.
