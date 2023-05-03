from fastapi import FastAPI, HTTPException
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi.responses import JSONResponse

model = CatBoostRegressor()
model.load_model("app/linktera_case_model.cbm")

app = FastAPI()


def datetime_features(df_temp):
    """
    Datetime feature üretir.
    """
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['year'] = df_temp['date'].dt.year
    df_temp['dayofweek'] = df_temp['date'].dt.dayofweek
    df_temp['quarter'] = df_temp['date'].dt.quarter
    df_temp['dayofmonth'] = df_temp['date'].dt.day
    df_temp['weekofyear'] = df_temp['date'].dt.weekofyear
    return df_temp


def seasonality_features(df_temp):
    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp.month / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp.month / 12)

    return df_temp


class PredictionInput(BaseModel):
    date: str
    otv_orani: float
    faiz: float
    euro_tl: float
    kredi_stok: float


@app.post('/predict')
async def predict(input_data: PredictionInput):
    inputs_dict = input_data.dict()

    try:
        df = pd.DataFrame([inputs_dict], columns=inputs_dict.keys())
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. "
                                                    "Please use 'dd-mm-yyyy'.")

    for key, value in inputs_dict.items():
        if key != 'date' and value < 0:
            raise HTTPException(status_code=400, detail=f"Invalid value for "
                                                        f"'{key}'. Must be a "
                                                        f"positive number.")

    df = datetime_features(df)
    df = seasonality_features(df)
    df = df.drop(['date'], axis=1)

    df.columns = ['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'month',
                  'year', 'dayofweek', 'quarter', 'dayofmonth', 'weekofyear',
                  'month_sin', 'month_cos']

    predictions = model.predict(df)

    predictions_dict = {"predictions": predictions[0]}
    return JSONResponse(content=predictions_dict, status_code=200)