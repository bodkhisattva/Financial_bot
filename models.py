import numpy as np
import pandas as pd
import time
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import rmse, mape, mae

def make_lag_features(s: pd.Series, lags: int = 10) -> pd.DataFrame: # создаем лаги
    df = pd.DataFrame({"y": s})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    idx = df.index  
    # календарные признаки
    try:
        df["dow"] = idx.dayofweek
        df["month"] = idx.month
    except Exception:
        pass
    df = df.dropna()
    return df

def forecast_ridge(train: pd.Series, test: pd.Series, horizon: int = 30, lags: int = 10):
    df_train = make_lag_features(train, lags=lags) # проходимся по тренировочной выборке
    X_train = df_train.drop(columns=["y"])
    y_train = df_train["y"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    model.fit(X_train, y_train)

    # проходимся по тестовой выборке
    history = train.copy()
    preds_test = []
    for t in range(len(test)):
        df_hist = make_lag_features(history, lags=lags)
        last_row = df_hist.drop(columns=["y"]).iloc[-1:]
        yhat = float(model.predict(last_row)[0])
        preds_test.append(yhat)
        history = pd.concat([history, pd.Series([test.iloc[t]], index=[test.index[t]])])

    preds_test = np.array(preds_test)

    # предсказания на будущее рекурсивно
    full = pd.concat([train, test])
    future_idx = pd.bdate_range(start=full.index[-1], periods=horizon + 1)[1:]
    hist2 = full.copy()
    future_preds = []
    for i in range(horizon):
        df_hist2 = make_lag_features(hist2, lags=lags)
        last_row = df_hist2.drop(columns=["y"]).iloc[-1:]
        yhat = float(model.predict(last_row)[0])
        future_preds.append(yhat)
        hist2 = pd.concat([hist2, pd.Series([yhat], index=[future_idx[i]])])

    return preds_test, pd.Series(future_preds, index=future_idx, name="forecast"), f"Ridge(lags={lags})"

def forecast_sarimax(train: pd.Series, test: pd.Series, horizon: int = 30):
    best_aic = float("inf")
    best_order = None
    best_res = None

    # сетка подбора
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = SARIMAX(
                        train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    res = model.fit(disp=False)
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                        best_res = res
                except:
                    continue

    if best_res is None or best_order is None:
        raise RuntimeError("Не удалось подобрать ARIMA")

    preds_test = best_res.get_forecast(steps=len(test)).predicted_mean.values.astype(float)

    full = pd.concat([train, test])
    model2 = SARIMAX(
        full,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res2 = model2.fit(disp=False)

    future_vals = res2.get_forecast(steps=horizon).predicted_mean.values.astype(float)
    future_idx = pd.bdate_range(start=full.index[-1], periods=horizon + 1)[1:]

    future = pd.Series(future_vals, index=future_idx, name="forecast")

    return preds_test, future, f"SARIMAX{best_order}(AIC={best_aic:.1f})"

def make_lstm_dataset(values: np.ndarray, lookback: int = 20):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X)[:, :, None] 
    y = np.array(y)
    return X, y

def forecast_lstm(train: pd.Series, test: pd.Series, horizon: int = 30, lookback: int = 20):
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass
    tf.random.set_seed(42)
    np.random.seed(42)

    # scale
    tr = train.values.astype("float32")
    mu = float(tr.mean())
    sd = float(tr.std()) if float(tr.std()) > 1e-9 else 1.0
    tr_s = (tr - mu) / sd

    X_train, y_train = make_lstm_dataset(tr_s, lookback=lookback)

    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
    X_train,
    y_train,
    epochs=10,          
    batch_size=32,
    verbose=0
    )

    # предсказание на тесте
    full_values = np.concatenate([train.values, test.values]).astype("float32")
    preds_test = []
    for i in range(len(test)):
        start = len(train) + i - lookback
        end = len(train) + i
        window = full_values[start:end]
        window_s = (window - mu) / sd
        X = window_s.reshape(1, lookback, 1)
        yhat_s = model.predict(X, verbose=0)[0, 0]
        yhat = yhat_s * sd + mu
        preds_test.append(yhat)

    preds_test = np.array(preds_test, dtype=float)

    # предсказание будущего с использованием полученной в ходе обучения информации
    future_idx = pd.bdate_range(start=pd.concat([train, test]).index[-1], periods=horizon + 1)[1:]
    history = full_values.copy().astype("float32")
    future_preds = []
    for i in range(horizon):
        window = history[-lookback:]
        window_s = (window - mu) / sd
        X = window_s.reshape(1, lookback, 1)
        yhat_s = float(model.predict(X, verbose=0)[0, 0])
        yhat = yhat_s * sd + mu
        future_preds.append(float(yhat))
        history = np.append(history, yhat)

    return preds_test, pd.Series(future_preds, index=future_idx, name="forecast"), f"LSTM(lookback={lookback})"

def select_best_model(train: pd.Series, test: pd.Series, horizon: int = 30): # выбор лучшей модели
    start_train = time.time()
    candidates = []
    # Rigde модель
    p_test, f, name = forecast_ridge(train, test, horizon=horizon)
    candidates.append((name, p_test, f))

    # SARIMAX
    p_test, f, name = forecast_sarimax(train, test, horizon=horizon)
    candidates.append((name, p_test, f))

    # LSTM
    p_test, f, name = forecast_lstm(train, test, horizon=horizon, lookback=20)
    candidates.append((name, p_test, f))

    # оценка модели
    metrics_table = []
    for name, p_test, _f in candidates:
        metrics_table.append({
            "model": name,
            "rmse": rmse(test.values, p_test),
            "mape": mape(test.values, p_test),
            "mae": mae(test.values, p_test)
        })

    # выбор лучшей модели по метрике RMSE
    best_idx = int(np.argmin([m["rmse"] for m in metrics_table]))
    best_row = metrics_table[best_idx]
    best_name = best_row["model"]

    best_test_pred = candidates[best_idx][1]
    best_forecast = candidates[best_idx][2]

    train_time = time.time() - start_train

    return (
        best_name,
        float(best_row["rmse"]),
        float(best_row["mape"]),
        float(best_row["mae"]),
        best_forecast,
        best_test_pred,
        metrics_table,
        float(train_time),
    ) 