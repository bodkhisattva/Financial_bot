import os
import csv
from datetime import datetime, timezone, timedelta
import time
import random

import numpy as np
import pandas as pd
import yfinance as yf
import requests

LOG_FILE = "logs.csv" # логирование


class DataDownloadError(Exception):
    # Yahoo Finance недоступен / пустой ответ / проблемы сети
    pass


class RateLimitError(DataDownloadError):
    # статус 429
    pass


def cache_path(ticker: str) -> str:
    os.makedirs("cache", exist_ok=True)
    safe = ticker.replace("/", "_")
    return os.path.join("cache", f"{safe}.csv")


def save_cache(ticker: str, series: pd.Series):
    series.to_frame(name="price").to_csv(cache_path(ticker), index=True)


def load_cache(ticker: str) -> pd.Series | None:
    p = cache_path(ticker)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    if df.empty or "price" not in df.columns:
        return None
    s = df["price"].dropna()
    return s if not s.empty else None


def _yahoo_quote_ping(ticker: str, timeout: int = 10) -> int:
    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={ticker}"
    r = requests.get(url, timeout=timeout)
    return r.status_code


def fetch_prices(
    ticker: str,
    history_days: int = 730,
    retries: int = 3,          
    base_sleep: float = 2.0,
    max_sleep: float = 20.0,  
) -> pd.Series:

    # Сначала пробуем кэш
    cached = load_cache(ticker)
    if cached is not None and len(cached) > 0:
        return cached

    end_date = datetime.now()
    start_date = end_date - timedelta(days=history_days)

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                threads=False,
            )

            if data is None or data.empty:
                # проверяем 429, пингуем (не каждый раз)
                try:
                    status = _yahoo_quote_ping(ticker, timeout=10)
                    if status == 429:
                        raise RateLimitError("HTTP 429 Too Many Requests")
                except RateLimitError as e:
                    last_error = e
                    raise
                except Exception:
                    pass

                raise DataDownloadError("Пустой ответ от Yahoo Finance (data.empty).")

            if "Close" not in data.columns:
                raise DataDownloadError("В ответе нет колонки Close.")

            s = data["Close"].dropna()
            if s.empty:
                raise DataDownloadError("Close пустой после dropna().")

            s.name = "price"
            save_cache(ticker, s)
            return s

        except RateLimitError as e:
            last_error = e
            sleep = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            sleep = sleep * (0.8 + 0.4 * random.random())
            time.sleep(sleep)

        except DataDownloadError as e:
            last_error = e
            sleep = min(max_sleep, base_sleep * attempt)
            time.sleep(sleep)

    raise DataDownloadError(
        "Не удалось загрузить данные с Yahoo Finance и нет локального кэша.\n"
        f"Последняя ошибка: {last_error}"
    )


def train_test_split_series(s: pd.Series, test_size: int = 60):
    if len(s) <= test_size + 30:
        raise ValueError("Ряд слишком короткий для корректного разбиения.")
    train = s.iloc[:-test_size]
    test = s.iloc[-test_size:]
    return train, test


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def parse_ticker(text: str) -> str:
    t = text.strip().upper().replace(" ", "")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    if not t or any(ch not in allowed for ch in t) or len(t) > 20:
        raise ValueError("Тикер выглядит некорректно. Пример: AAPL, MSFT, TSLA.")
    return t


def parse_amount(text: str) -> float:
    x = text.strip().replace(",", ".")
    val = float(x)
    if val <= 0:
        raise ValueError("Сумма должна быть > 0.")
    return val


def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_utc",
                "user_id",
                "ticker",
                "amount",
                "best_model",
                "rmse",
                "mape",
                "mae",
                "potential_profit",
                "train_time_sec",
                "total_request_time_sec",
            ])


def append_log(
    user_id: int,
    ticker: str,
    amount: float,
    best_model: str,
    rmse_val: float,
    mape_val: float,
    mae_val: float,
    profit: float,
    train_time: float,
    total_time: float,
):
    init_log()
    ts = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            ts, user_id, ticker, amount, best_model,
            rmse_val, mape_val, mae_val,
            profit, train_time, total_time,
        ])