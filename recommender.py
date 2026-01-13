import numpy as np
import pandas as pd

def local_extrema(series: pd.Series, order: int = 2):
    # возвращает список (дата, значение) минимумов и максимумов с попомщью простого сравнения с соседями
    s = series.values.astype(float)
    idx = series.index
    mins, maxs = [], []
    for i in range(order, len(s) - order):
        left = s[i - order:i]
        right = s[i + 1:i + order + 1]
        center = s[i]
        if center < np.min(left) and center < np.min(right):
            mins.append((idx[i], float(center)))
        if center > np.max(left) and center > np.max(right):
            maxs.append((idx[i], float(center)))
    return mins, maxs

def simulate_strategy(forecast: pd.Series, amount: float, order: int = 2):
    # покупка в каждом локальном минимуме за деньги или за ценные бумаги на локальном максимуме
    mins, maxs = local_extrema(forecast, order=order)
    min_dates = set([d for d, _ in mins])
    max_dates = set([d for d, _ in maxs])

    cash = float(amount)
    shares = 0.0
    actions = []

    for dt, price in forecast.items():
        price = float(price)
        if dt in min_dates and shares == 0.0 and cash > 0:
            shares = cash / price
            actions.append((dt, "Покупать", price))
            cash = 0.0
        elif dt in max_dates and shares > 0.0:
            cash = shares * price
            actions.append((dt, "Продавать", price))
            shares = 0.0

    if shares > 0.0:
        last_dt = forecast.index[-1]
        last_price = float(forecast.iloc[-1])
        cash = shares * last_price
        actions.append((last_dt, "SELL_END", last_price))
        shares = 0.0

    profit = cash - float(amount)
    return float(profit), mins, maxs, actions