import matplotlib.pyplot as plt
import pandas as pd

def plot_history_and_forecast(history: pd.Series, forecast: pd.Series, path: str = "forecasts/forecast.png"):
    hist = history.sort_index()
    fc = forecast.sort_index()

    # добавляем последнюю историческую точку в начало прогноза
    fc_plot = pd.concat([
        pd.Series([hist.iloc[-1]], index=[hist.index[-1]], name="forecast"),
        fc
    ])

    # берем только последние 120 торговых дней для наглядности
    hist_plot = hist.iloc[-120:]

    plt.figure(figsize=(10, 5))
    plt.plot(hist_plot.index, hist_plot.values, label="History")
    plt.plot(fc_plot.index, fc_plot.values, label="Forecast")
    plt.axvline(hist_plot.index[-1], linestyle="--", linewidth=1)

    plt.title("Stock price forecast for next 30 trading days")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

    return path