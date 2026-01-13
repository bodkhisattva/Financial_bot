import os
import time
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.types import Message, FSInputFile
from aiogram.filters import CommandStart, Command
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from utils import (
    fetch_prices,
    train_test_split_series,
    parse_ticker,
    parse_amount,
    append_log,
    DataDownloadError,
    RateLimitError,
)
from models import select_best_model
from recommender import simulate_strategy
from plots import plot_history_and_forecast

# чтобы не ловить 429
LAST_REQ_BY_USER = {}
GLOBAL_COOLDOWN = 60  # секунд

BOT_TOKEN = "ВАШ_TELEGRAM_TOKEN"


class Form(StatesGroup):
    ticker = State()
    amount = State()


dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("Привет! Введите тикер компании (например, AAPL):")
    await state.set_state(Form.ticker)


@dp.message(Command("help"))
async def help_command(message: Message):
    await message.answer(
        "Помощь по боту\n\n"
        "Как пользоваться:\n"
        "1) /start\n"
        "2) Введите тикер (AAPL, MSFT, GAZP.ME)\n"
        "3) Введите сумму (например 10000)\n\n"
        "Что я делаю:\n"
        "• Загружаю данные за 2 года с Yahoo Finance\n"
        "• Обучаю 3 модели и выбираю лучшую по метрикам\n"
        "• Строю прогноз на 30 торговых дней\n"
        "• Даю рекомендации о покупке/продаже и считаю условную прибыль\n\n"
        "Учебный проект. Не финансовая рекомендация."
    )


@dp.message(Form.ticker)
async def get_ticker(message: Message, state: FSMContext):
    try:
        t = parse_ticker(message.text)
    except Exception as e:
        await message.answer(f"Ошибка: {e}\nПопробуйте снова, пример: AAPL")
        return

    await state.update_data(ticker=t)
    await message.answer("Введите сумму для условной инвестиции (например, 10000):")
    await state.set_state(Form.amount)


@dp.message(Form.amount)
async def get_amount(message: Message, state: FSMContext):
    start_total = time.time()

    try:
        amount = parse_amount(message.text)
    except Exception as e:
        await message.answer(f"Ошибка: {e}\nВведите число, например 10000")
        return

    data = await state.get_data()
    ticker = data["ticker"]
    user_id = message.from_user.id if message.from_user else 0

    now = time.time()
    prev = LAST_REQ_BY_USER.get(user_id, 0.0)
    if now - prev < GLOBAL_COOLDOWN:
        await message.answer(
            f"Подождите {GLOBAL_COOLDOWN} секунд между запросами — Yahoo Finance ограничивает частые запросы."
        )
        return
    LAST_REQ_BY_USER[user_id] = now

    await message.answer("Загружаю данные и обучаю модели… (это может занять немного времени)")

    try:
        series = fetch_prices(ticker, history_days=730)
        train, test = train_test_split_series(series, test_size=60)

        (
            best_name,
            best_rmse,
            best_mape,
            best_mae,
            forecast,
            best_test_pred,
            metrics_table,
            train_time,
        ) = select_best_model(train, test, horizon=30)

        current_price = float(series.iloc[-1])
        future_last = float(forecast.iloc[-1])
        change_pct = (future_last / current_price - 1.0) * 100.0

        profit, mins, maxs, actions = simulate_strategy(forecast, amount, order=1)

        plot_path = plot_history_and_forecast(series, forecast, path="forecast.png")
        await message.answer_photo(FSInputFile(plot_path))

        metrics_lines = [
            f"- {row['model']}: RMSE={row['rmse']:.3f}, MAPE={row['mape']:.2f}%, MAE={row['mae']:.3f}"
            for row in metrics_table
        ]

        rec_lines = ["Рекомендации (по прогнозу):"]
        if not actions:
            rec_lines.append("- Нет выраженных локальных минимумов/максимумов на горизонте.")
        else:
            for dt, act, price in actions[:10]:
                rec_lines.append(f"- {dt.date()} {act} @ {price:.2f}")
            if len(actions) > 10:
                rec_lines.append(f"... и ещё {len(actions) - 10} действий")

        text = (
            f"Тикер: {ticker}\n"
            f"Лучшая модель: {best_name}\n\n"
            f"Метрики лучшей модели (тест):\n"
            f"RMSE={best_rmse:.3f}\n"
            f"MAPE={best_mape:.2f}%\n"
            f"MAE={best_mae:.3f}\n\n"
            f"Сейчас: {current_price:.2f}\n"
            f"Прогноз (конец горизонта 30 торговых дней): {future_last:.2f}\n"
            f"Ожидаемое изменение: {change_pct:+.2f}%\n\n"
            f"Сравнение моделей:\n" + "\n".join(metrics_lines) + "\n\n"
            + "\n".join(rec_lines) + "\n\n"
            f"Условная прибыль по стратегии: {profit:+.2f} (на сумму {amount:.2f})\n"
            f"Время обучения моделей: {train_time:.2f} сек\n\n"
            "Учебный проект. Не является финансовой рекомендацией."
        )

        await message.answer(text)

        total_time = time.time() - start_total
        append_log(
            user_id=user_id,
            ticker=ticker,
            amount=amount,
            best_model=best_name,
            rmse_val=best_rmse,
            mape_val=best_mape,
            mae_val=best_mae,
            profit=profit,
            train_time=train_time,
            total_time=total_time,
        )

    except RateLimitError as e:
        await message.answer(
            f"{e}\n\n"
            "Попробуйте подождать 2–10 минут и повторить /start.\n"
            "Также помогает смена сети/VPN сервера (смена IP)."
        )
    except DataDownloadError as e:
        await message.answer(
            f"{e}\n\n"
            "Обычно это связано с сетью/VPN/ограничениями Yahoo Finance, а не с тикером.\n"
            "Попробуйте другую сеть или повторите позже."
        )
    except Exception as e:
        await message.answer(f"Ошибка при обработке: {e}\nПопробуйте снова /start")
    finally:
        await state.clear()


@dp.message()
async def unknown_message(message: Message):
    await message.answer(
        "Я не понимаю эту команду.\n\n"
        "Доступные команды:\n"
        "• /start — начать работу\n"
        "• /help — справка по боту\n\n"
        "Следуйте подсказкам бота после /start"
    )


async def main():
    bot = Bot(BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())